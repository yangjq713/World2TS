"""AnySearch web search source — unified multi-provider news/web search.

Uses the AnySearch Search API (https://anysearch.com) — a gateway that routes a
query across multiple providers, fuses and re-ranks the results. Returns
web_snippet quality events. Uses system curl for HTTP (proxy-compatible).

Built to the same standard as the Tavily source (multi-key resolution, file
config, key pool with failover + cooldown), with three AnySearch-specific
adaptations, all handled here:

1. **No real date filter.** AnySearch's `constraint.from/to` is accepted but does
   not actually restrict results to a historical window (verified empirically:
   identical results with and without the window, and invalid dates return 200).
   So the search() date window is NOT sent to the API — the caller must put the
   time period in the *query text* (e.g. "... March 2015"), and this source
   POST-FILTERS results to ``[start, end]`` using the date it extracts from each
   result's URL/content. The coarse `freshness` enum is recency-only and unused.

2. **No publication date in results.** Result items carry no `published_at`/`date`
   field, so timestamps are recovered entirely from the URL / title / content via
   the shared extractor reused from the Tavily source (many results carry
   ``/YYYY/MM/DD/`` URL paths, and day-precision hits are upgraded inline by
   fetching the article's publication metadata). Results with no extractable
   in-window date are dropped.

3. **Transient 503s.** AnySearch returns ``503``/``50304`` ("capability
   temporarily unavailable") intermittently; the search loop retries the same key
   a few times on transient errors before failing over to the next key.

Auth: `Authorization: Bearer <key>` when a key is configured (`anysearch.toml`,
`anysearch_api_key(s)`, or `ANYSEARCH_API_KEY(S)`); otherwise anonymous mode
(allowed by AnySearch, lower quota).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from world2ts_core.config import get as config_get
from world2ts_core.errors import SourceError
from world2ts_core.events import DiscoveredEvent
from world2ts_core.http_curl import curl_post as _curl_post_base
from world2ts_core.registry import EventSourceCapability, register_event
from world2ts_core.sources.base import EventSource

# Reuse the Tavily source's generic helpers and date-extraction logic. Despite
# the names these are generic (Tavily-specific date fields are simply absent
# from AnySearch items, so the extractor falls through to URL then text).
from world2ts_core.sources.tavily import (
    _as_int,
    _dedupe_strings,
    _extract_tavily_timestamp as _extract_timestamp,
    _mask_key,
    _split_key_values,
    fetch_url_publication_timestamp,
)

logger = logging.getLogger(__name__)

_ANYSEARCH_URL = "https://api.anysearch.com/v1/search"
_DEFAULT_ANYSEARCH_CONFIG_PATH = Path("anysearch.toml")
_TRANSIENT_TOKENS = ("503", "502", "504", "50304", "temporarily unavailable", "timeout", "timed out")


def load_anysearch_file_config() -> dict:
    configured = str(
        config_get("anysearch_config_file", default="")
        or os.environ.get("ANYSEARCH_CONFIG_FILE", "")
    ).strip()
    path = Path(configured) if configured else _DEFAULT_ANYSEARCH_CONFIG_PATH
    if not path.is_file():
        return {}
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _anysearch_file_setting(file_config: dict, local_key: str, legacy_key: str, default):
    if local_key in file_config:
        return file_config[local_key]
    if legacy_key in file_config:
        return file_config[legacy_key]
    return config_get(legacy_key, default=default)


def resolve_anysearch_api_keys(
    *,
    api_key: str = "",
    api_keys: list[str] | str | None = None,
) -> list[str]:
    """Resolve AnySearch API keys. Empty list => anonymous mode (allowed)."""
    explicit = _split_key_values(api_keys)
    if api_key:
        explicit.insert(0, api_key)
    if explicit:
        return _dedupe_strings(explicit)

    file_config = load_anysearch_file_config()
    keys: list[str] = []
    keys.extend(_split_key_values(file_config.get("api_keys", [])))
    keys.extend(_split_key_values(file_config.get("anysearch_api_keys", [])))
    for env_name in _split_key_values(file_config.get("api_key_envs", [])):
        keys.extend(_split_key_values(os.environ.get(env_name, "")))
    keys.extend(_split_key_values(file_config.get("api_key", "")))
    keys.extend(_split_key_values(file_config.get("anysearch_api_key", "")))
    keys.extend(_split_key_values(config_get("anysearch_api_keys", default=[])))
    for env_name in _split_key_values(config_get("anysearch_api_key_envs", default=[])):
        keys.extend(_split_key_values(os.environ.get(env_name, "")))
    keys.extend(_split_key_values(config_get("anysearch_api_key", default="")))
    keys.extend(_split_key_values(os.environ.get("ANYSEARCH_API_KEYS", "")))
    keys.extend(_split_key_values(os.environ.get("ANYSEARCH_API_KEY", "")))
    return _dedupe_strings(keys)


def has_anysearch_api_key_configured() -> bool:
    """True if an explicit key is configured. (AnySearch also runs anonymously.)"""
    return bool(resolve_anysearch_api_keys())


def _is_transient(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in _TRANSIENT_TOKENS)


@dataclass
class _AnySearchKeyState:
    api_key: str
    cooldown_until: float = 0.0
    invalid: bool = False


class AnySearchKeyPool:
    """Select AnySearch API keys with round-robin + cooldown failover.

    Simpler than TavilyKeyPool: AnySearch has no /usage endpoint, so there is no
    usage-based routing — keys are tried round-robin, cooled down on
    rate/quota/transient failures, and disabled on auth failures. An empty key
    list means anonymous mode (a single keyless slot).
    """

    def __init__(
        self,
        api_keys: list[str],
        *,
        cooldown_seconds: int = 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._states = [_AnySearchKeyState(api_key=key) for key in _dedupe_strings(api_keys)]
        self._cooldown_seconds = max(int(cooldown_seconds), 0)
        self._clock = clock
        self._round_robin_index = 0

    @property
    def size(self) -> int:
        return len(self._states)

    @property
    def anonymous(self) -> bool:
        return not self._states

    def select_key(self, *, exclude: set[str] | None = None) -> str:
        exclude = exclude or set()
        if not self._states:
            if "" in exclude:
                raise SourceError("No AnySearch endpoint available (anonymous attempt already failed)")
            return ""  # anonymous request
        now = self._clock()
        candidates = [
            state
            for state in self._states
            if state.api_key not in exclude and not state.invalid and state.cooldown_until <= now
        ]
        if not candidates:
            raise SourceError("No AnySearch API key available (all cooled down or invalid)")
        selected = candidates[self._round_robin_index % len(candidates)]
        self._round_robin_index += 1
        return selected.api_key

    def record_success(self, api_key: str) -> None:  # noqa: D401 - parity with TavilyKeyPool
        return None

    def mark_failure(self, api_key: str, exc: Exception) -> None:
        state = self._state_for_key(api_key)
        if state is None:
            return
        message = str(exc).lower()
        if any(token in message for token in ("401", "403", "unauthorized", "forbidden", "invalid api key")):
            state.invalid = True
            return
        if any(token in message for token in ("429", "quota", "limit", "rate")) or _is_transient(exc):
            state.cooldown_until = self._clock() + self._cooldown_seconds

    def _state_for_key(self, api_key: str) -> _AnySearchKeyState | None:
        for state in self._states:
            if state.api_key == api_key:
                return state
        return None


def _curl_post(body: dict, headers: list[str]) -> str:
    """POST to AnySearch API. Extracted for mocking in tests."""
    return _curl_post_base(_ANYSEARCH_URL, body, headers=headers)


@register_event
class AnySearchSource(EventSource):
    """AnySearch unified web search source. Implements EventSource.search()."""

    capability = EventSourceCapability(
        name="anysearch",
        kind="search",
        domains=["finance", "crypto", "weather", "health", "macro", "energy", "transport", "sales", "agriculture", "forex", "pharma", "semiconductors"],
        entity_types=["stock", "crypto", "country", "weather_station", "macro_indicator", "company"],
        event_types=["news", "web"],
        geo=["global"],
        history_days=3650,
        time_resolution="day",
        rate_per_min=20,
        auth_required=False,
        reliability="medium",
        description="AnySearch unified search API — multi-provider news/web retrieval. No server-side date filter: put the time period in the query text; results are post-filtered to the window by URL/content date.",
        query_format="keyword",
        required_params=["query"],
        optional_params=["start", "end", "geo", "top_k"],
        output_fields=["title", "url", "description", "timestamp", "retrieval_score"],
    )

    def __init__(self, api_key: str = "", api_keys: list[str] | str | None = None):
        resolved_keys = resolve_anysearch_api_keys(api_key=api_key, api_keys=api_keys)
        file_config = load_anysearch_file_config()
        cooldown_seconds = (
            _as_int(_anysearch_file_setting(file_config, "cooldown_seconds", "anysearch_cooldown_seconds", 60))
            or 60
        )
        self._transient_retries = (
            _as_int(_anysearch_file_setting(file_config, "transient_retries", "anysearch_transient_retries", 2))
            if _as_int(_anysearch_file_setting(file_config, "transient_retries", "anysearch_transient_retries", 2)) is not None
            else 2
        )
        try:
            self._retry_backoff = float(
                _anysearch_file_setting(file_config, "retry_backoff_seconds", "anysearch_retry_backoff_seconds", 1.0)
            )
        except (TypeError, ValueError):
            self._retry_backoff = 1.0
        self._key_pool = AnySearchKeyPool(resolved_keys, cooldown_seconds=cooldown_seconds)

    def search(
        self,
        query: str,
        start: datetime,
        end: datetime,
        geo: str = "",
        top_k: int = 10,
    ) -> List[DiscoveredEvent]:
        start_day = start.date()
        end_day = end.date()
        # NOTE: no `constraint` date window is sent — it is a no-op on AnySearch.
        # The caller carries the period in the query text; we post-filter below.
        body = {
            "query": query,
            "max_results": max(int(top_k), 1),
            "content_types": ["news", "web"],
        }

        raw = ""
        tried_keys: set[str] = set()
        last_error: Exception | None = None
        for _ in range(max(self._key_pool.size, 1)):
            try:
                api_key = self._key_pool.select_key(exclude=tried_keys)
            except SourceError as exc:
                last_error = exc
                break
            tried_keys.add(api_key)
            headers = [f"Authorization: Bearer {api_key}"] if api_key else []
            for attempt in range(self._transient_retries + 1):
                try:
                    raw = _curl_post(body, headers)
                    self._key_pool.record_success(api_key)
                    break
                except (SourceError, Exception) as exc:
                    last_error = exc
                    if _is_transient(exc) and attempt < self._transient_retries:
                        logger.warning("AnySearch transient error (retrying): %s", str(exc)[:120])
                        time.sleep(self._retry_backoff)
                        continue
                    self._key_pool.mark_failure(api_key, exc)
                    logger.warning(
                        "AnySearch search failed for query %r with key %s: %s",
                        query,
                        _mask_key(api_key) if api_key else "<anonymous>",
                        exc,
                    )
                    break
            if raw:
                break

        if not raw:
            # Every key/attempt failed. Fail loudly so the pipeline records a
            # source failure instead of silently treating this as "no event".
            if last_error is not None:
                raise SourceError(f"AnySearch search failed for query {query!r}: {last_error}")
            raise SourceError(f"AnySearch search returned no response for query {query!r}")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SourceError(f"AnySearch response is not valid JSON for query {query!r}: {exc}") from exc

        # Success envelope: {"code":0,"message":"success","data":{"results":[...],"metadata":{...}}}
        if not isinstance(payload, dict) or payload.get("code") != 0 or "data" not in payload:
            detail = payload.get("message") if isinstance(payload, dict) else str(payload)[:200]
            raise SourceError(f"AnySearch error for query {query!r}: {detail}")

        results = (payload.get("data") or {}).get("results", []) or []
        events: List[DiscoveredEvent] = []

        for rank, item in enumerate(results):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            content = str(item.get("content") or "")
            description = str(item.get("description") or "")
            score = float(item.get("score") or 0.0)
            if score > 1.0:  # AnySearch scores are 0-100; normalise toward Tavily's 0-1
                score = score / 100.0

            if not title:
                continue

            # AnySearch returns no publication date; recover it from URL/title/content.
            probe = {"title": title, "content": f"{content}\n{description}", "url": url}
            ts, date_source, date_value, timestamp_precision = _extract_timestamp(probe)
            if ts is None and url:
                # (A) No date in the result itself (AnySearch has no published_at,
                # and many URLs lack a /YYYY/MM/DD/ date path). Before dropping the
                # candidate, fetch the article's HTML publication metadata to try to
                # anchor it to a real date. Retrospective articles will resolve to a
                # later publication date and be filtered out by the window below.
                enriched_ts, enriched_value = fetch_url_publication_timestamp(url)
                if enriched_ts is not None:
                    ts = enriched_ts
                    date_source = "url_metadata"
                    date_value = enriched_value
                    timestamp_precision = "exact"
            if ts is None:
                continue
            prior_timestamp = ts
            prior_date_source = date_source
            prior_date_value = date_value
            if timestamp_precision == "day" and url:
                enriched_ts, enriched_value = fetch_url_publication_timestamp(url)
                if enriched_ts is not None and start_day <= enriched_ts.date() <= end_day:
                    ts = enriched_ts
                    date_source = "url_metadata"
                    date_value = enriched_value
                    timestamp_precision = "exact"

            # No server-side date filter — enforce the window ourselves.
            if ts.date() < start_day or ts.date() > end_day:
                continue

            eid = hashlib.md5(f"anysearch_{url}_{ts.date().isoformat()}".encode()).hexdigest()[:12]
            structured_payload = {
                "date_source": date_source,
                "date_value": date_value,
            }
            if date_source == "url_metadata":
                structured_payload.update({
                    "timestamp_kind": "publication_time",
                    "prior_timestamp": prior_timestamp.isoformat(),
                    "prior_date_source": prior_date_source,
                    "prior_date_value": prior_date_value,
                })

            events.append(DiscoveredEvent(
                event_id=f"anysearch_{eid}",
                title=title,
                description=content if content.strip() else (description if description.strip() else title),
                timestamp=ts,
                timestamp_precision=timestamp_precision,
                source_name="anysearch",
                source_kind="news_search",
                evidence_quality="web_snippet",
                structured_payload=structured_payload,
                url=url,
                retrieval_rank=rank,
                retrieval_score=score,
            ))

        return events

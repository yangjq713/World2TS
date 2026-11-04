"""Tavily web search source — general-purpose news and web search.

Uses the Tavily Search API (https://tavily.com) to retrieve recent web
articles for any query. Returns web_snippet quality events.

Requires TAVILY_API_KEY/TAVILY_API_KEYS, `tavily_api_key`,
`tavily_api_keys`, or `tavily_api_key_envs`. Uses system curl for HTTP
(proxy-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from world2ts_core.config import get as config_get
from world2ts_core.errors import SourceError
from world2ts_core.events import DiscoveredEvent
from world2ts_core.http_curl import curl_get as _curl_get_base
from world2ts_core.http_curl import curl_post as _curl_post_base
from world2ts_core.registry import EventSourceCapability, register_event
from world2ts_core.sources.base import EventSource

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_TAVILY_USAGE_URL = "https://api.tavily.com/usage"
_DEFAULT_TAVILY_CONFIG_PATH = Path("tavily.toml")
_ISO_DATE_RE = re.compile(r"\b(?P<year>20\d{2})[-_/](?P<month>\d{1,2})[-_/](?P<day>\d{1,2})\b")
_URL_MIXED_DATE_RE = re.compile(r"/(?P<year>20\d{2})-(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/|$)")
_MONTH_DATE_RE = re.compile(
    r"\b(?P<month_name>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(?P<day>\d{1,2}),\s+(?P<year>20\d{2})\b",
    re.IGNORECASE,
)
_DATE_LABEL = r"(?:published|updated|posted|last updated|date|release date)"
_LABELED_ISO_DATE_RE = re.compile(
    r"\b" + _DATE_LABEL + r"\s*(?:on|at)?\s*[:,-]?\s*"
    r"(?P<year>20\d{2})[-_/](?P<month>\d{1,2})[-_/](?P<day>\d{1,2})\b",
    re.IGNORECASE,
)
_LABELED_MONTH_DATE_RE = re.compile(
    r"\b" + _DATE_LABEL + r"\s*(?:on|at)?\s*[:,-]?\s*"
    r"(?P<month_name>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(?P<day>\d{1,2}),\s+(?P<year>20\d{2})\b",
    re.IGNORECASE,
)
_EXACT_TIME_HINT_RE = re.compile(r"(?:T|\s)\d{1,2}:\d{2}")
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR_RE = re.compile(r"([a-zA-Z_:.-]+)\s*=\s*([\"'])(.*?)\2", re.DOTALL)
_TIME_DATETIME_RE = re.compile(
    r"<time\b[^>]*\bdatetime\s*=\s*([\"'])(?P<value>.*?)\1",
    re.IGNORECASE | re.DOTALL,
)
_JSON_DATE_PUBLISHED_RE = re.compile(
    r'"datePublished"\s*:\s*"(?P<value>[^"]+)"',
    re.IGNORECASE,
)
_PUBLISHED_META_NAMES = {
    "article:published_time",
    "og:published_time",
    "datepublished",
    "date.published",
    "publishdate",
    "pubdate",
    "parsely-pub-date",
    "sailthru.date",
}
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _curl_post(body: dict, api_key: str = "") -> str:
    """POST to Tavily API. Extracted for mocking in tests."""
    headers = [f"Authorization: Bearer {api_key}"] if api_key else []
    return _curl_post_base(_TAVILY_URL, body, headers=headers)


def _curl_usage(api_key: str) -> str:
    """Fetch Tavily usage for one API key. Extracted for mocking in tests."""
    return _curl_get_base(
        _TAVILY_USAGE_URL,
        timeout=15,
        retries=1,
        headers=[f"Authorization: Bearer {api_key}"],
    )


def _curl_get(url: str) -> str:
    """Fetch a result URL for publication metadata. Extracted for tests.

    Keep this short: metadata fetching is a best-effort enrichment per Tavily
    result, and a few slow article pages should not stall an offline batch.
    """
    return _curl_get_base(url, timeout=5, retries=1, headers=["User-Agent: Mozilla/5.0"])


@dataclass
class _TavilyKeyState:
    api_key: str
    usage: int | None = None
    limit: int | None = None
    remaining: int | None = None
    last_usage_check: float = 0.0
    cooldown_until: float = 0.0
    invalid: bool = False


class TavilyKeyPool:
    """Select Tavily API keys using cached usage metadata."""

    def __init__(
        self,
        api_keys: list[str],
        *,
        usage_cache_seconds: int = 300,
        min_remaining_credits: int = 1,
        usage_fetcher: Callable[[str], str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not api_keys:
            raise ValueError(
                "Tavily API key not found. Set TAVILY_API_KEY, TAVILY_API_KEYS, "
                "`tavily_api_key`, `tavily_api_keys`, or `tavily_api_key_envs`."
            )
        self._states = [_TavilyKeyState(api_key=key) for key in _dedupe_strings(api_keys)]
        self._usage_cache_seconds = max(int(usage_cache_seconds), 0)
        self._min_remaining_credits = max(int(min_remaining_credits), 0)
        self._usage_fetcher = usage_fetcher or _curl_usage
        self._clock = clock
        self._round_robin_index = 0

    @property
    def size(self) -> int:
        return len(self._states)

    def select_key(self, *, exclude: set[str] | None = None) -> str:
        exclude = exclude or set()
        now = self._clock()
        self._refresh_usage(now)
        candidates = [
            state
            for state in self._states
            if (
                state.api_key not in exclude
                and not state.invalid
                and state.cooldown_until <= now
                and (state.remaining is None or state.remaining >= self._min_remaining_credits)
            )
        ]
        if not candidates:
            raise SourceError("No Tavily API key has available quota")
        known = [state for state in candidates if state.remaining is not None]
        if known:
            return max(known, key=lambda state: state.remaining or 0).api_key
        selected = candidates[self._round_robin_index % len(candidates)]
        self._round_robin_index += 1
        return selected.api_key

    def record_success(self, api_key: str, *, cost: int = 1) -> None:
        state = self._state_for_key(api_key)
        if state and state.remaining is not None:
            state.remaining = max(0, state.remaining - max(cost, 0))
            if state.usage is not None:
                state.usage += max(cost, 0)

    def mark_failure(self, api_key: str, exc: Exception) -> None:
        state = self._state_for_key(api_key)
        if state is None:
            return
        message = str(exc).lower()
        if any(token in message for token in ("401", "unauthorized", "forbidden", "invalid api key")):
            state.invalid = True
            return
        if any(token in message for token in ("429", "rate", "quota", "limit")):
            state.cooldown_until = self._clock() + 60

    def _refresh_usage(self, now: float) -> None:
        if len(self._states) < 2:
            return
        for state in self._states:
            if state.invalid or state.cooldown_until > now:
                continue
            if state.last_usage_check and now - state.last_usage_check < self._usage_cache_seconds:
                continue
            try:
                raw = self._usage_fetcher(state.api_key)
                payload = json.loads(raw)
            except Exception as exc:
                logger.debug("Tavily usage check failed for key %s: %s", _mask_key(state.api_key), exc)
                state.last_usage_check = now
                continue
            usage, limit = _usage_and_limit_from_payload(payload)
            if usage is not None and limit is not None:
                state.usage = usage
                state.limit = limit
                state.remaining = max(0, limit - usage)
            state.last_usage_check = now

    def _state_for_key(self, api_key: str) -> _TavilyKeyState | None:
        for state in self._states:
            if state.api_key == api_key:
                return state
        return None


def _normalize_datetime(dt: datetime) -> datetime:
    """Convert a tz-aware datetime to a UTC-naive one.

    Reject naked datetimes loudly: we cannot tell which timezone an unmarked
    "2018-12-09 16:38:36" was written in (commonly ET on US news sites), so
    silently treating it as UTC would introduce a ~5h offset on hourly data.
    Callers that wrap us in `except ValueError` will gracefully fall through to
    the day-precision parsing path, which is correct regardless of timezone.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "datetime has no tzinfo; refusing to assume UTC. "
            "Caller should fall back to day precision."
        )
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_and_limit_from_payload(payload: dict) -> tuple[int | None, int | None]:
    if not isinstance(payload, dict):
        return None, None
    key_usage = payload.get("key") if isinstance(payload.get("key"), dict) else {}
    account_usage = payload.get("account") if isinstance(payload.get("account"), dict) else {}
    key_limit = _as_int(key_usage.get("limit"))
    if key_limit is not None:
        return _as_int(key_usage.get("usage")), key_limit
    account_limit = _as_int(account_usage.get("plan_limit"))
    if account_limit is not None:
        account_plan_usage = _as_int(account_usage.get("plan_usage"))
        return account_plan_usage, account_limit
    return _as_int(key_usage.get("usage")), None


def _split_key_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_split_key_values(item))
        return result
    text = str(value).strip()
    if not text:
        return []
    for separator in ("\n", ";"):
        text = text.replace(separator, ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _mask_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


def load_tavily_file_config() -> dict:
    configured = str(
        config_get("tavily_config_file", default="")
        or os.environ.get("TAVILY_CONFIG_FILE", "")
    ).strip()
    path = Path(configured) if configured else _DEFAULT_TAVILY_CONFIG_PATH
    if not path.is_file():
        return {}
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _tavily_file_setting(file_config: dict, local_key: str, legacy_key: str, default):
    if local_key in file_config:
        return file_config[local_key]
    if legacy_key in file_config:
        return file_config[legacy_key]
    return config_get(legacy_key, default=default)


def resolve_tavily_api_keys(
    *,
    api_key: str = "",
    api_keys: list[str] | str | None = None,
) -> list[str]:
    explicit = _split_key_values(api_keys)
    if api_key:
        explicit.insert(0, api_key)
    if explicit:
        return _dedupe_strings(explicit)

    file_config = load_tavily_file_config()
    keys: list[str] = []
    keys.extend(_split_key_values(file_config.get("api_keys", [])))
    keys.extend(_split_key_values(file_config.get("tavily_api_keys", [])))
    for env_name in _split_key_values(file_config.get("api_key_envs", [])):
        keys.extend(_split_key_values(os.environ.get(env_name, "")))
    for env_name in _split_key_values(file_config.get("tavily_api_key_envs", [])):
        keys.extend(_split_key_values(os.environ.get(env_name, "")))
    keys.extend(_split_key_values(file_config.get("api_key", "")))
    keys.extend(_split_key_values(file_config.get("tavily_api_key", "")))
    keys.extend(_split_key_values(config_get("tavily_api_keys", default=[])))
    for env_name in _split_key_values(config_get("tavily_api_key_envs", default=[])):
        keys.extend(_split_key_values(os.environ.get(env_name, "")))
    keys.extend(_split_key_values(config_get("tavily_api_key", default="")))
    keys.extend(_split_key_values(os.environ.get("TAVILY_API_KEYS", "")))
    keys.extend(_split_key_values(os.environ.get("TAVILY_API_KEY", "")))
    return _dedupe_strings(keys)


def has_tavily_api_key_configured() -> bool:
    return bool(resolve_tavily_api_keys())


def _parse_exact_datetime(value: str) -> Optional[datetime]:
    cleaned = html.unescape(value or "").strip()
    if not cleaned or not _EXACT_TIME_HINT_RE.search(cleaned):
        return None
    normalized = cleaned.replace("Z", "+00:00")
    try:
        return _normalize_datetime(datetime.fromisoformat(normalized))
    except ValueError:
        pass
    try:
        return _normalize_datetime(parsedate_to_datetime(cleaned))
    except (TypeError, ValueError):
        return None


def _parse_tavily_date(s: Optional[str]) -> Optional[datetime]:
    """Parse Tavily date strings without inventing a timestamp."""
    if not s:
        return None
    value = s.strip()
    exact = _parse_exact_datetime(value)
    if exact is not None:
        return exact
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m/%d", "%Y_%m_%d"):
        try:
            return datetime.strptime(value[:10], fmt)
        except ValueError:
            pass
    try:
        return _normalize_datetime(parsedate_to_datetime(value))
    except (TypeError, ValueError):
        return None


def _timestamp_precision_from_value(value: str) -> str:
    return "exact" if _parse_exact_datetime(value) is not None else "day"


def _date_from_parts(year: str, month: str, day: str) -> Optional[datetime]:
    try:
        return datetime(int(year), int(month), int(day))
    except ValueError:
        return None


def _extract_tavily_timestamp(item: dict) -> tuple[Optional[datetime], str, str, str]:
    """Extract the best available event/publication date from a Tavily result."""
    tavily_date_fields = ("published_date", "publication_date", "published_at", "date", "timestamp")
    for key in tavily_date_fields:
        value = str(item.get(key) or "")
        ts = _parse_exact_datetime(value)
        if ts is not None:
            return ts, key, value, "exact"

    for key in tavily_date_fields:
        value = str(item.get(key) or "")
        ts = _parse_tavily_date(value)
        if ts is not None:
            return ts, key, value, _timestamp_precision_from_value(value)

    url = str(item.get("url") or "")
    for regex in (_URL_MIXED_DATE_RE, _ISO_DATE_RE):
        match = regex.search(url)
        if match:
            ts = _date_from_parts(match.group("year"), match.group("month"), match.group("day"))
            if ts is not None:
                return ts, "url", match.group(0).strip("/"), "day"

    text = "\n".join(
        str(item.get(key) or "")
        for key in ("title", "content", "raw_content")
    )
    for match in _LABELED_ISO_DATE_RE.finditer(text):
        ts = _date_from_parts(match.group("year"), match.group("month"), match.group("day"))
        if ts is not None:
            return ts, "text", match.group(0), "day"
    for match in _LABELED_MONTH_DATE_RE.finditer(text):
        month = _MONTHS.get(match.group("month_name")[:3].lower())
        if month is None:
            continue
        ts = _date_from_parts(match.group("year"), str(month), match.group("day"))
        if ts is not None:
            return ts, "text", match.group(0), "day"

    return None, "", "", ""


def extract_publication_timestamp_from_html(raw_html: str) -> tuple[Optional[datetime], str]:
    """Extract exact publication time from common article metadata."""
    for tag_match in _META_TAG_RE.finditer(raw_html or ""):
        attrs = {
            name.lower(): html.unescape(value).strip()
            for name, _, value in _ATTR_RE.findall(tag_match.group(0))
        }
        field_name = (
            attrs.get("property")
            or attrs.get("name")
            or attrs.get("itemprop")
            or ""
        ).lower()
        content = attrs.get("content", "")
        if field_name in _PUBLISHED_META_NAMES and content:
            ts = _parse_exact_datetime(content)
            if ts is not None:
                return ts, content

    for match in _JSON_DATE_PUBLISHED_RE.finditer(raw_html or ""):
        value = html.unescape(match.group("value")).strip()
        ts = _parse_exact_datetime(value)
        if ts is not None:
            return ts, value

    for match in _TIME_DATETIME_RE.finditer(raw_html or ""):
        value = html.unescape(match.group("value")).strip()
        ts = _parse_exact_datetime(value)
        if ts is not None:
            return ts, value

    return None, ""


def fetch_url_publication_timestamp(
    url: str,
    *,
    fetcher=None,
) -> tuple[Optional[datetime], str]:
    """Fetch a URL and extract exact publication time when available."""
    if not url:
        return None, ""
    fetch = fetcher or _curl_get
    try:
        raw_html = fetch(url)
    except Exception as exc:
        logger.debug("URL metadata fetch failed for %s: %s", url, exc)
        return None, ""
    return extract_publication_timestamp_from_html(raw_html)


@register_event
class TavilySource(EventSource):
    """Tavily web search source. Implements EventSource.search()."""

    capability = EventSourceCapability(
        name="tavily",
        kind="search",
        domains=["finance", "crypto", "weather", "health", "macro", "energy", "transport", "sales", "agriculture", "forex", "pharma", "semiconductors"],
        entity_types=["stock", "crypto", "country", "weather_station", "macro_indicator", "company"],
        event_types=["news", "web"],
        geo=["global"],
        history_days=3650,
        time_resolution="day",
        rate_per_min=60,
        auth_required=True,
        reliability="medium",
        description="Tavily web search API — general-purpose news and web retrieval via Bearer auth.",
        query_format="keyword",
        required_params=["query"],
        optional_params=["start", "end", "geo", "top_k"],
        output_fields=["title", "url", "description", "timestamp", "retrieval_score"],
    )

    def __init__(self, api_key: str = "", api_keys: list[str] | str | None = None):
        resolved_keys = resolve_tavily_api_keys(api_key=api_key, api_keys=api_keys)
        file_config = load_tavily_file_config()
        usage_cache_seconds = (
            _as_int(_tavily_file_setting(file_config, "usage_cache_seconds", "tavily_usage_cache_seconds", 300))
            or 300
        )
        min_remaining_credits = (
            _as_int(_tavily_file_setting(file_config, "min_remaining_credits", "tavily_min_remaining_credits", 1))
            or 1
        )
        self._key_pool = TavilyKeyPool(
            resolved_keys,
            usage_cache_seconds=usage_cache_seconds,
            min_remaining_credits=min_remaining_credits,
        )

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
        body = {
            "query": query,
            "search_depth": "basic",
            "max_results": top_k,
            "start_date": start_day.isoformat(),
            "end_date": (end_day + timedelta(days=1)).isoformat(),
            "include_answer": False,
            "include_raw_content": True,
        }

        raw = ""
        tried_keys: set[str] = set()
        last_error: Exception | None = None
        for _ in range(self._key_pool.size):
            try:
                api_key = self._key_pool.select_key(exclude=tried_keys)
            except SourceError as exc:
                last_error = exc
                break
            tried_keys.add(api_key)
            try:
                raw = _curl_post(body, api_key=api_key)
                self._key_pool.record_success(api_key, cost=1)
                break
            except (SourceError, Exception) as exc:
                last_error = exc
                self._key_pool.mark_failure(api_key, exc)
                logger.warning(
                    "Tavily search failed for query %r with key %s: %s",
                    query,
                    _mask_key(api_key),
                    exc,
                )
        if not raw:
            # Every key failed (quota/auth/HTTP error). Fail loudly so the
            # pipeline records a source failure instead of silently treating
            # this window as "no event exists".
            if last_error is not None:
                raise SourceError(f"Tavily search failed for query {query!r}: {last_error}")
            raise SourceError(f"Tavily search returned no response for query {query!r}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SourceError(f"Tavily response is not valid JSON for query {query!r}: {exc}") from exc

        if "results" not in data:
            # A 2xx body without a `results` key is an error envelope
            # (e.g. {"detail": {"error": ...}}), not a zero-result search.
            detail = data.get("detail") or data
            raise SourceError(f"Tavily response has no 'results' for query {query!r}: {str(detail)[:200]}")

        results = data.get("results", [])
        events: List[DiscoveredEvent] = []

        for rank, item in enumerate(results):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            content = str(item.get("content") or "")
            score = float(item.get("score") or 0.0)

            if not title:
                continue

            ts, date_source, date_value, timestamp_precision = _extract_tavily_timestamp(item)
            if ts is None and url:
                # No date from the Tavily result itself (date fields/URL/text all
                # empty — common for older or evergreen-URL articles). Before
                # dropping, fetch the article's HTML publication metadata to try to
                # anchor it. Retrospective articles resolve to a later publication
                # date and are filtered out by the window check below.
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
                if (
                    enriched_ts is not None
                    and start_day <= enriched_ts.date() <= end_day
                ):
                    ts = enriched_ts
                    date_source = "url_metadata"
                    date_value = enriched_value
                    timestamp_precision = "exact"
            if ts.date() < start_day or ts.date() > end_day:
                continue
            eid = hashlib.md5(f"tavily_{url}_{ts.date().isoformat()}".encode()).hexdigest()[:12]
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
                event_id=f"tavily_{eid}",
                title=title,
                description=content if content.strip() else title,
                timestamp=ts,
                timestamp_precision=timestamp_precision,
                source_name="tavily",
                source_kind="news_search",
                evidence_quality="web_snippet",
                structured_payload=structured_payload,
                url=url,
                retrieval_rank=rank,
                retrieval_score=score,
            ))

        return events

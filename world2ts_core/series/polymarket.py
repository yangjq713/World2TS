"""Polymarket prediction market price series.

Fetches historical probability prices from Polymarket's CLOB API.
Each market is a binary question (e.g., "Will Fed cut rates?")
whose price (0-1) represents the market's probability estimate.

Price shifts in prediction markets have natural event semantics —
the market question itself describes what event drives the series.

CLOB API: https://clob.polymarket.com/prices-history
Gamma API: https://gamma-api.polymarket.com (market discovery)
No authentication required.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlencode

from world2ts_core.series.model import SeriesRecord
from world2ts_core.series.base import SeriesSource
from world2ts_core.http_curl import curl_get
from world2ts_core.registry import SeriesSourceCapability, register_series

_CLOB_BASE = "https://clob.polymarket.com"
_GAMMA_BASE = "https://gamma-api.polymarket.com"
_RATE_LIMIT = 1.0  # 1 req/sec to be safe


def _curl_get(url: str, timeout: int = 30) -> str:
    """GET via system curl with proxy support."""
    return curl_get(url, timeout=timeout)


def discover_markets(query: str = "", tag: str = "", limit: int = 20) -> List[dict]:
    """Search Gamma API for markets. Returns list of market metadata dicts."""
    params = [f"limit={limit}", "active=true", "closed=false"]
    if query:
        url = f"{_GAMMA_BASE}/public-search?query={query}&limit={limit}"
    else:
        url = f"{_GAMMA_BASE}/markets?{'&'.join(params)}"
        if tag:
            url += f"&tag={tag}"

    raw = _curl_get(url)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Polymarket market discovery returned invalid JSON") from exc

    # Handle different response formats
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        markets = data.get("markets", data.get("results", []))
        if not isinstance(markets, list):
            raise ValueError("Polymarket market discovery response must contain a list")
        return markets
    raise ValueError("Polymarket market discovery response must be a list or object")


@register_series
class PolymarketSource(SeriesSource):
    """Fetch prediction market price history from Polymarket CLOB API."""

    capability = SeriesSourceCapability(
        name="polymarket",
        domains=["polymarket", "finance", "macro", "crypto"],
        entity_types=["prediction_market"],
        frequencies=["1h", "1d"],
        geo=["global"],
        history_days=365,
        rate_per_min=60,
        auth_required=False,
        reliability="medium",
        description=(
            "Polymarket — prediction market probabilities. Each series is a binary "
            "question whose price (0-1) reflects market consensus probability. "
            "Shifts naturally correspond to real-world events."
        ),
        primary_target_fields=["probability"],
        target_option="",
        adapter_options={},
        metadata_dimensions=["question", "slug", "category", "outcomes", "token_id"],
        timestamp_semantics="UTC",
        range_semantics="[start, end)",
    )

    def __init__(self, interval: str = "1d"):
        if interval not in {"1h", "1d"}:
            raise ValueError("Polymarket interval must be '1h' or '1d'")
        self._interval = interval
        self._last_request = 0.0

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < _RATE_LIMIT:
            time.sleep(_RATE_LIMIT - elapsed)
        self._last_request = time.time()

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        """Fetch price history for each market token ID in symbols.

        Symbols are Polymarket token/condition IDs. Use discover_markets()
        to find relevant IDs for a topic.
        """
        records = []
        for token_id in symbols:
            rec = self._fetch_one(token_id, start, end, domain)
            if rec:
                records.append(rec)
        return records

    def _fetch_one(
        self, token_id: str, start: datetime, end: datetime, domain: str,
    ) -> Optional[SeriesRecord]:
        self._rate_limit()

        start_utc = (
            start.replace(tzinfo=timezone.utc)
            if start.tzinfo is None
            else start.astimezone(timezone.utc)
        )
        end_utc = (
            end.replace(tzinfo=timezone.utc)
            if end.tzinfo is None
            else end.astimezone(timezone.utc)
        )
        start_ts = int(start_utc.timestamp())
        end_ts = int(end_utc.timestamp())
        fidelity = 60 if self._interval == "1h" else 1440

        # Use interval=max for full history; date filtering done client-side
        # because API returns empty for resolved markets when startTs/endTs
        # don't exactly match the market's active period
        url = (
            f"{_CLOB_BASE}/prices-history"
            f"?market={token_id}"
            f"&interval=max"
            f"&fidelity={fidelity}"
        )

        raw = _curl_get(url)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Polymarket price response for token {token_id!r} is not valid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"Polymarket price response for token {token_id!r} must be an object"
            )

        history = data.get("history", [])
        if not isinstance(history, list):
            raise ValueError(
                f"Polymarket price response for token {token_id!r} has invalid history"
            )
        if not history:
            return None

        values_by_timestamp: dict[int, float] = {}
        for index, point in enumerate(history):
            if not isinstance(point, dict):
                raise ValueError(
                    f"Polymarket price response for token {token_id!r} has "
                    f"invalid history point {index}"
                )
            ts = point.get("t", 0)
            price = point.get("p")
            if ts and price is not None:
                try:
                    timestamp = int(ts)
                    value = float(price)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Polymarket price response for token {token_id!r} has "
                        f"invalid history point {index}"
                    ) from exc
                if start_ts <= timestamp < end_ts:
                    values_by_timestamp[timestamp] = value

        if not values_by_timestamp:
            return None

        ordered_timestamps = sorted(values_by_timestamp)
        timestamps = [datetime.utcfromtimestamp(ts) for ts in ordered_timestamps]
        values = [values_by_timestamp[ts] for ts in ordered_timestamps]

        metadata = self._get_market_meta(token_id)

        return SeriesRecord(
            series_id=token_id,
            domain=domain,
            entity=metadata["question"],
            geo="",
            freq=self._interval,
            start=timestamps[0],
            end=timestamps[-1],
            values=values,
            timestamps=timestamps,
            metadata={
                "source": "polymarket",
                "token_id": token_id,
                "question": metadata["question"],
                "slug": metadata.get("slug", ""),
                "category": metadata.get("category", ""),
                "outcomes": metadata.get("outcomes", []),
            },
        )

    def _get_market_meta(self, token_id: str) -> dict:
        """Fetch the unique Gamma market identified by a CLOB token ID."""
        query = urlencode({"clob_token_ids": token_id})
        raw = _curl_get(f"{_GAMMA_BASE}/markets?{query}", timeout=10)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Polymarket metadata response for token {token_id!r} is not valid JSON"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Polymarket metadata response for token {token_id!r} must be a list"
            )
        if len(data) != 1:
            raise ValueError(
                "Polymarket metadata lookup must return exactly one market for "
                f"token {token_id!r}; got {len(data)}"
            )
        market = data[0]
        if not isinstance(market, dict):
            raise ValueError(
                f"Polymarket metadata for token {token_id!r} must be an object"
            )
        question = market.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Polymarket metadata for token {token_id!r} has no non-empty question"
            )
        return {**market, "question": question.strip()}

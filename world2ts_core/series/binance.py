"""CoinGecko API series source.

Fetches cryptocurrency market data from CoinGecko's free API.
No authentication required. Works from China without proxy.

Replaces Binance (blocked in China with HTTP 451).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List

from world2ts_core.http_curl import curl_get
from world2ts_core.series.model import SeriesRecord
from world2ts_core.registry import SeriesSourceCapability, register_series

_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
_RATE_LIMIT_SLEEP = 2.0  # CoinGecko free tier: 10-50 calls/min


def _api_get(url: str) -> dict:
    """Fetch JSON from CoinGecko via system curl (LibreSSL)."""
    import json

    raw = curl_get(url, timeout=30)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("CoinGecko returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("CoinGecko response must be a JSON object")
    return payload


@register_series
class CoinGeckoSource:
    """Fetch crypto price data from CoinGecko public API.

    Uses the absolute ``market_chart/range`` endpoint and selects the last
    observation in each requested UTC hour or day. Hourly requests are limited
    to 90 days. Symbols are CoinGecko coin IDs (e.g. ``bitcoin``, ``ethereum``).

    Parameters
    ----------
    interval:
        Desired frequency: ``"1h"`` (default) or ``"1d"``.
    """

    capability = SeriesSourceCapability(
        name="coingecko",
        domains=["crypto"],
        entity_types=["crypto"],
        frequencies=["1h", "1d"],
        geo=["global"],
        history_days=365,
        rate_per_min=30,
        auth_required=False,
        reliability="medium",
        description="CoinGecko — cryptocurrency prices (hourly up to 90d, daily up to 365d)",
        primary_target_fields=["price"],
        target_option="",
        adapter_options={},
        metadata_dimensions=["coin_id", "interval"],
        timestamp_semantics="UTC",
        range_semantics="[start, end)",
    )

    def __init__(self, interval: str = "1h"):
        if interval not in {"1h", "1d"}:
            raise ValueError("CoinGecko interval must be '1h' or '1d'")
        self._interval = interval

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        records: List[SeriesRecord] = []

        for symbol in symbols:
            result = self._fetch_symbol(symbol, start, end, domain)
            if result is not None:
                records.append(result)

            if len(symbols) > 1:
                time.sleep(_RATE_LIMIT_SLEEP)

        return records

    def _fetch_symbol(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        domain: str,
    ) -> SeriesRecord | None:
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
        if end_utc <= start_utc:
            raise ValueError("CoinGecko end must be after start")
        if self._interval == "1h" and end_utc - start_utc > timedelta(days=90):
            raise ValueError("CoinGecko hourly collection supports at most 90 days")

        url = (
            f"{_MARKET_CHART_URL.format(id=symbol)}/range"
            f"?vs_currency=usd"
            f"&from={int(start_utc.timestamp())}"
            f"&to={int(end_utc.timestamp())}"
        )

        data = _api_get(url)

        if "error" in data:
            raise RuntimeError(
                f"CoinGecko price fetch failed for {symbol}: {data['error']}"
            )
        if "prices" not in data:
            return None

        prices = data["prices"]
        if not isinstance(prices, list):
            raise RuntimeError(
                f"CoinGecko price fetch failed for {symbol}: prices must be a list"
            )
        if not prices:
            return None

        selected_by_bucket: dict[object, tuple[int, datetime, float]] = {}
        for point in prices:
            if not isinstance(point, list) or len(point) < 2:
                continue
            timestamp_ms = int(point[0])
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            if not start_utc <= timestamp < end_utc:
                continue
            bucket = (
                timestamp.replace(minute=0, second=0, microsecond=0)
                if self._interval == "1h"
                else timestamp.date()
            )
            selected_by_bucket[bucket] = (timestamp_ms, timestamp, float(point[1]))

        selected = sorted(selected_by_bucket.values(), key=lambda item: item[1])
        if not selected:
            return None

        selected_timestamp_ms = [item[0] for item in selected]
        timestamps = [item[1].replace(tzinfo=None) for item in selected]
        values = [item[2] for item in selected]

        # Multi-variate: prices, market_caps, total_volumes
        columns = {"price": values}
        for key in ("market_caps", "total_volumes"):
            values_by_timestamp = {
                int(point[0]): float(point[1])
                for point in data.get(key, [])
                if isinstance(point, list) and len(point) >= 2
            }
            if all(
                timestamp_ms in values_by_timestamp
                for timestamp_ms in selected_timestamp_ms
            ):
                columns[key] = [
                    values_by_timestamp[timestamp_ms]
                    for timestamp_ms in selected_timestamp_ms
                ]

        return SeriesRecord(
            series_id=symbol,
            domain=domain,
            entity=symbol,
            geo="",
            freq=self._interval,
            start=timestamps[0],
            end=timestamps[-1],
            values=values,
            timestamps=timestamps,
            columns=columns,
            metadata={
                "source": "coingecko",
                "coin_id": symbol,
                "interval": self._interval,
            },
        )

"""EIA (US Energy Information Administration) — energy price time series."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from typing import Any, List, Optional

from world2ts_core.series.model import SeriesRecord
from world2ts_core.series.base import SeriesSource
from world2ts_core.config import get as config_get
from world2ts_core.registry import SeriesSourceCapability, register_series

_BASE_URL = "https://api.eia.gov/v2"


def _curl_get(url: str, timeout: int = 30) -> str:
    cmd = [
        "/usr/bin/curl", "-f", "-sS", "-g", "--max-time", str(timeout),
    ]
    proxy = config_get("http_proxy", default="")
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(url)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 5
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConnectionError(f"EIA curl execution failed: {exc}") from exc
    if result.returncode != 0:
        raise ConnectionError(
            f"curl failed (code {result.returncode}): {result.stderr[:200]}"
        )
    if not result.stdout.strip():
        raise ConnectionError("EIA returned an empty response body")
    return result.stdout


# EIA series routing: symbol -> (route, facet_name, facet_value, data_field)
_SERIES_MAP = {
    # Petroleum spot prices
    "RWTC": ("petroleum/pri/spt/data", "series", "RWTC", "value"),      # WTI Crude Oil
    "RBRTE": ("petroleum/pri/spt/data", "series", "RBRTE", "value"),    # Brent Crude Oil
    # Natural gas
    "RNGWHHD": ("natural-gas/pri/fut/data", "series", "RNGWHHD", "value"),  # Henry Hub
}


@register_series
class EiaSource(SeriesSource):
    """Fetch energy time series from EIA API."""

    capability = SeriesSourceCapability(
        name="eia",
        domains=["energy"],
        entity_types=["commodity", "energy_market"],
        frequencies=["1d"],
        geo=["US"],
        history_days=3650,
        rate_per_min=60,
        auth_required=True,
        reliability="high",
        description="EIA — US daily crude-oil and natural-gas prices",
        primary_target_fields=["value"],
        target_option="",
        adapter_options={},
        metadata_dimensions=["eia_series", "eia_route"],
        timestamp_semantics="US civil dates interpreted in plan timezone",
        range_semantics="[start, end)",
    )

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    def _get_key(self) -> str:
        import os
        key = self._api_key or config_get("eia_api_key", default="")
        return key or os.environ.get("WORLD2TS_EIA_API_KEY", "")

    @staticmethod
    def primary_target_field(symbol: str) -> str:
        if symbol not in _SERIES_MAP:
            raise ValueError(
                f"Unsupported EIA series {symbol!r}; choose an explicitly registered series"
            )
        return _SERIES_MAP[symbol][3]

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        key = self._get_key()
        if not key:
            raise ValueError(
                "EIA API key not configured. Set eia_api_key in world2ts.toml "
                "or WORLD2TS_EIA_API_KEY env var. Get a free key at "
                "https://www.eia.gov/opendata/register.php"
            )

        records: List[SeriesRecord] = []
        for symbol in symbols:
            records.append(self._fetch_one(symbol, start, end, domain, key))
        return records

    def _fetch_one(
        self, symbol: str, start: datetime, end: datetime, domain: str, key: str,
    ) -> SeriesRecord:
        self.primary_target_field(symbol)
        route, facet_name, facet_value, data_field = _SERIES_MAP[symbol]

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        url = (
            f"{_BASE_URL}/{route}"
            f"?api_key={key}"
            f"&frequency=daily"
            f"&data[0]={data_field}"
            f"&facets[{facet_name}][]={facet_value}"
            f"&start={start_str}"
            f"&end={end_str}"
            f"&sort[0][column]=period"
            f"&sort[0][direction]=asc"
            f"&length=5000"
        )

        try:
            raw = _curl_get(url)
            data = json.loads(raw)
        except (ConnectionError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"EIA fetch failed for {symbol}: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                f"EIA fetch failed for {symbol}: response must be a JSON object"
            )
        response = data.get("response")
        if not isinstance(response, dict):
            detail = data.get("error") or data.get("message") or "missing response"
            raise RuntimeError(f"EIA fetch failed for {symbol}: {detail}")
        rows = response.get("data")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"EIA fetch failed for {symbol}: response.data must be a list"
            )
        if not rows:
            raise RuntimeError(f"EIA returned no data for {symbol}")

        values: List[float] = []
        timestamps: List[datetime] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"EIA returned malformed row {row_index} for {symbol}: "
                    "expected a JSON object"
                )
            period = row.get("period")
            if not isinstance(period, str) or not period.strip():
                raise RuntimeError(
                    f"EIA returned malformed row {row_index} for {symbol}: "
                    "period must be a non-empty string"
                )
            if data_field not in row:
                raise RuntimeError(
                    f"EIA returned malformed row {row_index} for {symbol}: "
                    f"missing {data_field!r}"
                )
            value: Any = row[data_field]
            if value is None:
                continue
            try:
                timestamp = datetime.strptime(period, "%Y-%m-%d")
            except ValueError as exc:
                raise RuntimeError(
                    f"EIA returned malformed row {row_index} for {symbol}: "
                    f"invalid daily period {period!r}"
                ) from exc
            if not start <= timestamp < end:
                continue
            if isinstance(value, bool):
                raise RuntimeError(
                    f"EIA returned malformed row {row_index} for {symbol}: "
                    f"invalid {data_field!r} value {value!r}"
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"EIA returned malformed row {row_index} for {symbol}: "
                    f"invalid {data_field!r} value {value!r}"
                ) from exc
            if not math.isfinite(numeric_value):
                raise RuntimeError(
                    f"EIA returned malformed row {row_index} for {symbol}: "
                    f"non-finite {data_field!r} value {value!r}"
                )
            values.append(numeric_value)
            timestamps.append(timestamp)

        if not values:
            raise RuntimeError(
                f"EIA returned no usable in-range data for {symbol}"
            )

        # Infer frequency from data gaps
        if len(timestamps) >= 2:
            avg_gap = (timestamps[-1] - timestamps[0]).days / (len(timestamps) - 1)
            if avg_gap > 25:
                freq = "1mo"
            elif avg_gap > 5:
                freq = "1w"
            else:
                freq = "1d"
        else:
            freq = "1d"

        # Entity name from series description
        desc = response.get("description", symbol)

        return SeriesRecord(
            series_id=symbol,
            domain=domain,
            entity=desc if isinstance(desc, str) else symbol,
            geo="US",
            freq=freq,
            start=timestamps[0],
            end=timestamps[-1],
            values=values,
            timestamps=timestamps,
            metadata={
                "source": "eia",
                "eia_series": facet_value,
                "eia_route": route,
            },
        )

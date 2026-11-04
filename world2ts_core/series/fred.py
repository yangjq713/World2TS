"""FRED API series source.

Federal Reserve Economic Data — 800+ macroeconomic time series.
Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

from world2ts_core.series.model import SeriesRecord
from world2ts_core.config import get as config_get
from world2ts_core.registry import SeriesSourceCapability, register_series


_FRED_FREQUENCIES = {
    "daily": "1d",
    "weekly": "1w",
    "monthly": "1mo",
    "quarterly": "1q",
}


@register_series
class FredSource:
    """Fetch economic data from FRED API."""

    capability = SeriesSourceCapability(
        name="fred",
        domains=["macro"],
        entity_types=["macro_indicator"],
        frequencies=["1d", "1w", "1mo", "1q"],
        geo=["US"],
        history_days=36500,
        rate_per_min=120,
        auth_required=True,
        reliability="high",
        description="FRED — 800+ US macroeconomic time series (GDP, unemployment, CPI, interest rates)",
        primary_target_fields=["observation"],
        target_option="",
        adapter_options={},
        metadata_dimensions=[
            "series_id_fred",
            "title",
            "units",
            "seasonal_adjustment",
        ],
        timestamp_semantics="civil dates interpreted in plan timezone",
        range_semantics="[start, end)",
    )

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    def _get_client(self):
        import fredapi
        key = (
            self._api_key
            or config_get("fred_api_key")
            or os.environ.get("WORLD2TS_FRED_API_KEY")
        )
        if not key:
            raise ValueError(
                "FRED API key not configured. Set fred_api_key in world2ts.toml "
                "or WORLD2TS_FRED_API_KEY env var."
            )
        return fredapi.Fred(api_key=key)

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        client = self._get_client()
        records: List[SeriesRecord] = []

        for series_id in symbols:
            try:
                data = client.get_series(series_id, start, end)
            except Exception as exc:
                raise RuntimeError(
                    f"FRED series fetch failed for {series_id}: {exc}"
                ) from exc

            if not isinstance(data, pd.Series):
                raise RuntimeError(
                    f"FRED series fetch failed for {series_id}: "
                    "get_series must return a pandas Series"
                )

            if data.empty:
                continue

            data = data.dropna()
            data = data[
                (data.index >= pd.Timestamp(start))
                & (data.index < pd.Timestamp(end))
            ]
            if data.empty:
                continue

            values = data.tolist()
            timestamps = [t.to_pydatetime() for t in data.index]

            meta, freq = self._series_metadata(client, series_id)

            entity = meta["title"]

            records.append(SeriesRecord(
                series_id=series_id,
                domain=domain,
                entity=entity,
                geo="US",
                freq=freq,
                start=timestamps[0],
                end=timestamps[-1],
                values=values,
                timestamps=timestamps,
                metadata=meta,
            ))

        return records

    @staticmethod
    def _series_metadata(
        client,
        series_id: str,
    ) -> tuple[dict[str, str], str]:
        try:
            info = client.get_series_info(series_id)
            if not isinstance(info, pd.Series):
                raise ValueError(
                    "get_series_info must return a non-empty pandas Series"
                )
            if info.empty:
                raise ValueError("get_series_info returned no metadata")

            title = info.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("get_series_info returned no non-empty title")

            frequency = info.get("frequency")
            if not isinstance(frequency, str) or not frequency.strip():
                raise ValueError("get_series_info returned no non-empty frequency")
            rendered_frequency = frequency.strip()
            canonical_frequency = _FRED_FREQUENCIES.get(
                rendered_frequency.casefold()
            )
            if canonical_frequency is None:
                raise ValueError(
                    f"unsupported FRED frequency {rendered_frequency!r}; "
                    f"supported values: {', '.join(_FRED_FREQUENCIES)}"
                )

            metadata = {
                "source": "fred",
                "series_id_fred": series_id,
                "title": title.strip(),
                "frequency": rendered_frequency,
            }
            for key in ["units", "seasonal_adjustment"]:
                value = info.get(key)
                if value is not None and pd.notna(value):
                    rendered = str(value).strip()
                    if rendered:
                        metadata[key] = rendered
            return metadata, canonical_frequency
        except Exception as exc:
            raise RuntimeError(
                f"FRED metadata fetch failed for {series_id}: {exc}"
            ) from exc

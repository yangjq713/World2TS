"""Yahoo Finance series source.

Collects rich metadata so Claude Code can reason about the entity:
company name, sector, industry, market cap, etc.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from world2ts_core.config import get as config_get
from world2ts_core.series.model import SeriesRecord
from world2ts_core.registry import SeriesSourceCapability, register_series


_HISTORY_METADATA_FIELDS = {
    "shortName": "shortName",
    "longName": "longName",
    "currency": "currency",
    "exchangeName": "exchange",
    "fullExchangeName": "exchange_full_name",
    "instrumentType": "quote_type",
    "exchangeTimezoneName": "exchange_timezone",
}


# Map yfinance interval strings to World2TS frequency aliases.
_INTERVAL_TO_FREQ = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1w",
    "1mo": "1mo",
    "3mo": "1q",
}


@register_series
class YahooSource:
    """Fetch stock price data + rich metadata from Yahoo Finance."""

    capability = SeriesSourceCapability(
        name="yahoo",
        domains=["finance"],
        entity_types=["stock"],
        frequencies=["1d", "1h", "15min", "1w", "1mo"],
        geo=["global"],
        history_days=3650,
        rate_per_min=60,
        auth_required=False,
        reliability="high",
        description="Yahoo Finance chart API — market prices and exchange metadata",
        primary_target_fields=["close"],
        target_option="",
        adapter_options={},
        metadata_dimensions=["ticker", *_HISTORY_METADATA_FIELDS.values()],
        timestamp_semantics="source timezone when aware; otherwise plan timezone",
        range_semantics="[start, end)",
    )

    def __init__(self, interval: str = "1d"):
        self._interval = interval

    def fetch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        domain: str,
    ) -> List[SeriesRecord]:
        records: List[SeriesRecord] = []
        freq = _INTERVAL_TO_FREQ.get(self._interval, self._interval)
        proxy = config_get("http_proxy", default="")
        if proxy and not isinstance(proxy, str):
            raise ValueError("Yahoo http_proxy configuration must be a string")
        yf.config.network.proxy = (
            {"http": proxy, "https": proxy} if proxy else None
        )

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start,
                    end=end,
                    interval=self._interval,
                    actions=False,
                    auto_adjust=False,
                    raise_errors=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Yahoo price fetch failed for {symbol}: {exc}"
                ) from exc
            if not isinstance(df, pd.DataFrame):
                raise RuntimeError(
                    f"Yahoo price fetch failed for {symbol}: "
                    "history must return a pandas DataFrame"
                )
            if df.empty:
                raise RuntimeError(f"Yahoo returned no price data for {symbol}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                raise RuntimeError(
                    f"Yahoo returned no Close column for {symbol}"
                )

            timestamps = [t.to_pydatetime() for t in df.index]
            values = df["Close"].tolist()  # primary variable

            # Multi-variate: store OHLCV columns
            columns = {}
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    columns[col.lower()] = df[col].tolist()

            # Reuse metadata from Yahoo's chart API instead of quote-summary.
            meta = {"source": "yahoo_finance", "ticker": symbol}
            geo = ""
            entity = symbol
            try:
                history_metadata = ticker.get_history_metadata()
                if not isinstance(history_metadata, dict) or not history_metadata:
                    raise ValueError("Yahoo chart returned no metadata")
                for source_key, metadata_key in _HISTORY_METADATA_FIELDS.items():
                    value = history_metadata.get(source_key)
                    if value is not None and str(value).strip():
                        meta[metadata_key] = value
                entity = (
                    meta.get("shortName")
                    or meta.get("longName")
                    or symbol
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Yahoo metadata fetch failed for {symbol}: {exc}"
                ) from exc

            records.append(SeriesRecord(
                series_id=symbol,
                domain=domain,
                entity=entity,
                geo=geo,
                freq=freq,
                start=timestamps[0],
                end=timestamps[-1],
                values=values,
                timestamps=timestamps,
                columns=columns,
                metadata=meta,
            ))

        return records

"""Tests for CoinGecko series source (replaces Binance)."""

from datetime import datetime
from unittest.mock import patch

import pytest

from world2ts_core.series.binance import CoinGeckoSource


MOCK_MARKET_CHART = {
    "prices": [
        [1704067200000, 42300.0],
        [1704070800000, 42600.0],
        [1704074400000, 42900.0],
    ],
    "market_caps": [
        [1704067200000, 800000000000],
        [1704070800000, 810000000000],
        [1704074400000, 820000000000],
    ],
    "total_volumes": [
        [1704067200000, 20000000000],
        [1704070800000, 21000000000],
        [1704074400000, 22000000000],
    ],
}


class TestCoinGeckoSource:

    @patch("world2ts_core.series.binance._api_get")
    def test_fetch_single_coin_1h(self, mock_api_get):
        """Fetch a single coin with 1h interval: values are prices, freq='1h'."""
        mock_api_get.return_value = MOCK_MARKET_CHART

        source = CoinGeckoSource()
        records = source.fetch(
            symbols=["bitcoin"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            domain="crypto",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "bitcoin"
        assert rec.domain == "crypto"
        assert rec.freq == "1h"
        assert rec.values == [42300.0, 42600.0, 42900.0]
        assert len(rec.timestamps) == 3
        url = mock_api_get.call_args.args[0]
        assert "/market_chart/range" in url
        assert "from=1704067200" in url
        assert "to=1704153600" in url

    @patch("world2ts_core.series.binance._api_get")
    def test_fetch_daily_interval(self, mock_api_get):
        """1d interval uses market_chart without hourly param."""
        mock_api_get.return_value = MOCK_MARKET_CHART

        source = CoinGeckoSource(interval="1d")
        records = source.fetch(
            symbols=["ethereum"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 4, 1),
            domain="crypto",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.freq == "1d"
        assert rec.series_id == "ethereum"
        assert rec.values == [42900.0]

    @patch("world2ts_core.series.binance._api_get")
    def test_fetch_empty_response(self, mock_api_get):
        """Empty API response yields no records."""
        mock_api_get.return_value = {}

        source = CoinGeckoSource()
        records = source.fetch(
            symbols=["bitcoin"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            domain="crypto",
        )

        assert records == []

    @patch("world2ts_core.series.binance._api_get")
    def test_timestamps_parsed_from_ms(self, mock_api_get):
        """Timestamps are converted from milliseconds to datetime."""
        mock_api_get.return_value = MOCK_MARKET_CHART

        source = CoinGeckoSource()
        records = source.fetch(
            symbols=["bitcoin"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            domain="crypto",
        )

        assert len(records) == 1
        rec = records[0]
        # timestamp 1704067200000 ms = 2024-01-01 00:00:00 UTC
        expected_ts = datetime(2024, 1, 1, 0, 0, 0)
        assert rec.timestamps[0] == expected_ts
        assert isinstance(rec.timestamps[0], datetime)

    @patch("world2ts_core.series.binance._api_get")
    def test_metadata_set(self, mock_api_get):
        """Metadata includes source, coin_id, and interval."""
        mock_api_get.return_value = MOCK_MARKET_CHART

        source = CoinGeckoSource()
        records = source.fetch(
            symbols=["bitcoin"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            domain="crypto",
        )

        assert records[0].metadata["source"] == "coingecko"
        assert records[0].metadata["coin_id"] == "bitcoin"
        assert records[0].metadata["interval"] == "1h"

    @patch("world2ts_core.series.binance._api_get")
    def test_hourly_range_over_90_days_fails_before_request(self, mock_api_get):
        source = CoinGeckoSource(interval="1h")

        with pytest.raises(ValueError, match="at most 90 days"):
            source.fetch(
                symbols=["bitcoin"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 4, 2),
                domain="crypto",
            )

        mock_api_get.assert_not_called()

    @patch("world2ts_core.series.binance._api_get")
    def test_end_timestamp_is_excluded(self, mock_api_get):
        mock_api_get.return_value = {
            "prices": [
                [1704067200000, 42300.0],
                [1704153600000, 43000.0],
            ]
        }

        records = CoinGeckoSource().fetch(
            symbols=["bitcoin"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            domain="crypto",
        )

        assert records[0].values == [42300.0]

    @patch("world2ts_core.series.binance._api_get")
    def test_api_error_fails_loudly(self, mock_api_get):
        mock_api_get.return_value = {
            "error": {
                "status": {
                    "error_code": 10012,
                    "error_message": "historical range is not available",
                }
            }
        }

        with pytest.raises(RuntimeError, match="CoinGecko price fetch failed for bitcoin"):
            CoinGeckoSource(interval="1d").fetch(
                ["bitcoin"],
                datetime(2026, 7, 1),
                datetime(2026, 7, 10),
                "crypto",
            )

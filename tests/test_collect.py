"""Tests for series collection."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from world2ts_core.series.yahoo import YahooSource


@pytest.fixture
def mock_yfinance_data():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    df = pd.DataFrame({"Close": [185.0, 186.5, 184.0, 187.0, 188.5]}, index=dates)
    df.index.name = "Date"
    return df


class TestYahooSource:

    @patch("world2ts_core.series.yahoo.yf")
    def test_fetch_single_symbol(self, mock_yf, mock_yfinance_data):
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = mock_yfinance_data
        ticker_mock.get_history_metadata.return_value = {
            "shortName": "Apple Inc.",
            "currency": "USD",
            "exchangeName": "NMS",
            "instrumentType": "EQUITY",
            "exchangeTimezoneName": "America/New_York",
        }
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource()
        records = source.fetch(
            symbols=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 8),
            domain="finance",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "AAPL"
        assert rec.domain == "finance"
        assert rec.entity == "Apple Inc."
        assert len(rec.values) == 5
        assert len(rec.timestamps) == 5
        assert rec.freq == "1d"

    @patch("world2ts_core.series.yahoo.yf")
    def test_fetch_empty_fails_loudly(self, mock_yf):
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource()
        with pytest.raises(RuntimeError, match="no price data for INVALID"):
            source.fetch(
                ["INVALID"],
                datetime(2024, 1, 1),
                datetime(2024, 1, 8),
                "finance",
            )

    @patch("world2ts_core.series.yahoo.yf")
    def test_price_failure_is_not_silently_replaced(self, mock_yf):
        ticker_mock = MagicMock()
        ticker_mock.history.side_effect = RuntimeError("429 Too Many Requests")
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource()
        with pytest.raises(RuntimeError, match="Yahoo price fetch failed for AAPL.*429"):
            source.fetch(
                ["AAPL"],
                datetime(2024, 1, 1),
                datetime(2024, 1, 8),
                "finance",
            )

    @patch("world2ts_core.series.yahoo.yf")
    def test_metadata_failure_is_not_silently_replaced(self, mock_yf, mock_yfinance_data):
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = mock_yfinance_data
        ticker_mock.get_history_metadata.side_effect = RuntimeError(
            "metadata unavailable"
        )
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource()
        with pytest.raises(RuntimeError, match="Yahoo metadata fetch failed for AAPL"):
            source.fetch(
                ["AAPL"],
                datetime(2024, 1, 1),
                datetime(2024, 1, 8),
                "finance",
            )

    @patch("world2ts_core.series.yahoo.config_get")
    @patch("world2ts_core.series.yahoo.yf")
    def test_configured_proxy_is_forwarded_to_yfinance(
        self, mock_yf, mock_config_get, mock_yfinance_data
    ):
        mock_config_get.return_value = "http://127.0.0.1:7890"
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = mock_yfinance_data
        ticker_mock.get_history_metadata.return_value = {"currency": "USD"}
        mock_yf.Ticker.return_value = ticker_mock

        YahooSource().fetch(
            ["AAPL"],
            datetime(2024, 1, 1),
            datetime(2024, 1, 8),
            "finance",
        )

        assert mock_yf.config.network.proxy == {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890",
        }

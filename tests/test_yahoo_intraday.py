"""Tests for Yahoo Finance intraday (sub-daily) support."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from world2ts_core.series.yahoo import YahooSource


@pytest.fixture
def mock_hourly_data():
    """Hourly OHLCV data with timezone-aware DatetimeIndex."""
    dates = pd.date_range("2024-01-02 09:30", periods=6, freq="h", tz="America/New_York")
    df = pd.DataFrame({"Close": [185.0, 186.5, 184.0, 187.0, 188.5, 189.0]}, index=dates)
    df.index.name = "Datetime"
    return df


@pytest.fixture
def mock_daily_data():
    """Daily OHLCV data — standard business day index."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    df = pd.DataFrame({"Close": [185.0, 186.5, 184.0, 187.0, 188.5]}, index=dates)
    df.index.name = "Date"
    return df


class TestYahooIntraday:

    @patch("world2ts_core.series.yahoo.yf")
    def test_hourly_interval_freq(self, mock_yf, mock_hourly_data):
        """1h interval maps to freq='1h' in the resulting SeriesRecord."""
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = mock_hourly_data
        ticker_mock.get_history_metadata.return_value = {
            "shortName": "Apple Inc."
        }
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource(interval="1h")
        records = source.fetch(
            symbols=["AAPL"],
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            domain="finance",
        )

        assert len(records) == 1
        assert records[0].freq == "1h"

    @patch("world2ts_core.series.yahoo.yf")
    def test_hourly_interval_passed_to_history(self, mock_yf, mock_hourly_data):
        """interval='1h' is forwarded to the strict history request."""
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = mock_hourly_data
        ticker_mock.get_history_metadata.return_value = {
            "shortName": "Apple Inc."
        }
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource(interval="1h")
        source.fetch(
            symbols=["AAPL"],
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            domain="finance",
        )

        ticker_mock.history.assert_called_once()
        _, kwargs = ticker_mock.history.call_args
        assert kwargs.get("interval") == "1h"
        assert kwargs.get("auto_adjust") is False
        assert kwargs.get("actions") is False
        assert kwargs.get("raise_errors") is True

    @patch("world2ts_core.series.yahoo.yf")
    def test_default_daily_interval_unchanged(self, mock_yf, mock_daily_data):
        """Default YahooSource() (no args) still produces freq='1d'."""
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = mock_daily_data
        ticker_mock.get_history_metadata.return_value = {
            "shortName": "Apple Inc."
        }
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource()  # no args — default interval
        records = source.fetch(
            symbols=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 8),
            domain="finance",
        )

        assert len(records) == 1
        assert records[0].freq == "1d"
        _, kwargs = ticker_mock.history.call_args
        assert kwargs.get("interval") == "1d"

    @patch("world2ts_core.series.yahoo.yf")
    def test_15min_interval_freq(self, mock_yf):
        """15m interval maps to freq='15min'."""
        dates = pd.date_range("2024-01-02 09:30", periods=4, freq="15min", tz="America/New_York")
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0]}, index=dates)
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = df
        ticker_mock.get_history_metadata.return_value = {
            "shortName": "SPDR S&P 500 ETF Trust"
        }
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource(interval="15m")
        records = source.fetch(
            symbols=["SPY"],
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            domain="finance",
        )

        assert len(records) == 1
        assert records[0].freq == "15min"

    @patch("world2ts_core.series.yahoo.yf")
    def test_weekly_interval_freq(self, mock_yf):
        """1wk interval maps to freq='1w'."""
        dates = pd.date_range("2024-01-01", periods=4, freq="W")
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0]}, index=dates)
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = df
        ticker_mock.get_history_metadata.return_value = {
            "shortName": "SPDR S&P 500 ETF Trust"
        }
        mock_yf.Ticker.return_value = ticker_mock

        source = YahooSource(interval="1wk")
        records = source.fetch(
            symbols=["SPY"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 2, 1),
            domain="finance",
        )

        assert len(records) == 1
        assert records[0].freq == "1w"

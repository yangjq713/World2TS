"""Tests for FRED API series source."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from world2ts_core.series.fred import FredSource


@pytest.fixture
def mock_fred_data():
    dates = pd.date_range("2023-01-01", periods=8, freq="QS")
    return pd.Series(
        [25.5, 25.7, 25.9, 26.1, 26.3, 26.5, 26.7, 26.9],
        index=dates,
        name="GDP",
    )


@pytest.fixture
def mock_fred_info():
    return pd.Series({
        "id": "GDP",
        "title": "Gross Domestic Product",
        "frequency": "Quarterly",
        "units": "Billions of Dollars",
        "seasonal_adjustment": "Seasonally Adjusted Annual Rate",
    })


class TestFredSource:

    @patch("fredapi.Fred")
    def test_fetch_single_series(self, MockFred, mock_fred_data, mock_fred_info):
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.return_value = mock_fred_info
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        records = source.fetch(
            symbols=["GDP"],
            start=datetime(2023, 1, 1),
            end=datetime(2024, 12, 31),
            domain="macro",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "GDP"
        assert rec.domain == "macro"
        assert len(rec.values) == 8
        assert rec.metadata["source"] == "fred"
        assert rec.metadata["title"] == "Gross Domestic Product"
        assert rec.entity == "Gross Domestic Product"

    def test_capability_excludes_reserved_frequency_dimension(self):
        assert "frequency" not in FredSource.capability.metadata_dimensions

    @patch("fredapi.Fred")
    def test_fetch_empty(self, MockFred):
        mock_client = MagicMock()
        mock_client.get_series.return_value = pd.Series(dtype=float)
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        records = source.fetch(["INVALID"], datetime(2024, 1, 1), datetime(2024, 6, 1), "macro")
        assert records == []

    @patch("fredapi.Fred")
    def test_fetch_uses_config_api_key(self, MockFred, monkeypatch):
        monkeypatch.setenv("WORLD2TS_FRED_API_KEY", "env_key")
        mock_client = MagicMock()
        mock_client.get_series.return_value = pd.Series(dtype=float)
        MockFred.return_value = mock_client

        source = FredSource()
        source.fetch(["GDP"], datetime(2024, 1, 1), datetime(2024, 6, 1), "macro")
        MockFred.assert_called_once_with(api_key="env_key")

    @patch("fredapi.Fred")
    def test_fetch_exception_fails_loudly(self, MockFred):
        mock_client = MagicMock()
        mock_client.get_series.side_effect = Exception("API error")
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        with pytest.raises(
            RuntimeError,
            match="FRED series fetch failed for BAD_SERIES: API error",
        ):
            source.fetch(
                ["BAD_SERIES"],
                datetime(2024, 1, 1),
                datetime(2024, 6, 1),
                "macro",
            )

    @patch("fredapi.Fred")
    def test_fetch_uses_authoritative_quarterly_frequency(
        self, MockFred, mock_fred_data, mock_fred_info
    ):
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.return_value = mock_fred_info
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        records = source.fetch(["GDP"], datetime(2023, 1, 1), datetime(2024, 12, 31), "macro")

        assert len(records) == 1
        assert records[0].freq == "1q"
        assert records[0].metadata["frequency"] == "Quarterly"

    @patch("fredapi.Fred")
    def test_single_observation_uses_authoritative_weekly_frequency(self, MockFred):
        mock_client = MagicMock()
        mock_client.get_series.return_value = pd.Series(
            [4.25],
            index=pd.DatetimeIndex(["2024-01-05"]),
        )
        mock_client.get_series_info.return_value = pd.Series(
            {
                "title": "Weekly Economic Index",
                "frequency": "Weekly",
            }
        )
        MockFred.return_value = mock_client

        records = FredSource(api_key="test_key").fetch(
            ["WEI"],
            datetime(2024, 1, 1),
            datetime(2024, 1, 8),
            "macro",
        )

        assert records[0].freq == "1w"

    @patch("fredapi.Fred")
    def test_unsupported_authoritative_frequency_fails_loudly(
        self,
        MockFred,
        mock_fred_data,
    ):
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.return_value = pd.Series(
            {
                "title": "Annual Indicator",
                "frequency": "Annual",
            }
        )
        MockFred.return_value = mock_client

        with pytest.raises(
            RuntimeError,
            match="FRED metadata fetch failed for GDP: unsupported FRED frequency 'Annual'",
        ):
            FredSource(api_key="test_key").fetch(
                ["GDP"],
                datetime(2023, 1, 1),
                datetime(2024, 12, 31),
                "macro",
            )

    @patch("fredapi.Fred")
    def test_fetch_multiple_series(self, MockFred, mock_fred_data, mock_fred_info):
        """Fetching multiple symbols returns multiple records."""
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.return_value = mock_fred_info
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        records = source.fetch(
            symbols=["GDP", "UNRATE"],
            start=datetime(2023, 1, 1),
            end=datetime(2024, 12, 31),
            domain="macro",
        )

        assert len(records) == 2
        assert {r.series_id for r in records} == {"GDP", "UNRATE"}

    @patch("fredapi.Fred")
    def test_geo_is_us(self, MockFred, mock_fred_data, mock_fred_info):
        """FRED records should always have geo='US'."""
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.return_value = mock_fred_info
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        records = source.fetch(["GDP"], datetime(2023, 1, 1), datetime(2024, 12, 31), "macro")

        assert records[0].geo == "US"

    @patch("fredapi.Fred")
    def test_metadata_network_failure_is_not_silently_skipped(
        self, MockFred, mock_fred_data
    ):
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.side_effect = ConnectionError("API unavailable")
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        with pytest.raises(
            RuntimeError,
            match="FRED metadata fetch failed for GDP: API unavailable",
        ):
            source.fetch(
                ["GDP"],
                datetime(2023, 1, 1),
                datetime(2024, 12, 31),
                "macro",
            )

    @pytest.mark.parametrize(
        "metadata",
        [None, pd.Series(dtype=object), {"title": "Gross Domestic Product"}],
    )
    @patch("fredapi.Fred")
    def test_empty_or_malformed_metadata_fails_loudly(
        self, MockFred, metadata, mock_fred_data
    ):
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.return_value = metadata
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        with pytest.raises(RuntimeError, match="FRED metadata fetch failed for GDP"):
            source.fetch(
                ["GDP"],
                datetime(2023, 1, 1),
                datetime(2024, 12, 31),
                "macro",
            )

    @pytest.mark.parametrize("title", [None, "", "   "])
    @patch("fredapi.Fred")
    def test_missing_or_blank_title_fails_loudly(
        self, MockFred, title, mock_fred_data
    ):
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.return_value = pd.Series({"title": title})
        MockFred.return_value = mock_client

        source = FredSource(api_key="test_key")
        with pytest.raises(
            RuntimeError,
            match="FRED metadata fetch failed for GDP: .*non-empty title",
        ):
            source.fetch(
                ["GDP"],
                datetime(2023, 1, 1),
                datetime(2024, 12, 31),
                "macro",
            )

    @pytest.mark.parametrize("frequency", [None, "", "   "])
    @patch("fredapi.Fred")
    def test_missing_or_blank_frequency_fails_loudly(
        self,
        MockFred,
        frequency,
        mock_fred_data,
    ):
        mock_client = MagicMock()
        mock_client.get_series.return_value = mock_fred_data
        mock_client.get_series_info.return_value = pd.Series(
            {
                "title": "Gross Domestic Product",
                "frequency": frequency,
            }
        )
        MockFred.return_value = mock_client

        with pytest.raises(
            RuntimeError,
            match="FRED metadata fetch failed for GDP: .*non-empty frequency",
        ):
            FredSource(api_key="test_key").fetch(
                ["GDP"],
                datetime(2023, 1, 1),
                datetime(2024, 12, 31),
                "macro",
            )

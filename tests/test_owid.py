"""Tests for OWID COVID series source."""

from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from world2ts_core.series.owid import OwidSource, _fetch_owid_csv


MOCK_CSV = """iso_code,continent,location,date,new_cases,new_deaths,total_cases,total_deaths,new_vaccinations
USA,North America,United States,2024-01-01,50000,200,100000000,1200000,100000
USA,North America,United States,2024-01-02,48000,180,100048000,1200180,95000
USA,North America,United States,2024-01-03,52000,210,100100000,1200390,110000
GBR,Europe,United Kingdom,2024-01-01,10000,50,24000000,200000,20000
GBR,Europe,United Kingdom,2024-01-02,9500,45,24009500,200045,18000
"""


def _mock_fetch():
    return pd.read_csv(StringIO(MOCK_CSV))


@pytest.fixture(autouse=True)
def _reset_owid_cache():
    """Reset OWID module cache between tests."""
    import world2ts_core.series.owid as owid_mod
    owid_mod._cached_df = None
    yield
    owid_mod._cached_df = None


class TestOwidSource:

    @patch("subprocess.run")
    def test_download_failure_does_not_switch_endpoint(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=22,
            stderr="HTTP 503",
        )

        with pytest.raises(ConnectionError, match="OWID CSV download failed"):
            _fetch_owid_csv()

        assert mock_run.call_count == 1
        command = mock_run.call_args.args[0]
        assert command[-1] == (
            "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        )

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_fetch_single_country(self, mock_fetch):
        """Fetching one ISO code returns one SeriesRecord."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "USA"
        assert rec.domain == "health"
        assert rec.metadata["source"] == "owid_covid"
        assert rec.metadata["iso_code"] == "USA"
        assert rec.metadata["metric"] == "new_cases"

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_fetch_multiple_countries(self, mock_fetch):
        """Fetching multiple ISO codes returns one record per country."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        records = source.fetch(
            symbols=["USA", "GBR"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        assert len(records) == 2
        ids = {r.series_id for r in records}
        assert ids == {"USA", "GBR"}

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_fetch_values_correct(self, mock_fetch):
        """Values match the CSV data for the requested country."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        rec = records[0]
        assert rec.values == pytest.approx([50000.0, 48000.0, 52000.0])

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_fetch_timestamps_correct(self, mock_fetch):
        """Timestamps are parsed from date strings in the CSV."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        rec = records[0]
        assert rec.timestamps[0] == datetime(2024, 1, 1)
        assert rec.timestamps[1] == datetime(2024, 1, 2)
        assert rec.timestamps[2] == datetime(2024, 1, 3)

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_fetch_missing_country_returns_empty(self, mock_fetch):
        """ISO code not present in data returns empty list (not an error)."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        records = source.fetch(
            symbols=["ZZZ"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        assert records == []

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_fetch_different_metric(self, mock_fetch):
        """Requesting a different metric (new_deaths) is reflected in series_id and values."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_deaths")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "USA"
        assert rec.metadata["metric"] == "new_deaths"
        assert rec.values == pytest.approx([200.0, 180.0, 210.0])

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_fetch_vaccinations_metric(self, mock_fetch):
        """new_vaccinations metric works correctly."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_vaccinations")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "USA"
        assert rec.values == pytest.approx([100000.0, 95000.0, 110000.0])

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_date_range_filtering(self, mock_fetch):
        """Only rows within the half-open [start, end) range are included."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            domain="health",
        )

        assert len(records) == 1
        rec = records[0]
        assert len(rec.values) == 1
        assert rec.values[0] == pytest.approx(48000.0)

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_location_in_metadata(self, mock_fetch):
        """Country location name is stored in metadata."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        rec = records[0]
        assert rec.metadata["location"] == "United States"

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_csv_fetched_once_for_multiple_symbols(self, mock_fetch):
        """_fetch_owid_csv is called only once even when multiple symbols requested."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        source.fetch(
            symbols=["USA", "GBR"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        assert mock_fetch.call_count == 1

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_freq_is_daily(self, mock_fetch):
        """OWID COVID data is daily, so freq should be '1d'."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource()
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="health",
        )

        assert records[0].freq == "1d"

"""Tests for OWID weekly (and monthly) resample functionality."""

from datetime import datetime
from io import StringIO
from unittest.mock import patch

import pandas as pd
import pytest

from world2ts_core.series.owid import OwidSource


MOCK_CSV_DAILY = """iso_code,continent,location,date,new_cases,new_deaths
USA,North America,United States,2024-01-01,10000,100
USA,North America,United States,2024-01-02,12000,110
USA,North America,United States,2024-01-03,11000,105
USA,North America,United States,2024-01-04,13000,120
USA,North America,United States,2024-01-05,9000,90
USA,North America,United States,2024-01-06,8000,80
USA,North America,United States,2024-01-07,7000,70
USA,North America,United States,2024-01-08,15000,130
USA,North America,United States,2024-01-09,14000,125
USA,North America,United States,2024-01-10,13500,115
USA,North America,United States,2024-01-11,12500,110
USA,North America,United States,2024-01-12,11000,100
USA,North America,United States,2024-01-13,10000,95
USA,North America,United States,2024-01-14,9500,85
"""


def _mock_fetch():
    return pd.read_csv(StringIO(MOCK_CSV_DAILY))


@pytest.fixture(autouse=True)
def _reset_owid_cache():
    """Reset OWID module cache between tests."""
    import world2ts_core.series.owid as owid_mod
    owid_mod._cached_df = None
    yield
    owid_mod._cached_df = None


class TestOwidWeeklyResample:

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_resample_returns_one_record(self, mock_fetch):
        """Weekly resample returns a single SeriesRecord for the requested symbol."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases", resample="W")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        assert len(records) == 1

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_resample_sums_seven_days(self, mock_fetch):
        """Weekly resample sums values within each calendar week."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases", resample="W")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        rec = records[0]
        # Week ending 2024-01-07 (Mon–Sun): 10000+12000+11000+13000+9000+8000+7000 = 70000
        # Week ending 2024-01-14 (Mon–Sun): 15000+14000+13500+12500+11000+10000+9500 = 85500
        assert len(rec.values) == 2
        assert rec.values[0] == pytest.approx(70000.0)
        assert rec.values[1] == pytest.approx(85500.0)

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_resample_freq_is_1w(self, mock_fetch):
        """Weekly resample sets freq='1w' on the SeriesRecord."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases", resample="W")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        assert records[0].freq == "1w"

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_default_no_resample_still_daily(self, mock_fetch):
        """OwidSource with default (no resample) returns daily freq and all rows."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.freq == "1d"
        assert len(rec.values) == 14

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_empty_resample_string_is_daily(self, mock_fetch):
        """OwidSource(resample='') behaves identically to the default."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases", resample="")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        assert records[0].freq == "1d"
        assert len(records[0].values) == 14

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_metadata_preserved(self, mock_fetch):
        """Weekly resample keeps expected metadata fields."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases", resample="W")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        rec = records[0]
        assert rec.series_id == "USA"
        assert rec.domain == "health"
        assert rec.metadata["source"] == "owid_covid"
        assert rec.metadata["iso_code"] == "USA"
        assert rec.metadata["metric"] == "new_cases"
        assert rec.metadata["location"] == "United States"

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_monthly_resample_freq_is_1mo(self, mock_fetch):
        """Monthly resample (resample='M') sets freq='1mo'."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases", resample="M")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        assert len(records) == 1
        assert records[0].freq == "1mo"

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_monthly_resample_sums_all_days(self, mock_fetch):
        """Monthly resample sums all 14 days into one January total."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases", resample="M")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        # All 14 days fall in January: total = 70000 + 85500 = 155500
        assert len(records[0].values) == 1
        assert records[0].values[0] == pytest.approx(155500.0)

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_timestamps_are_week_end_dates(self, mock_fetch):
        """Weekly resample timestamps correspond to week-ending Sundays."""
        mock_fetch.return_value = _mock_fetch()

        source = OwidSource(metric="new_cases", resample="W")
        records = source.fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        rec = records[0]
        # pandas W resamples to week-ending Sunday
        assert rec.timestamps[0] == datetime(2024, 1, 7)
        assert rec.timestamps[1] == datetime(2024, 1, 14)

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_resample_drops_leading_partial_bucket(self, mock_fetch):
        mock_fetch.return_value = _mock_fetch()

        records = OwidSource(metric="new_cases", resample="W-SUN").fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        rec = records[0]
        assert rec.timestamps == [datetime(2024, 1, 14)]
        assert rec.values == pytest.approx([85500.0])

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_resample_drops_trailing_partial_bucket(self, mock_fetch):
        mock_fetch.return_value = _mock_fetch()

        records = OwidSource(metric="new_cases", resample="W-SUN").fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 14),
            domain="health",
        )

        rec = records[0]
        assert rec.timestamps == [datetime(2024, 1, 7)]
        assert rec.values == pytest.approx([70000.0])
        assert all(
            datetime(2024, 1, 1) <= timestamp < datetime(2024, 1, 14)
            for timestamp in rec.timestamps
        )

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_resample_uses_same_buckets_for_auxiliary_columns(self, mock_fetch):
        mock_fetch.return_value = _mock_fetch()

        records = OwidSource(metric="new_cases", resample="W-SUN").fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        rec = records[0]
        assert rec.columns["new_cases"] == pytest.approx(rec.values)
        assert rec.columns["new_deaths"] == pytest.approx([760.0])
        assert len(rec.columns["new_deaths"]) == len(rec.timestamps)

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_resample_returns_empty_without_a_complete_bucket(self, mock_fetch):
        mock_fetch.return_value = _mock_fetch()

        records = OwidSource(metric="new_cases", resample="W-SUN").fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 14),
            domain="health",
        )

        assert records == []

    @patch("world2ts_core.series.owid._fetch_owid_csv")
    def test_weekly_resample_drops_bucket_with_missing_primary_day(self, mock_fetch):
        frame = _mock_fetch()
        frame.loc[frame["date"] == "2024-01-10", "new_cases"] = None
        mock_fetch.return_value = frame

        records = OwidSource(metric="new_cases", resample="W-SUN").fetch(
            symbols=["USA"],
            start=datetime(2024, 1, 8),
            end=datetime(2024, 1, 15),
            domain="health",
        )

        assert records == []

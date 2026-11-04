"""Tests for NOAA ISD hourly weather series source."""

from datetime import datetime, timezone
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from world2ts_core.series.noaa_isd import NoaaIsdSource, _fetch_isd_csv


# The real NOAA ISD CSV quotes multi-value fields (e.g. "+0125,1" as one cell).
MOCK_CSV = (
    'STATION,DATE,SOURCE,REPORT_TYPE,TMP,DEW,WND\n'
    '72505394728,2024-01-01T01:00:00,4,FM-15,"+0125,1","+0055,1","010,1,0050,1"\n'
    '72505394728,2024-01-01T02:00:00,4,FM-15,"+0130,1","+0060,1","015,1,0060,1"\n'
    '72505394728,2024-01-01T03:00:00,4,FM-15,"+0115,1","+0045,1","020,1,0045,1"\n'
    '72505394728,2024-01-01T04:00:00,4,FM-15,"+0110,1","+0040,1","025,1,0040,1"\n'
)

MOCK_CSV_WITH_MISSING = (
    'STATION,DATE,SOURCE,REPORT_TYPE,TMP,DEW,WND\n'
    '72505394728,2024-01-01T01:00:00,4,FM-15,"+0125,1","+0055,1","010,1,0050,1"\n'
    '72505394728,2024-01-01T02:00:00,4,FM-15,"+9999,9","+0060,1","015,1,0060,1"\n'
    '72505394728,2024-01-01T03:00:00,4,FM-15,"+0115,1","+0045,1","020,1,0045,1"\n'
    '72505394728,2024-01-01T04:00:00,4,FM-15,"+9999,9","+0040,1","025,1,0040,1"\n'
)

MOCK_CSV_EMPTY = 'STATION,DATE,SOURCE,REPORT_TYPE,TMP,DEW,WND\n'

MOCK_CSV_SUBHOURLY = (
    'STATION,DATE,SOURCE,REPORT_TYPE,TMP,DEW,WND\n'
    '72505394728,2024-01-01T01:05:00,4,FM-15,"+0100,1","+0040,1","010,1,0050,1"\n'
    '72505394728,2024-01-01T01:55:00,4,FM-15,"+0140,1","+0060,1","015,1,0060,1"\n'
    '72505394728,2024-01-01T02:10:00,4,FM-15,"+9999,9","+0070,1","020,1,0045,1"\n'
    '72505394728,2024-01-01T02:20:00,4,FM-15,"+0200,1","+0080,1","025,1,0040,1"\n'
    '72505394728,2024-01-01T02:50:00,4,FM-15,"+0220,1","+0090,1","025,1,0040,1"\n'
)


def _make_df(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(csv_text))


class TestNoaaIsdSource:

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_fetch_single_station(self, mock_fetch):
        """Fetching one station returns one SeriesRecord with 4 values."""
        mock_fetch.return_value = _make_df(MOCK_CSV)

        source = NoaaIsdSource()
        records = source.fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "72505394728"
        assert rec.domain == "weather"
        assert len(rec.values) == 4
        assert rec.metadata["source"] == "noaa_isd"
        assert rec.metadata["station_id"] == "72505394728"

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_freq_is_hourly(self, mock_fetch):
        """ISD data is hourly, so freq should be '1h'."""
        mock_fetch.return_value = _make_df(MOCK_CSV)

        source = NoaaIsdSource()
        records = source.fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert records[0].freq == "1h"

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_temperature_parsed_correctly(self, mock_fetch):
        """TMP '+0125,1' should parse to 12.5°C."""
        mock_fetch.return_value = _make_df(MOCK_CSV)

        source = NoaaIsdSource()
        records = source.fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        rec = records[0]
        # +0125,1 → 12.5; +0130,1 → 13.0; +0115,1 → 11.5; +0110,1 → 11.0
        assert rec.values == pytest.approx([12.5, 13.0, 11.5, 11.0])

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_missing_values_skipped(self, mock_fetch):
        """Rows with TMP=+9999 should be skipped."""
        mock_fetch.return_value = _make_df(MOCK_CSV_WITH_MISSING)

        source = NoaaIsdSource()
        records = source.fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        # 4 rows total, 2 have +9999 → 2 valid values
        assert len(records) == 1
        rec = records[0]
        assert len(rec.values) == 2
        assert rec.values == pytest.approx([12.5, 11.5])

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_empty_csv_returns_no_records(self, mock_fetch):
        """Empty CSV (only headers) returns empty list."""
        mock_fetch.return_value = _make_df(MOCK_CSV_EMPTY)

        source = NoaaIsdSource()
        records = source.fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert records == []

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_fetch_called_once_per_year(self, mock_fetch):
        """_fetch_isd_csv is called once for each year in the date range."""
        mock_fetch.return_value = _make_df(MOCK_CSV)

        source = NoaaIsdSource()
        source.fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            domain="weather",
        )

        # Only 2024, so one call
        assert mock_fetch.call_count == 1

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_multi_year_range(self, mock_fetch):
        """Date range spanning two years calls _fetch_isd_csv twice."""
        mock_fetch.return_value = _make_df(MOCK_CSV)

        source = NoaaIsdSource()
        source.fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1),
            end=datetime(2025, 1, 31),
            domain="weather",
        )

        assert mock_fetch.call_count == 2

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_exception_in_year_fetch_fails_loudly(self, mock_fetch):
        """A failed annual file cannot be treated as a no-observation year."""
        mock_fetch.side_effect = ConnectionError("HTTP 404")

        source = NoaaIsdSource()
        with pytest.raises(RuntimeError, match="72505394728.*2024"):
            source.fetch(
                symbols=["72505394728"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 12, 31),
                domain="weather",
            )

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_timestamps_parsed_correctly(self, mock_fetch):
        """Timestamps in the record match the DATE column in the CSV."""
        mock_fetch.return_value = _make_df(MOCK_CSV)

        source = NoaaIsdSource()
        records = source.fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        rec = records[0]
        assert rec.timestamps[0] == datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
        assert rec.timestamps[1] == datetime(2024, 1, 1, 2, tzinfo=timezone.utc)

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_end_timestamp_is_excluded(self, mock_fetch):
        mock_fetch.return_value = _make_df(MOCK_CSV)

        records = NoaaIsdSource().fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 3, tzinfo=timezone.utc),
            domain="weather",
        )

        assert records[0].timestamps == [
            datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
        ]

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_subhourly_observations_are_averaged_by_utc_hour(self, mock_fetch):
        mock_fetch.return_value = _make_df(MOCK_CSV_SUBHOURLY)

        records = NoaaIsdSource().fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            domain="weather",
        )

        record = records[0]
        assert record.timestamps == [
            datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
        ]
        assert record.values == pytest.approx([12.0, 21.0])
        assert all(
            earlier < later
            for earlier, later in zip(record.timestamps, record.timestamps[1:])
        )

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_partial_first_hour_is_not_labeled_before_start(self, mock_fetch):
        mock_fetch.return_value = _make_df(MOCK_CSV_SUBHOURLY)

        records = NoaaIsdSource().fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1, 1, 30, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 3, tzinfo=timezone.utc),
            domain="weather",
        )

        assert records[0].timestamps == [
            datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
        ]
        assert records[0].values == pytest.approx([21.0])

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_partial_last_hour_is_not_labeled_as_complete(self, mock_fetch):
        mock_fetch.return_value = _make_df(MOCK_CSV_SUBHOURLY)

        records = NoaaIsdSource().fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 2, 30, tzinfo=timezone.utc),
            domain="weather",
        )

        assert records[0].timestamps == [
            datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        ]
        assert records[0].values == pytest.approx([12.0])

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_half_open_year_boundary_does_not_fetch_excluded_year(self, mock_fetch):
        mock_fetch.return_value = _make_df(MOCK_CSV)

        NoaaIsdSource().fetch(
            symbols=["72505394728"],
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            domain="weather",
        )

        assert mock_fetch.call_count == 1
        assert mock_fetch.call_args.args == ("72505394728", 2024)

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_later_year_failure_does_not_return_partial_data(self, mock_fetch):
        mock_fetch.side_effect = [
            _make_df(MOCK_CSV),
            ConnectionError("2025 unavailable"),
        ]

        with pytest.raises(RuntimeError, match="2025"):
            NoaaIsdSource().fetch(
                symbols=["72505394728"],
                start=datetime(2024, 1, 1),
                end=datetime(2025, 1, 31),
                domain="weather",
            )

    @patch("world2ts_core.series.noaa_isd._fetch_isd_csv")
    def test_missing_required_columns_is_not_empty_data(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame({"TMP": []})

        with pytest.raises(ValueError, match="DATE"):
            NoaaIsdSource().fetch(
                symbols=["72505394728"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31),
                domain="weather",
            )

    @patch("world2ts_core.series.noaa_isd.subprocess.run")
    def test_curl_failure_is_not_empty_data(self, mock_run):
        mock_run.return_value = MagicMock(returncode=22, stderr="HTTP 404")

        with pytest.raises(ConnectionError, match="curl failed"):
            _fetch_isd_csv("72505394728", 2024)

    @patch("world2ts_core.series.noaa_isd.subprocess.run")
    def test_empty_download_is_not_empty_data(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with pytest.raises(ConnectionError, match="empty response"):
            _fetch_isd_csv("72505394728", 2024)

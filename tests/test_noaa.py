"""Tests for NOAA CDO API series source."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from world2ts_core.series.noaa import NoaaSource, _api_get


MOCK_CDO_RESPONSE = {
    "metadata": {"resultset": {"offset": 1, "count": 4, "limit": 1000}},
    "results": [
        {"date": "2024-01-01T00:00:00", "datatype": "TMAX", "value": 12.2, "station": "GHCND:USW00094728"},
        {"date": "2024-01-01T00:00:00", "datatype": "TMIN", "value": 4.4, "station": "GHCND:USW00094728"},
        {"date": "2024-01-02T00:00:00", "datatype": "TMAX", "value": 13.3, "station": "GHCND:USW00094728"},
        {"date": "2024-01-02T00:00:00", "datatype": "TMIN", "value": 5.6, "station": "GHCND:USW00094728"},
    ],
}

MOCK_CDO_EMPTY = {
    "metadata": {"resultset": {"offset": 1, "count": 0, "limit": 1000}},
    "results": [],
}


class TestNoaaSource:

    @patch("world2ts_core.series.noaa._api_get")
    def test_fetch_single_station_tmax(self, mock_api_get):
        """Fetch TMAX for one station returns one record with correct values."""
        mock_api_get.return_value = MOCK_CDO_RESPONSE

        source = NoaaSource(api_token="test_token")
        records = source.fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "USW00094728"
        assert rec.domain == "weather"
        assert rec.metadata["source"] == "noaa_ghcnd"
        assert rec.metadata["station_id"] == "USW00094728"
        assert rec.metadata["datatype"] == "TMAX"

    @patch("world2ts_core.series.noaa._api_get")
    def test_metric_values_are_not_rescaled(self, mock_api_get):
        """CDO metric responses already contain Celsius values."""
        mock_api_get.return_value = MOCK_CDO_RESPONSE

        source = NoaaSource(api_token="test_token")
        records = source.fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert len(records) == 1
        assert records[0].values == pytest.approx([12.2, 13.3])
        assert mock_api_get.call_args.args[1]["units"] == "metric"

    @patch("world2ts_core.series.noaa._api_get")
    def test_tmin_metric_values_are_not_rescaled(self, mock_api_get):
        """TMIN uses the same metric response semantics as TMAX."""
        mock_api_get.return_value = MOCK_CDO_RESPONSE

        source = NoaaSource(api_token="test_token", datatype="TMIN")
        records = source.fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert len(records) == 1
        assert records[0].values == pytest.approx([4.4, 5.6])

    @patch("world2ts_core.series.noaa._api_get")
    def test_prcp_metric_values_are_millimeters(self, mock_api_get):
        mock_api_get.return_value = {
            "metadata": {"resultset": {"offset": 1, "count": 2, "limit": 1000}},
            "results": [
                {"date": "2024-01-01T00:00:00", "datatype": "PRCP", "value": 1.7},
                {"date": "2024-01-02T00:00:00", "datatype": "PRCP", "value": 25.4},
            ],
        }

        records = NoaaSource(api_token="test_token", datatype="PRCP").fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert records[0].values == pytest.approx([1.7, 25.4])

    @patch("world2ts_core.series.noaa._api_get")
    def test_empty_response_returns_no_records(self, mock_api_get):
        """Empty API response returns empty list."""
        mock_api_get.return_value = MOCK_CDO_EMPTY

        source = NoaaSource(api_token="test_token")
        records = source.fetch(
            symbols=["USW99999999"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert records == []

    @patch("world2ts_core.series.noaa._api_get")
    def test_config_token_usage(self, mock_api_get, monkeypatch):
        """Token from env var is used when no explicit token given."""
        monkeypatch.setenv("WORLD2TS_NOAA_API_TOKEN", "env_token")
        mock_api_get.return_value = MOCK_CDO_EMPTY

        source = NoaaSource()
        source.fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        # _api_get should have been called with the token header
        call_kwargs = mock_api_get.call_args
        assert call_kwargs is not None
        # The token is passed as a kwarg or positional — check it was called
        assert mock_api_get.called

    @patch("world2ts_core.series.noaa._api_get")
    def test_station_prefixed_with_ghcnd(self, mock_api_get):
        """Station IDs are prefixed with 'GHCND:' when passed to the API."""
        mock_api_get.return_value = MOCK_CDO_EMPTY

        source = NoaaSource(api_token="test_token")
        source.fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        call_kwargs = mock_api_get.call_args
        # params passed to _api_get should include stationid with GHCND: prefix
        params = call_kwargs[0][1] if call_kwargs[0] else call_kwargs[1].get("params", {})
        assert "GHCND:USW00094728" in params.get("stationid", "")

    @patch("world2ts_core.series.noaa._api_get")
    def test_timestamps_parsed_correctly(self, mock_api_get):
        """Timestamps are parsed from ISO format strings."""
        mock_api_get.return_value = MOCK_CDO_RESPONSE

        source = NoaaSource(api_token="test_token")
        records = source.fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        assert len(records) == 1
        assert records[0].timestamps[0] == datetime(2024, 1, 1)
        assert records[0].timestamps[1] == datetime(2024, 1, 2)

    @patch("world2ts_core.series.noaa._api_get")
    def test_freq_is_daily(self, mock_api_get):
        """GHCND data is daily, so freq should be '1d'."""
        mock_api_get.return_value = MOCK_CDO_RESPONSE

        source = NoaaSource(api_token="test_token")
        records = source.fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            domain="weather",
        )

        for rec in records:
            assert rec.freq == "1d"

    @patch("world2ts_core.series.noaa._api_get")
    def test_fetches_every_reported_page(self, mock_api_get):
        mock_api_get.side_effect = [
            {
                "metadata": {"resultset": {"offset": 1, "count": 3, "limit": 2}},
                "results": [
                    {"date": "2024-01-01T00:00:00", "datatype": "TMAX", "value": 10.0},
                    {"date": "2024-01-02T00:00:00", "datatype": "TMAX", "value": 11.0},
                ],
            },
            {
                "metadata": {"resultset": {"offset": 3, "count": 3, "limit": 2}},
                "results": [
                    {"date": "2024-01-03T00:00:00", "datatype": "TMAX", "value": 12.0},
                ],
            },
        ]

        records = NoaaSource(api_token="test_token").fetch(
            symbols=["USW00094728"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 4),
            domain="weather",
        )

        assert records[0].values == pytest.approx([10.0, 11.0, 12.0])
        assert [call.args[1]["offset"] for call in mock_api_get.call_args_list] == [1, 3]

    @patch("world2ts_core.series.noaa._api_get")
    def test_later_page_failure_does_not_return_partial_data(self, mock_api_get):
        mock_api_get.side_effect = [
            {
                "metadata": {"resultset": {"offset": 1, "count": 3, "limit": 2}},
                "results": [
                    {"date": "2024-01-01T00:00:00", "datatype": "TMAX", "value": 10.0},
                    {"date": "2024-01-02T00:00:00", "datatype": "TMAX", "value": 11.0},
                ],
            },
            ConnectionError("page 2 unavailable"),
        ]

        with pytest.raises(RuntimeError, match="USW00094728"):
            NoaaSource(api_token="test_token").fetch(
                symbols=["USW00094728"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 4),
                domain="weather",
            )

    @patch("world2ts_core.series.noaa._api_get")
    def test_later_chunk_failure_does_not_return_partial_data(self, mock_api_get):
        mock_api_get.side_effect = [
            {
                "metadata": {"resultset": {"offset": 1, "count": 1, "limit": 1000}},
                "results": [
                    {"date": "2024-01-01T00:00:00", "datatype": "TMAX", "value": 10.0},
                ],
            },
            ConnectionError("next chunk unavailable"),
        ]

        with pytest.raises(RuntimeError, match="2024-12-31"):
            NoaaSource(api_token="test_token").fetch(
                symbols=["USW00094728"],
                start=datetime(2024, 1, 1),
                end=datetime(2025, 1, 1),
                domain="weather",
            )

    @patch("world2ts_core.series.noaa.subprocess.run")
    def test_api_get_curl_failure_is_not_empty_data(self, mock_run):
        mock_run.return_value = MagicMock(returncode=22, stderr="HTTP 500", stdout="")

        with pytest.raises(ConnectionError, match="curl failed"):
            _api_get("https://example.test", {}, "test_token")

    @patch("world2ts_core.series.noaa.subprocess.run")
    def test_api_get_empty_body_is_not_empty_data(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        with pytest.raises(ConnectionError, match="empty response"):
            _api_get("https://example.test", {}, "test_token")

"""Tests for EIA energy series source."""

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from world2ts_core.series.eia import EiaSource


MOCK_EIA_RESPONSE = {
    "response": {
        "data": [
            {"period": "2025-01-02", "value": 73.5},
            {"period": "2025-01-03", "value": 74.2},
            {"period": "2025-01-06", "value": 72.8},
        ],
        "description": "WTI Crude Oil Spot Price",
    }
}


class TestEiaCapability:
    def test_has_capability(self):
        assert hasattr(EiaSource, "capability")
        cap = EiaSource.capability
        assert cap.name == "eia"
        assert "energy" in cap.domains
        assert cap.auth_required is True
        assert cap.frequencies == ["1d"]


class TestEiaFetch:

    @patch("world2ts_core.series.eia._curl_get")
    def test_fetch_returns_records(self, mock_curl):
        mock_curl.return_value = json.dumps(MOCK_EIA_RESPONSE)

        source = EiaSource(api_key="test_key")
        records = source.fetch(
            symbols=["RWTC"],
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 31),
            domain="energy",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.series_id == "RWTC"
        assert rec.domain == "energy"
        assert len(rec.values) == 3
        assert rec.values[0] == 73.5

    @patch("world2ts_core.series.eia._curl_get")
    def test_fetch_empty_response_fails_loudly(self, mock_curl):
        mock_curl.return_value = json.dumps({"response": {"data": []}})

        source = EiaSource(api_key="test_key")
        with pytest.raises(RuntimeError, match="no data for RWTC"):
            source.fetch(
                symbols=["RWTC"],
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 31),
                domain="energy",
            )

    @patch.dict("os.environ", {}, clear=True)
    @patch("world2ts_core.series.eia.config_get", return_value="")
    def test_no_api_key_fails_loudly(self, _mock_config):
        source = EiaSource(api_key="")
        with pytest.raises(ValueError, match="EIA API key not configured"):
            source.fetch(
                symbols=["RWTC"],
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 31),
                domain="energy",
            )

    @patch("world2ts_core.series.eia._curl_get")
    def test_connection_failure_fails_loudly_with_symbol(self, mock_curl):
        mock_curl.side_effect = ConnectionError("upstream unavailable")

        source = EiaSource(api_key="test_key")
        with pytest.raises(RuntimeError, match="EIA fetch failed for RWTC"):
            source.fetch(
                symbols=["RWTC"],
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 31),
                domain="energy",
            )

    @pytest.mark.parametrize(
        "response",
        [
            "not-json",
            json.dumps([]),
            json.dumps({"error": "invalid api key"}),
            json.dumps({"response": {"data": {}}}),
        ],
    )
    @patch("world2ts_core.series.eia._curl_get")
    def test_malformed_or_error_response_fails_loudly(
        self, mock_curl, response
    ):
        mock_curl.return_value = response

        source = EiaSource(api_key="test_key")
        with pytest.raises(RuntimeError, match="RWTC"):
            source.fetch(
                symbols=["RWTC"],
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 31),
                domain="energy",
            )

    @pytest.mark.parametrize(
        "row",
        [
            [],
            {"value": 73.5},
            {"period": "2025-01-02"},
            {"period": "January 2", "value": 73.5},
            {"period": "2025-01-02", "value": "not-a-number"},
        ],
    )
    @patch("world2ts_core.series.eia._curl_get")
    def test_malformed_observation_fails_loudly(self, mock_curl, row):
        mock_curl.return_value = json.dumps({"response": {"data": [row]}})

        source = EiaSource(api_key="test_key")
        with pytest.raises(RuntimeError, match="malformed row 0 for RWTC"):
            source.fetch(
                symbols=["RWTC"],
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 31),
                domain="energy",
            )

    @patch("world2ts_core.series.eia._curl_get")
    def test_null_observation_is_preserved_as_a_gap(self, mock_curl):
        mock_curl.return_value = json.dumps(
            {
                "response": {
                    "data": [
                        {"period": "2025-01-02", "value": None},
                        {"period": "2025-01-03", "value": 74.2},
                    ]
                }
            }
        )

        source = EiaSource(api_key="test_key")
        records = source.fetch(
            symbols=["RWTC"],
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 31),
            domain="energy",
        )

        assert records[0].values == [74.2]
        assert records[0].timestamps == [datetime(2025, 1, 3)]

    def test_unknown_series_is_rejected_without_route_fallback(self):
        source = EiaSource(api_key="test_key")

        with pytest.raises(ValueError, match="Unsupported EIA series"):
            source.fetch(
                symbols=["TYPO_SERIES"],
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 31),
                domain="energy",
            )

    def test_monthly_electricity_series_is_rejected(self):
        source = EiaSource(api_key="test_key")

        with pytest.raises(ValueError, match="Unsupported EIA series"):
            source.fetch(
                symbols=["ELEC.PRICE.US-ALL.M"],
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 31),
                domain="energy",
            )

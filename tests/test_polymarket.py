"""Tests for Polymarket prediction market series source."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from world2ts_core.series.polymarket import PolymarketSource


MOCK_PRICE_HISTORY = json.dumps({
    "history": [
        {"t": 1704067200, "p": 0.45},  # 2024-01-01
        {"t": 1704153600, "p": 0.48},  # 2024-01-02
        {"t": 1704240000, "p": 0.52},  # 2024-01-03
        {"t": 1704326400, "p": 0.55},  # 2024-01-04
        {"t": 1704412800, "p": 0.50},  # 2024-01-05
    ]
})

MARKET_META = {
    "question": "Will Trump win 2024 election?",
    "slug": "trump-2024",
    "category": "politics",
    "outcomes": ["Yes", "No"],
}
MOCK_MARKET_META = json.dumps([MARKET_META])


class TestPolymarketCapability:

    def test_has_capability(self):
        cap = PolymarketSource.capability
        assert cap.name == "polymarket"
        assert "finance" in cap.domains

    def test_capability_fields(self):
        cap = PolymarketSource.capability
        assert "crypto" in cap.domains
        assert "macro" in cap.domains
        assert "polymarket" in cap.domains
        assert cap.auth_required is False
        assert "prediction_market" in cap.entity_types


class TestPolymarketFetch:

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_market_discovery_connection_error_propagates(self, mock_curl):
        mock_curl.side_effect = ConnectionError("network down")

        from world2ts_core.series.polymarket import discover_markets

        with pytest.raises(ConnectionError, match="network down"):
            discover_markets()

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_market_discovery_invalid_json_fails_loudly(self, mock_curl):
        mock_curl.return_value = "not-json"

        from world2ts_core.series.polymarket import discover_markets

        with pytest.raises(ValueError, match="invalid JSON"):
            discover_markets()

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_fetch_returns_records(self, mock_curl):
        """Fetch returns a SeriesRecord from mocked _curl_get response."""
        mock_curl.side_effect = [MOCK_PRICE_HISTORY, MOCK_MARKET_META]

        source = PolymarketSource()
        source._last_request = 0.0
        records = source.fetch(
            symbols=["token123"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
            domain="finance",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.domain == "finance"
        assert rec.entity == "Will Trump win 2024 election?"
        assert len(rec.values) == 4
        assert rec.metadata["source"] == "polymarket"
        assert rec.metadata["question"] == "Will Trump win 2024 election?"
        assert rec.metadata["slug"] == "trump-2024"
        assert "clob_token_ids=token123" in mock_curl.call_args_list[1].args[0]

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_fetch_empty_history(self, mock_curl):
        """Empty history from API yields empty record list."""
        mock_curl.return_value = json.dumps({"history": []})

        source = PolymarketSource()
        source._last_request = 0.0
        records = source.fetch(
            symbols=["token_empty"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
            domain="finance",
        )

        assert records == []

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_fetch_connection_error(self, mock_curl):
        """Connection errors propagate instead of becoming empty data."""
        mock_curl.side_effect = ConnectionError("network down")

        source = PolymarketSource()
        source._last_request = 0.0
        with pytest.raises(ConnectionError, match="network down"):
            source.fetch(
                symbols=["token_fail"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 5),
                domain="finance",
            )

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_timestamps_parsed_from_unix(self, mock_curl):
        """Timestamps are parsed correctly from unix epoch seconds."""
        mock_curl.side_effect = [MOCK_PRICE_HISTORY, MOCK_MARKET_META]

        source = PolymarketSource()
        source._last_request = 0.0
        records = source.fetch(
            symbols=["token123"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
            domain="finance",
        )

        rec = records[0]
        # 1704067200 = 2024-01-01 00:00:00 UTC
        assert rec.timestamps[0] == datetime(2024, 1, 1, 0, 0, 0)
        # 1704153600 = 2024-01-02 00:00:00 UTC
        assert rec.timestamps[1] == datetime(2024, 1, 2, 0, 0, 0)
        assert isinstance(rec.timestamps[0], datetime)

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_price_values_stored(self, mock_curl):
        """Price values are stored correctly as floats."""
        mock_curl.side_effect = [MOCK_PRICE_HISTORY, MOCK_MARKET_META]

        source = PolymarketSource()
        source._last_request = 0.0
        records = source.fetch(
            symbols=["token123"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
            domain="finance",
        )

        rec = records[0]
        assert rec.values == [0.45, 0.48, 0.52, 0.55]
        assert all(isinstance(v, float) for v in rec.values)

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_preserves_full_id_filters_range_and_routes_fidelity(self, mock_curl):
        long_token_id = "token_" + "x" * 40
        history = json.dumps(
            {
                "history": [
                    {"t": 1703980800, "p": 0.40},
                    {"t": 1704153600, "p": 0.48},
                    {"t": 1704499200, "p": 0.60},
                ]
            }
        )
        mock_curl.side_effect = [history, MOCK_MARKET_META]

        source = PolymarketSource(interval="1h")
        source._last_request = 0.0
        records = source.fetch(
            symbols=[long_token_id],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
            domain="polymarket",
        )

        assert len(records) == 1
        assert records[0].series_id == long_token_id
        assert records[0].timestamps == [datetime(2024, 1, 2)]
        assert records[0].values == [0.48]
        assert "fidelity=60" in mock_curl.call_args_list[0].args[0]

    @pytest.mark.parametrize(
        ("metadata_response", "error"),
        [
            ("not-json", "not valid JSON"),
            (json.dumps({}), "must be a list"),
            (json.dumps([]), "exactly one market"),
            (json.dumps([MARKET_META, MARKET_META]), "exactly one market"),
            (json.dumps([{"question": "  "}]), "no non-empty question"),
        ],
    )
    @patch("world2ts_core.series.polymarket._curl_get")
    def test_metadata_contract_fails_loudly(
        self,
        mock_curl,
        metadata_response,
        error,
    ):
        mock_curl.side_effect = [MOCK_PRICE_HISTORY, metadata_response]

        source = PolymarketSource()
        source._last_request = 0.0
        with pytest.raises(ValueError, match=error):
            source.fetch(
                symbols=["token123"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 5),
                domain="polymarket",
            )

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_metadata_connection_error_propagates(self, mock_curl):
        mock_curl.side_effect = [MOCK_PRICE_HISTORY, ConnectionError("network down")]

        source = PolymarketSource()
        source._last_request = 0.0
        with pytest.raises(ConnectionError, match="network down"):
            source.fetch(
                symbols=["token123"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 5),
                domain="polymarket",
            )

    @patch("world2ts_core.series.polymarket._curl_get")
    def test_malformed_price_history_fails_loudly(self, mock_curl):
        mock_curl.return_value = json.dumps({"history": {}})

        with pytest.raises(ValueError, match="invalid history"):
            PolymarketSource().fetch(
                ["token123"],
                datetime(2024, 1, 1),
                datetime(2024, 1, 5),
                "polymarket",
            )

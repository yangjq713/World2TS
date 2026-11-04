"""Integration tests for the source capability registry."""

import json

from typer.testing import CliRunner

from world2ts_core.bootstrap import ensure_sources_registered
from world2ts_core.cli import app
from world2ts_core.registry import list_event_sources, list_series_sources

runner = CliRunner()


class TestRegistryCompleteness:

    def test_all_event_sources_registered(self):
        """Both approved offline Web sources are registered."""
        ensure_sources_registered()
        caps = list_event_sources()
        names = {c.name for c in caps}
        assert names == {"tavily", "anysearch"}

    def test_all_series_sources_registered(self):
        """Every series source has a capability declaration."""
        ensure_sources_registered()
        caps = list_series_sources()
        names = {c.name for c in caps}
        expected = {
            "yahoo", "fred", "coingecko", "noaa", "noaa_isd", "owid",
            "eia", "polymarket",
        }
        assert names == expected


class TestSourcesListCLI:

    def test_json_output_structure(self):
        result = runner.invoke(app, ["--json", "sources", "list"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "event_sources" in data
        assert "series_sources" in data
        assert len(data["event_sources"]) == 2
        assert len(data["series_sources"]) == 8

    def test_each_event_source_has_all_fields(self):
        result = runner.invoke(app, ["--json", "sources", "list"])
        data = json.loads(result.stdout)
        required_fields = {
            "name", "kind", "domains", "entity_types", "event_types",
            "geo", "history_days", "time_resolution", "rate_per_min",
            "auth_required", "reliability", "description",
            "query_format", "required_params", "output_fields",
        }
        for src in data["event_sources"]:
            missing = required_fields - set(src.keys())
            assert not missing, f"{src['name']} missing fields: {missing}"

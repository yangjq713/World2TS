"""Tests for sources list CLI command."""

import json

from typer.testing import CliRunner

from world2ts_core.cli import app

runner = CliRunner()


class TestSourcesList:

    def test_list_all_event_sources(self):
        result = runner.invoke(app, ["--json", "sources", "list"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        event_sources = data["event_sources"]
        names = {s["name"] for s in event_sources}
        assert names == {"tavily", "anysearch"}

    def test_list_all_series_sources(self):
        result = runner.invoke(app, ["--json", "sources", "list"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        series_sources = data["series_sources"]
        names = [s["name"] for s in series_sources]
        assert "yahoo" in names
        assert "fred" in names
        assert "coingecko" in names
        assert "noaa" in names
        assert "noaa_isd" in names
        assert "owid" in names
        yahoo = next(source for source in series_sources if source["name"] == "yahoo")
        assert yahoo["primary_target_fields"] == ["close"]
        assert yahoo["range_semantics"] == "[start, end)"
        noaa = next(source for source in series_sources if source["name"] == "noaa")
        assert noaa["target_option"] == "datatype"
        assert noaa["adapter_options"]["datatype"] == ["TMAX", "TMIN", "PRCP"]

    def test_filter_by_domain(self):
        result = runner.invoke(app, ["--json", "sources", "list", "--domain", "finance"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        event_names = {s["name"] for s in data["event_sources"]}
        series_names = [s["name"] for s in data["series_sources"]]
        assert event_names == {"tavily", "anysearch"}
        assert "yahoo" in series_names
        assert "noaa" not in series_names

    def test_filter_by_event_type(self):
        result = runner.invoke(app, ["--json", "sources", "list", "--event-type", "web"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        names = {s["name"] for s in data["event_sources"]}
        assert names == {"tavily", "anysearch"}

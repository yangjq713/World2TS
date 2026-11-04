"""Tests for capability models and source registry."""

from __future__ import annotations

from world2ts_core.registry import (
    EventSourceCapability,
    SeriesSourceCapability,
    _EVENT_REGISTRY,
    _SERIES_REGISTRY,
    register_event,
    register_series,
    list_event_sources,
    list_series_sources,
)


def _make_event_cap(**overrides):
    """Helper to create EventSourceCapability with sensible defaults."""
    defaults = dict(
        name="test", kind="search", domains=["finance"],
        entity_types=["stock"], event_types=["news"],
        geo=["global"], history_days=90, time_resolution="day",
        rate_per_min=60, auth_required=False, reliability="medium",
        description="Test source", query_format="keyword",
        required_params=["query"], optional_params=[],
        output_fields=["headline"],
    )
    defaults.update(overrides)
    return EventSourceCapability(**defaults)


def _make_series_cap(**overrides):
    """Helper to create SeriesSourceCapability with sensible defaults."""
    defaults = dict(
        name="test", domains=["finance"], entity_types=["stock"],
        frequencies=["1d"], geo=["global"], history_days=365,
        rate_per_min=60, auth_required=False, reliability="high",
        description="Test series",
        primary_target_fields=["value"], target_option="",
        adapter_options={}, metadata_dimensions=[],
        timestamp_semantics="plan timezone", range_semantics="[start, end)",
    )
    defaults.update(overrides)
    return SeriesSourceCapability(**defaults)


class TestEventSourceCapability:

    def test_create_with_all_fields(self):
        cap = _make_event_cap(name="web", kind="search", domains=["finance", "crypto"])
        assert cap.name == "web"
        assert cap.kind == "search"
        assert cap.domains == ["finance", "crypto"]
        assert cap.geo == ["global"]
        assert cap.reliability == "medium"

    def test_required_fields(self):
        """All spec fields are required (no optional defaults)."""
        cap = _make_event_cap(
            query_format="keyword",
            required_params=["query", "start", "end"],
            optional_params=["geo"],
            output_fields=["headline", "date", "url"],
        )
        assert cap.required_params == ["query", "start", "end"]
        assert cap.output_fields == ["headline", "date", "url"]


class TestSeriesSourceCapability:

    def test_create(self):
        cap = _make_series_cap(
            name="yahoo", frequencies=["1d", "1h", "15min"],
            geo=["global"], history_days=3650,
        )
        assert cap.name == "yahoo"
        assert "1h" in cap.frequencies
        assert cap.geo == ["global"]
        assert cap.reliability == "high"


class TestRegistry:

    def setup_method(self):
        self._saved_event = list(_EVENT_REGISTRY)
        self._saved_series = list(_SERIES_REGISTRY)
        _EVENT_REGISTRY.clear()
        _SERIES_REGISTRY.clear()

    def teardown_method(self):
        _EVENT_REGISTRY.clear()
        _EVENT_REGISTRY.extend(self._saved_event)
        _SERIES_REGISTRY.clear()
        _SERIES_REGISTRY.extend(self._saved_series)

    def test_register_event_decorator(self):
        @register_event
        class FakeSource:
            capability = _make_event_cap(name="fake")

        assert FakeSource in _EVENT_REGISTRY
        assert len(_EVENT_REGISTRY) == 1

    def test_register_series_decorator(self):
        @register_series
        class FakeSeries:
            capability = _make_series_cap(name="fake_series")

        assert FakeSeries in _SERIES_REGISTRY
        assert len(_SERIES_REGISTRY) == 1

    def test_register_returns_class_unchanged(self):
        @register_event
        class MySource:
            capability = _make_event_cap(name="my")

        assert MySource.capability.name == "my"

    def test_multiple_registrations(self):
        @register_event
        class A:
            capability = _make_event_cap(name="a")

        @register_event
        class B:
            capability = _make_event_cap(name="b")

        assert len(_EVENT_REGISTRY) == 2


class TestListEventSources:

    def setup_method(self):
        self._saved_event = list(_EVENT_REGISTRY)
        self._saved_series = list(_SERIES_REGISTRY)
        _EVENT_REGISTRY.clear()
        _SERIES_REGISTRY.clear()
        self._register_fixtures()

    def teardown_method(self):
        _EVENT_REGISTRY.clear()
        _EVENT_REGISTRY.extend(self._saved_event)
        _SERIES_REGISTRY.clear()
        _SERIES_REGISTRY.extend(self._saved_series)

    def _register_fixtures(self):
        @register_event
        class FinanceSearch:
            capability = _make_event_cap(
                name="finance_search", kind="search",
                domains=["finance"], event_types=["earnings", "merger"],
            )

        @register_event
        class WeatherSearch:
            capability = _make_event_cap(
                name="weather_search", kind="search",
                domains=["weather"], event_types=["storm", "hurricane"],
            )

        @register_event
        class FinanceEarningsSearch:
            capability = _make_event_cap(
                name="finance_earnings_search", kind="search",
                domains=["finance"], event_types=["earnings"],
            )

    def test_no_filters(self):
        results = list_event_sources()
        assert len(results) == 3

    def test_filter_by_domain(self):
        results = list_event_sources(domain="finance")
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"finance_search", "finance_earnings_search"}

    def test_filter_by_event_type(self):
        results = list_event_sources(event_type="storm")
        assert len(results) == 1
        assert results[0].name == "weather_search"

    def test_filter_no_match(self):
        results = list_event_sources(domain="crypto")
        assert results == []


class TestListSeriesSources:

    def setup_method(self):
        self._saved_event = list(_EVENT_REGISTRY)
        self._saved_series = list(_SERIES_REGISTRY)
        _EVENT_REGISTRY.clear()
        _SERIES_REGISTRY.clear()
        self._register_fixtures()

    def teardown_method(self):
        _EVENT_REGISTRY.clear()
        _EVENT_REGISTRY.extend(self._saved_event)
        _SERIES_REGISTRY.clear()
        _SERIES_REGISTRY.extend(self._saved_series)

    def _register_fixtures(self):
        @register_series
        class FredSource:
            capability = _make_series_cap(
                name="fred", domains=["macro", "finance"],
            )

        @register_series
        class CoinGeckoSource:
            capability = _make_series_cap(
                name="coingecko", domains=["crypto"],
                frequencies=["1h", "1d"],
            )

    def test_no_filters(self):
        results = list_series_sources()
        assert len(results) == 2

    def test_filter_by_domain(self):
        results = list_series_sources(domain="crypto")
        assert len(results) == 1
        assert results[0].name == "coingecko"

    def test_filter_by_domain_multi(self):
        """Source with multiple domains matches any of them."""
        results = list_series_sources(domain="finance")
        assert len(results) == 1
        assert results[0].name == "fred"

    def test_filter_no_match(self):
        results = list_series_sources(domain="weather")
        assert results == []


class TestExistingEventSources:
    """Verify canonical Web-search sources declare capabilities."""

    def test_tavily_registered(self):
        from world2ts_core.sources.tavily import TavilySource

        cap = TavilySource.capability
        assert cap.name == "tavily"
        assert cap.kind == "search"
        assert cap.reliability == "medium"
        assert cap.geo == ["global"]

    def test_anysearch_registered(self):
        from world2ts_core.sources.anysearch import AnySearchSource

        cap = AnySearchSource.capability
        assert cap.name == "anysearch"
        assert cap.kind == "search"
        assert cap.reliability == "medium"


class TestExistingSeriesSources:
    """Verify all series sources declare capabilities."""

    def test_yahoo_registered(self):
        from world2ts_core.series.yahoo import YahooSource
        assert hasattr(YahooSource, "capability")
        cap = YahooSource.capability
        assert cap.name == "yahoo"
        assert "1d" in cap.frequencies
        assert "1h" in cap.frequencies

    def test_fred_registered(self):
        from world2ts_core.series.fred import FredSource
        assert hasattr(FredSource, "capability")
        cap = FredSource.capability
        assert cap.name == "fred"
        assert "macro" in cap.domains

    def test_coingecko_registered(self):
        from world2ts_core.series.binance import CoinGeckoSource
        assert hasattr(CoinGeckoSource, "capability")
        cap = CoinGeckoSource.capability
        assert cap.name == "coingecko"
        assert "crypto" in cap.domains

    def test_noaa_registered(self):
        from world2ts_core.series.noaa import NoaaSource
        assert hasattr(NoaaSource, "capability")
        cap = NoaaSource.capability
        assert cap.name == "noaa"
        assert cap.auth_required is True

    def test_noaa_isd_registered(self):
        from world2ts_core.series.noaa_isd import NoaaIsdSource
        assert hasattr(NoaaIsdSource, "capability")
        cap = NoaaIsdSource.capability
        assert cap.name == "noaa_isd"
        assert "1h" in cap.frequencies

    def test_owid_registered(self):
        from world2ts_core.series.owid import OwidSource
        assert hasattr(OwidSource, "capability")
        cap = OwidSource.capability
        assert cap.name == "owid"
        assert "health" in cap.domains

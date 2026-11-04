"""Tests for exact source/frequency adapter construction."""

from __future__ import annotations

import pytest

from world2ts_core.freq_routing import constructor_options_for_source, normalize_freq


class TestNormalizeFreq:
    def test_canonical(self):
        assert normalize_freq("1d") == "1d"
        assert normalize_freq("1h") == "1h"

    def test_aliases(self):
        assert normalize_freq("daily") == "1d"
        assert normalize_freq("hourly") == "1h"
        assert normalize_freq("weekly") == "1w"
        assert normalize_freq("15m") == "15min"
        assert normalize_freq("d") == "1d"

    def test_case_insensitive(self):
        assert normalize_freq("1D") == "1d"
        assert normalize_freq("Daily") == "1d"

    def test_unknown_passthrough(self):
        assert normalize_freq("5min") == "5min"


class TestConstructorOptionsForSource:
    def test_yahoo_frequencies(self):
        assert constructor_options_for_source("yahoo", "1d") == {"interval": "1d"}
        assert constructor_options_for_source("yahoo", "1h") == {"interval": "1h"}
        assert constructor_options_for_source("yahoo", "15min") == {"interval": "15m"}

    def test_weather_sources_are_explicit(self):
        assert constructor_options_for_source("noaa", "1d") == {}
        assert constructor_options_for_source("noaa_isd", "1h") == {}

    def test_owid_resampling(self):
        assert constructor_options_for_source("owid", "1d") == {"resample": ""}
        assert constructor_options_for_source("owid", "1w") == {"resample": "W-SUN"}
        assert constructor_options_for_source("owid", "1mo") == {"resample": "ME"}

    def test_coingecko_hourly(self):
        assert constructor_options_for_source("coingecko", "1h") == {"interval": "1h"}

    def test_unsupported_freq_raises_with_supported_values(self):
        with pytest.raises(ValueError, match="Supported frequencies"):
            constructor_options_for_source("noaa", "15min")

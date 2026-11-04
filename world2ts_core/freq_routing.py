"""Build an explicitly selected series adapter for a requested frequency."""

from __future__ import annotations

from world2ts_core.series.base import SeriesSource

_FREQ_ALIASES = {
    "15min": "15min",
    "15m": "15min",
    "1h": "1h",
    "h": "1h",
    "hourly": "1h",
    "1d": "1d",
    "d": "1d",
    "daily": "1d",
    "1w": "1w",
    "w": "1w",
    "weekly": "1w",
    "1mo": "1mo",
    "mo": "1mo",
    "monthly": "1mo",
    "1q": "1q",
    "q": "1q",
    "quarterly": "1q",
}

# Exact source/frequency pairs and their adapter constructor arguments.
_FREQ_CONSTRUCTORS = {
    ("yahoo", "15min"): {"interval": "15m"},
    ("yahoo", "1h"): {"interval": "1h"},
    ("yahoo", "1d"): {"interval": "1d"},
    ("yahoo", "1w"): {"interval": "1wk"},
    ("yahoo", "1mo"): {"interval": "1mo"},
    ("yahoo", "1q"): {"interval": "3mo"},
    ("coingecko", "1h"): {"interval": "1h"},
    ("coingecko", "1d"): {"interval": "1d"},
    ("owid", "1d"): {"resample": ""},
    ("owid", "1w"): {"resample": "W-SUN"},
    ("owid", "1mo"): {"resample": "ME"},
    ("owid", "1q"): {"resample": "QE"},
    ("polymarket", "1h"): {"interval": "1h"},
    ("polymarket", "1d"): {"interval": "1d"},
    ("eia", "1d"): {},
    ("noaa", "1d"): {},
    ("noaa_isd", "1h"): {},
    ("fred", "1d"): {},
    ("fred", "1w"): {},
    ("fred", "1mo"): {},
    ("fred", "1q"): {},
}


def normalize_freq(freq: str) -> str:
    """Normalize a common frequency alias."""
    return _FREQ_ALIASES.get(freq.lower().strip(), freq)


def constructor_options_for_source(source_name: str, freq: str) -> dict:
    """Return constructor options for an Agent-selected source and frequency."""
    normalized = normalize_freq(freq)
    constructor_key = (source_name, normalized)
    if constructor_key in _FREQ_CONSTRUCTORS:
        return dict(_FREQ_CONSTRUCTORS[constructor_key])

    supported = sorted(f for source, f in _FREQ_CONSTRUCTORS if source == source_name)
    raise ValueError(
        f"Frequency {normalized!r} is not supported by exact source {source_name!r}. "
        f"Supported frequencies: {', '.join(supported) if supported else 'none'}"
    )


def build_adapter(source_name: str, constructor_kwargs: dict) -> SeriesSource:
    """Construct a registered series adapter with explicit parameters."""
    from world2ts_core.bootstrap import ensure_sources_registered
    from world2ts_core.registry import get_series_source_class

    ensure_sources_registered()
    cls = get_series_source_class(source_name)
    if cls is None:
        raise ValueError(
            f"Unknown series source: {source_name}. "
            f"Register it with @register_series in world2ts_core/series/."
        )
    return cls(**constructor_kwargs)

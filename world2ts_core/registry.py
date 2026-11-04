"""Capability models and source registry.

Each data source (event or series) declares a CAPABILITY class attribute
describing what it can do. Decorators register source classes so the agent
can query available sources by domain, event type, or kind.
"""

from __future__ import annotations

from typing import List, Optional, Type

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Capability models
# ---------------------------------------------------------------------------

class EventSourceCapability(BaseModel):
    """Describes what an event source can provide."""

    name: str
    kind: str
    domains: List[str]
    entity_types: List[str]
    event_types: List[str]
    geo: List[str]                               # ["US", "global"]
    history_days: int
    time_resolution: str                         # "minute" | "hour" | "day" | "quarter"
    rate_per_min: int
    auth_required: bool
    reliability: str                             # "high" | "medium" | "low"
    description: str
    query_format: str                            # "keyword" | "symbol" | "symbol+date_range"
    required_params: List[str]
    optional_params: List[str]
    output_fields: List[str]


class SeriesSourceCapability(BaseModel):
    """Describes what a time series source can provide."""

    name: str
    domains: List[str]
    entity_types: List[str]
    frequencies: List[str]
    geo: List[str]                               # ["US", "global"]
    history_days: int
    rate_per_min: int
    auth_required: bool
    reliability: str                             # "high" | "medium" | "low"
    description: str
    primary_target_fields: List[str]
    target_option: str
    adapter_options: dict[str, List[str]]
    metadata_dimensions: List[str]
    timestamp_semantics: str
    range_semantics: str


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_EVENT_REGISTRY: List[Type] = []
_SERIES_REGISTRY: List[Type] = []


def register_event(cls: Type) -> Type:
    """Class decorator — adds *cls* to the event source registry."""
    _EVENT_REGISTRY.append(cls)
    return cls


def register_series(cls: Type) -> Type:
    """Class decorator — adds *cls* to the series source registry."""
    _SERIES_REGISTRY.append(cls)
    return cls


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def list_event_sources(
    domain: Optional[str] = None,
    event_type: Optional[str] = None,
) -> List[EventSourceCapability]:
    """Return event source capabilities matching all given filters."""
    results = [cls.capability for cls in _EVENT_REGISTRY]
    if domain:
        results = [c for c in results if domain in c.domains]
    if event_type:
        results = [c for c in results if event_type in c.event_types]
    return results


def list_series_sources(
    domain: Optional[str] = None,
) -> List[SeriesSourceCapability]:
    """Return series source capabilities matching the given domain."""
    results = [cls.capability for cls in _SERIES_REGISTRY]
    if domain:
        results = [c for c in results if domain in c.domains]
    return results


def get_event_source_class(name: str) -> Optional[Type]:
    """Return the registered event source class by capability.name, or None."""
    for cls in _EVENT_REGISTRY:
        if cls.capability.name == name:
            return cls
    return None


def get_series_source_class(name: str) -> Optional[Type]:
    """Return the registered series source class by capability.name, or None."""
    for cls in _SERIES_REGISTRY:
        if cls.capability.name == name:
            return cls
    return None


def event_source_classes_by_name() -> dict:
    """Return {name: cls} for all registered event sources."""
    return {cls.capability.name: cls for cls in _EVENT_REGISTRY}


def series_source_classes_by_name() -> dict:
    """Return {name: cls} for all registered series sources."""
    return {cls.capability.name: cls for cls in _SERIES_REGISTRY}

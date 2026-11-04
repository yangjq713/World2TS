"""Canonical lookup for Web-search adapters used by offline discovery."""

from __future__ import annotations

from world2ts_core.bootstrap import ensure_sources_registered
from world2ts_core.registry import event_source_classes_by_name


def get_search_source_classes() -> dict[str, type]:
    """Return registered adapters implementing the search protocol."""

    ensure_sources_registered()
    return {
        name: source_class
        for name, source_class in event_source_classes_by_name().items()
        if callable(getattr(source_class, "search", None))
    }

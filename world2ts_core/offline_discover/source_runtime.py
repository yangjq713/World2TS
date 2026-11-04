"""Search-source runtime shared by offline discovery phases."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

from world2ts_core.events import DiscoveredEvent
from world2ts_core.search_sources import get_search_source_classes


SourceAvailabilityFn = Callable[[str], bool]
SEARCH_SOURCES = ("tavily", "anysearch")
_SOURCE_ADAPTERS: dict[str, object] = {}
_SEARCH_CACHE: dict[tuple[str, str, str, str], list[DiscoveredEvent]] = {}
_SOURCE_FAILURES: dict[str, str] = {}
_SEARCH_LOCK = threading.RLock()


def planned_source_names(
    requested_sources: list[str],
    *,
    source_available_fn: SourceAvailabilityFn | None = None,
) -> list[str]:
    """Resolve the strict general-search source set without hidden fallbacks."""
    requested = list(dict.fromkeys(requested_sources))
    unknown = sorted(set(requested) - set(SEARCH_SOURCES))
    if unknown:
        raise ValueError(
            "offline-discover supports only Tavily and AnySearch: "
            + ", ".join(unknown)
        )
    if requested:
        return requested

    available = source_available_fn or source_available
    defaults = [name for name in SEARCH_SOURCES if available(name)]
    if not defaults:
        raise ValueError(
            "No offline-discover search source is available. "
            "Configure Tavily or AnySearch."
        )
    return defaults


def source_available(source_name: str) -> bool:
    if source_name not in SEARCH_SOURCES:
        return False
    source_classes = get_search_source_classes()
    if source_name not in source_classes:
        return False
    if source_name == "tavily":
        from world2ts_core.sources.tavily import has_tavily_api_key_configured

        return has_tavily_api_key_configured()
    return True  # AnySearch supports anonymous access.


def _search_source(
    source_name: str,
    query: str,
    start: datetime,
    end: datetime,
) -> list[DiscoveredEvent]:
    source_classes = get_search_source_classes()
    if source_name not in source_classes:
        raise ValueError(f"Unknown event source: {source_name}")
    cache_key = (source_name, query.strip().lower(), start.isoformat(), end.isoformat())
    with _SEARCH_LOCK:
        if cache_key in _SEARCH_CACHE:
            return list(_SEARCH_CACHE[cache_key])
        adapter = _SOURCE_ADAPTERS.get(source_name)
        if adapter is None:
            adapter = source_classes[source_name]()
            _SOURCE_ADAPTERS[source_name] = adapter

    results = adapter.search(query=query, start=start, end=end)
    with _SEARCH_LOCK:
        _SOURCE_FAILURES.pop(source_name, None)
        _SEARCH_CACHE[cache_key] = list(results)
    return list(results)


def reset_source_runtime_state() -> None:
    with _SEARCH_LOCK:
        _SOURCE_FAILURES.clear()


def get_source_failures() -> dict[str, str]:
    with _SEARCH_LOCK:
        return dict(_SOURCE_FAILURES)


def record_source_failure(source_name: str, message: str) -> None:
    with _SEARCH_LOCK:
        _SOURCE_FAILURES[source_name] = message


def search_sources_concurrent(
    source_names: list[str],
    query: str,
    start: datetime,
    end: datetime,
    *,
    search_fn,
    max_workers: int = 4,
) -> list[tuple[str, list[DiscoveredEvent]]]:
    """Search sources concurrently while preserving configured source order."""
    names = list(source_names)
    if not names:
        return []
    if len(names) == 1:
        return [(names[0], list(search_fn(names[0], query, start, end)))]
    with ThreadPoolExecutor(max_workers=min(len(names), max(1, max_workers))) as executor:
        return list(
            executor.map(
                lambda name: (name, list(search_fn(name, query, start, end))),
                names,
            )
        )

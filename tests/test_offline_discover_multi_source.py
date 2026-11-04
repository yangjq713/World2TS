from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import patch

import pytest

from world2ts_core.events import DiscoveredEvent
from world2ts_core.offline_discover import source_runtime
from world2ts_core.offline_discover.source_runtime import planned_source_names


class TestDefaultSourceSet:
    def test_default_is_tavily_and_anysearch(self):
        assert planned_source_names([], source_available_fn=lambda n: True) == ["tavily", "anysearch"]

    def test_default_subset_when_one_unavailable(self):
        assert planned_source_names([], source_available_fn=lambda n: n == "anysearch") == ["anysearch"]

    def test_explicit_sources_pass_through_unchanged(self):
        assert planned_source_names(
            ["anysearch", "tavily"], source_available_fn=lambda n: True
        ) == ["anysearch", "tavily"]

    def test_does_not_fall_back_to_unapproved_sources(self):
        def avail(name):
            return name == "other_source"

        with pytest.raises(ValueError):
            planned_source_names([], source_available_fn=avail)

    def test_raises_when_nothing_available(self):
        with pytest.raises(ValueError):
            planned_source_names([], source_available_fn=lambda n: False)


def _mk_event(url, eid, source):
    return DiscoveredEvent(
        event_id=eid,
        title=f"title {eid}",
        description="body",
        timestamp=datetime(2015, 3, 17),
        timestamp_precision="day",
        source_name=source,
        source_kind="news_search",
        evidence_quality="web_snippet",
        structured_payload={},
        url=url,
        retrieval_rank=0,
        retrieval_score=0.5,
    )


class _FakeSource:
    capability = None

    def __init__(self, source_name):
        self._name = source_name

    def search(self, query, start, end, **kwargs):
        return [_mk_event(f"https://{self._name}.example/{query}", f"{self._name}_1", self._name)]


class TestSearchSourceThreadSafe:
    def test_concurrent_search_source_calls_are_consistent(self):
        fake_sources = {name: (lambda n=name: _FakeSource(n)) for name in ("s1", "s2", "s3", "s4")}
        source_runtime._SEARCH_CACHE.clear()
        source_runtime._SOURCE_ADAPTERS.clear()
        source_runtime.reset_source_runtime_state()
        with patch(
            "world2ts_core.offline_discover.source_runtime.get_search_source_classes",
            return_value=fake_sources,
        ):
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(
                    lambda n: source_runtime._search_source(
                        n,
                        "q",
                        datetime(2015, 3, 10),
                        datetime(2015, 3, 24),
                    ),
                    [n for n in ("s1", "s2", "s3", "s4")] * 8,
                ))
        assert all(len(r) == 1 for r in results)
        assert source_runtime.get_source_failures() == {}

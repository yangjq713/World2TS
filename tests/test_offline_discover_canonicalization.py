from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from world2ts_core.events import DiscoveredEvent
from world2ts_core.offline_discover.canonicalization import (
    validate_canonical_events,
)
from world2ts_core.offline_discover.models import (
    AgentEventSelectionRecord,
    CanonicalEventRecord,
    CurationStrategy,
)


JAN_1 = datetime(2024, 1, 1)
JAN_5 = datetime(2024, 1, 5)
JAN_10 = datetime(2024, 1, 10)


def _strategy(
    *,
    allow_forecasts: bool = True,
    allow_retrospective: bool = True,
) -> CurationStrategy:
    return CurationStrategy.model_validate(
        {
            "schema_version": "2.0",
            "dataset_id": "energy",
            "greenfield": True,
            "dataset_understanding": "Daily regional energy demand.",
            "event_policy": {
                "allow_weak_context": True,
                "allow_retrospective_context": allow_retrospective,
                "allow_forecasts": allow_forecasts,
                "exclude_target_only_milestones": True,
                "canonical_date_preference": ["actual_start", "effective_date"],
                "duplicate_policy": "merge_same_episode",
                "compound_record_policy": "split_distinct_episodes",
            },
            "search_policy": {
                "default_sources": ["tavily", "anysearch"],
                "top_k": 10,
                "initial_queries_per_outline": 1,
                "max_refinement_queries_per_window": 1,
                "time_window_period": "monthly",
                "stop_policy": "outline coverage",
                "holiday_calendars": [
                    {
                        "country_code": "US",
                        "timezone": "UTC",
                        "target_series_ids": ["series_1"],
                    }
                ],
            },
            "search_intent_outline": [
                {
                    "outline_id": "weather",
                    "event_family": "weather",
                    "priority": "high",
                    "why_relevant": "Weather affects demand.",
                    "context_pathways": ["weather -> operations"],
                    "target_dimensions": ["region"],
                    "time_window_period": "monthly",
                    "initial_query_roles": ["broad"],
                    "in_scope_examples": ["storm"],
                    "out_of_scope_examples": ["sports"],
                    "completion_evidence": "Major weather episodes reviewed.",
                }
            ],
            "projection_policy": {
                "allowed_dimensions": ["region"],
                "allow_dataset_wide": True,
                "ambiguous_scope_action": "hold_for_agent_review",
                "link_time_policy": "event_must_overlap_series_coverage",
            },
            "quality_policy": {
                "zero_event_series_allowed": True,
                "holiday_share_is_diagnostic_only": True,
                "require_user_final_approval": True,
            },
        }
    )


def _candidate(
    source_id: str,
    *,
    source: str = "tavily",
    source_kind: str = "web_search",
    timestamp: datetime = JAN_10,
    window_id: str = "january",
    title: str = "Regional winter storm disrupts power service",
    description: str = "A winter storm disrupted power service in the north.",
    target_only: bool = False,
) -> DiscoveredEvent:
    return DiscoveredEvent(
        event_id=f"raw_{source_id}",
        event_summary_id=source_id,
        source_name=source,
        source_kind=source_kind,
        title=title,
        description=description,
        evidence_quality="web_snippet",
        timestamp=timestamp,
        window_id=window_id,
        structured_payload={"target_only_milestone": target_only},
    )


def _selection(source_ids: list[str], *, window_id: str = "january"):
    return AgentEventSelectionRecord(
        window_id=window_id,
        decision="accept",
        selected_event_summary_ids=source_ids,
        completion_evidence="Candidate evidence reviewed.",
        confidence=0.8,
        selected_by="agent_active_review",
    )


def _canonical(
    event_id: str,
    source_ids: list[str],
    *,
    start: datetime = JAN_5,
    end: datetime | None = None,
    summary: str = (
        "On January 5, 2024, a winter storm disrupted electricity service "
        "across the northern region."
    ),
    context_role: str = "actual",
    notes: str = "Sources describe one weather episode.",
) -> CanonicalEventRecord:
    return CanonicalEventRecord(
        event_id=event_id,
        event_start=start,
        event_end=end,
        summary=summary,
        event_family="weather",
        context_role=context_role,
        affected_scope_claim="Northern region electricity service",
        source_event_summary_ids=source_ids,
        canonical_date_basis="Official incident start date.",
        merge_or_split_notes=notes,
        confidence=0.8,
    )


def _write_json(path: Path, model) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, models: list) -> None:
    path.write_text(
        "".join(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for model in models
        ),
        encoding="utf-8",
    )


def _run(
    tmp_path: Path,
    *,
    candidates: list[DiscoveredEvent],
    selections: list[AgentEventSelectionRecord],
    events: list[CanonicalEventRecord],
    strategy: CurationStrategy | None = None,
) -> dict:
    _write_json(
        tmp_path / "curation_strategy.approved.json",
        strategy or _strategy(),
    )
    _write_jsonl(tmp_path / "discovered_events.jsonl", candidates)
    _write_jsonl(tmp_path / "agent_event_selections.jsonl", selections)
    _write_jsonl(tmp_path / "canonical_events.jsonl", events)
    return validate_canonical_events(tmp_path)


def test_agent_merges_duplicate_sources_into_one_canonical_episode(tmp_path) -> None:
    candidates = [
        _candidate("summary_a", window_id="january"),
        _candidate(
            "summary_b",
            source="anysearch",
            window_id="february",
            title="Northern storm caused widespread outages",
            description="The same storm caused outages across the northern region.",
        ),
    ]
    result = _run(
        tmp_path,
        candidates=candidates,
        selections=[
            _selection(["summary_a"], window_id="january"),
            _selection(["summary_b"], window_id="february"),
        ],
        events=[_canonical("event_storm", ["summary_a", "summary_b"])],
    )

    assert result["canonical_event_count"] == 1
    assert result["selected_source_event_count"] == 2
    assert result["duplicate_hints"] == []


def test_duplicate_hint_does_not_merge_agent_events(tmp_path) -> None:
    candidates = [
        _candidate("summary_a", window_id="january"),
        _candidate(
            "summary_b",
            source="anysearch",
            window_id="february",
            title="Northern storm outage update",
            description="An update covered the northern winter storm outage.",
        ),
    ]
    events = [
        _canonical("event_a", ["summary_a"]),
        _canonical(
            "event_b",
            ["summary_b"],
            summary=(
                "On January 5, 2024, the northern region winter storm "
                "disrupted electricity service."
            ),
        ),
    ]
    result = _run(
        tmp_path,
        candidates=candidates,
        selections=[
            _selection(["summary_a"], window_id="january"),
            _selection(["summary_b"], window_id="february"),
        ],
        events=events,
    )

    assert result["canonical_event_count"] == 2
    assert result["duplicate_hints"][0]["event_ids"] == ["event_a", "event_b"]
    assert result["duplicate_hints"][0]["source_names"] == ["anysearch", "tavily"]
    assert result["duplicate_hints"][0]["window_ids"] == ["february", "january"]


def test_agent_can_split_a_compound_source_into_distinct_episodes(tmp_path) -> None:
    candidate = _candidate(
        "summary_compound",
        title="Storm and later restoration described in one report",
        description="One report describes both the storm onset and a later restoration.",
    )
    result = _run(
        tmp_path,
        candidates=[candidate],
        selections=[_selection(["summary_compound"])],
        events=[
            _canonical(
                "event_onset",
                ["summary_compound"],
                start=JAN_1,
                summary=(
                    "On January 1, 2024, a winter storm interrupted electricity "
                    "service in the northern region."
                ),
                notes="The compound report was split at the incident onset.",
            ),
            _canonical(
                "event_restoration",
                ["summary_compound"],
                start=JAN_10,
                summary=(
                    "On January 10, 2024, utilities restored electricity service "
                    "after the northern region storm."
                ),
                notes="The restoration is a distinct dated episode.",
            ),
        ],
    )

    assert result["canonical_event_count"] == 2
    assert result["duplicate_hints"][0]["reasons"] == ["shared_source_event"]


def test_effective_date_may_precede_article_date(tmp_path) -> None:
    result = _run(
        tmp_path,
        candidates=[_candidate("summary_article", timestamp=JAN_10)],
        selections=[_selection(["summary_article"])],
        events=[
            _canonical(
                "event_effective",
                ["summary_article"],
                start=JAN_1,
                summary=(
                    "A regional emergency order took effect on January 1, 2024, "
                    "for northern electricity operators."
                ),
            )
        ],
    )

    assert result["status"] == "valid"


def test_calendar_page_uses_agent_canonical_event_date(tmp_path) -> None:
    candidate = _candidate(
        "summary_calendar",
        source_kind="calendar",
        timestamp=JAN_1,
        title="Northern region emergency schedule",
        description="The calendar lists the January emergency exercise.",
    )
    result = _run(
        tmp_path,
        candidates=[candidate],
        selections=[_selection(["summary_calendar"])],
        events=[
            _canonical(
                "event_calendar",
                ["summary_calendar"],
                start=JAN_10,
                summary=(
                    "On January 10, 2024, the northern region conducted its "
                    "scheduled electricity emergency exercise."
                ),
            )
        ],
    )

    assert result["status"] == "valid"


def test_retrospective_context_obeys_approved_strategy(tmp_path) -> None:
    candidate = _candidate(
        "summary_review",
        title="Annual review examines northern grid storm",
        description="A later review assessed the January grid disruption.",
    )
    event = _canonical(
        "event_review",
        ["summary_review"],
        context_role="retrospective",
        summary=(
            "A retrospective published on January 10, 2024, reviewed the "
            "northern grid's response to the winter storm."
        ),
    )
    assert _run(
        tmp_path,
        candidates=[candidate],
        selections=[_selection(["summary_review"])],
        events=[event],
    )["status"] == "valid"

    with pytest.raises(ValueError, match="retrospective context is disallowed"):
        _run(
            tmp_path,
            candidates=[candidate],
            selections=[_selection(["summary_review"])],
            events=[event],
            strategy=_strategy(allow_retrospective=False),
        )


def test_target_only_milestone_is_rejected(tmp_path) -> None:
    candidate = _candidate(
        "summary_target",
        title="Demand reached a monthly high",
        description="The target series reached its highest January value.",
        target_only=True,
    )
    with pytest.raises(ValueError, match="target-derived milestones"):
        _run(
            tmp_path,
            candidates=[candidate],
            selections=[_selection(["summary_target"])],
            events=[_canonical("event_target", ["summary_target"])],
        )


def test_canonical_sources_must_be_selected_in_the_current_run(tmp_path) -> None:
    with pytest.raises(ValueError, match="outside the current run"):
        _run(
            tmp_path,
            candidates=[_candidate("summary_current")],
            selections=[_selection(["summary_current"])],
            events=[_canonical("event_old", ["summary_from_old_run"])],
        )

    with pytest.raises(ValueError, match="unselected candidate"):
        _run(
            tmp_path,
            candidates=[
                _candidate("summary_selected"),
                _candidate("summary_unselected"),
            ],
            selections=[_selection(["summary_selected"])],
            events=[_canonical("event_unselected", ["summary_unselected"])],
        )


def test_reject_selection_cannot_smuggle_selected_candidates() -> None:
    with pytest.raises(ValueError, match="invalid for reject"):
        AgentEventSelectionRecord(
            window_id="january",
            decision="reject",
            selected_event_summary_ids=["summary_a"],
            selected_by="agent_active_review",
        )


def test_selected_source_may_be_dropped_during_final_curation(tmp_path) -> None:
    result = _run(
        tmp_path,
        candidates=[
            _candidate("summary_a"),
            _candidate("summary_b", source="anysearch"),
        ],
        selections=[_selection(["summary_a", "summary_b"])],
        events=[_canonical("event_a", ["summary_a"])],
    )

    assert result["uncanonicalized_selected_source_event_ids"] == ["summary_b"]


def test_retrieved_headline_or_snippet_cannot_be_public_summary(tmp_path) -> None:
    candidate = _candidate("summary_raw")
    for raw_text in (candidate.title, candidate.description):
        with pytest.raises(ValueError, match="must rewrite"):
            _run(
                tmp_path,
                candidates=[candidate],
                selections=[_selection(["summary_raw"])],
                events=[
                    _canonical(
                        "event_raw",
                        ["summary_raw"],
                        summary=raw_text,
                    )
                ],
            )


def test_canonical_artifact_is_strict_jsonl(tmp_path) -> None:
    strategy = _strategy()
    candidate = _candidate("summary_a")
    event = _canonical("event_a", ["summary_a"])
    _write_json(tmp_path / "curation_strategy.approved.json", strategy)
    _write_jsonl(tmp_path / "discovered_events.jsonl", [candidate])
    _write_jsonl(tmp_path / "agent_event_selections.jsonl", [_selection(["summary_a"])])
    (tmp_path / "canonical_events.jsonl").write_text(
        json.dumps([event.model_dump(mode="json")]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be JSONL"):
        validate_canonical_events(tmp_path)

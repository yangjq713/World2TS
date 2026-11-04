from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from world2ts_core.offline_discover.models import (
    CanonicalEventRecord,
    CurationStrategy,
    DataReadinessReport,
    DimensionReadinessProfile,
    SemanticProjectionFilter,
    SemanticProjectionRule,
    SeriesReadinessProfile,
    SeriesSemanticDimensionsRecord,
)
from world2ts_core.offline_discover.projection import (
    _event_overlaps_profile,
    load_projected_links,
    project_events,
    validate_projected_links,
)


START = datetime(2024, 1, 1)
END = datetime(2024, 1, 31)


def _strategy(
    *,
    allowed_dimensions: list[str] | None = None,
    allow_dataset_wide: bool = True,
) -> CurationStrategy:
    allowed_dimensions = allowed_dimensions or [
        "city",
        "primary_use",
        "variable",
        "capacity",
    ]
    return CurationStrategy.model_validate(
        {
            "schema_version": "2.0",
            "dataset_id": "operations",
            "greenfield": True,
            "dataset_understanding": "Daily operational measurements.",
            "event_policy": {
                "allow_weak_context": True,
                "allow_retrospective_context": True,
                "allow_forecasts": True,
                "exclude_target_only_milestones": True,
                "canonical_date_preference": ["actual_start"],
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
                        "timezone": "America/New_York",
                        "target_series_ids": [
                            "boston_school",
                            "boston_office",
                            "chicago_school",
                        ],
                    }
                ],
            },
            "search_intent_outline": [
                {
                    "outline_id": "disruption",
                    "event_family": "disruption",
                    "priority": "high",
                    "why_relevant": "Disruptions affect operations.",
                    "context_pathways": ["disruption -> operations"],
                    "target_dimensions": ["city"],
                    "time_window_period": "monthly",
                    "initial_query_roles": ["broad"],
                    "in_scope_examples": ["outage"],
                    "out_of_scope_examples": ["sports"],
                    "completion_evidence": "Major disruptions reviewed.",
                }
            ],
            "projection_policy": {
                "allowed_dimensions": allowed_dimensions,
                "allow_dataset_wide": allow_dataset_wide,
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


def _semantics() -> list[SeriesSemanticDimensionsRecord]:
    return [
        SeriesSemanticDimensionsRecord(
            series_id="boston_school",
            confidence=1.0,
            dimensions={
                "city": "Boston",
                "primary_use": "education",
                "variable": ["pm25", "temperature"],
                "capacity": 80,
            },
        ),
        SeriesSemanticDimensionsRecord(
            series_id="boston_office",
            confidence=1.0,
            dimensions={
                "city": "Boston",
                "primary_use": "office",
                "variable": ["temperature"],
                "capacity": 40,
            },
        ),
        SeriesSemanticDimensionsRecord(
            series_id="chicago_school",
            confidence=1.0,
            dimensions={
                "city": "Chicago",
                "primary_use": "education",
                "variable": ["pm25"],
                "capacity": 120,
            },
        ),
    ]


def _readiness(
    semantics: list[SeriesSemanticDimensionsRecord],
    *,
    starts: dict[str, datetime] | None = None,
) -> DataReadinessReport:
    starts = starts or {}
    return DataReadinessReport(
        dataset_id="operations",
        status="ready",
        arrow_files=["data.arrow"],
        series_count=len(semantics),
        series_ids=[series.series_id for series in semantics],
        series_id_order_sha256="a" * 64,
        series_profiles=[
            SeriesReadinessProfile(
                series_id=series.series_id,
                start=starts.get(series.series_id, START),
                end=END,
                freq="1d",
                length=31,
                missing_rate=0.0,
                has_explicit_timestamps=False,
            )
            for series in semantics
        ],
        dataset_semantics_coverage=1.0,
        series_semantics_coverage=1.0,
        dimensions=[
            DimensionReadinessProfile(
                dimension=dimension,
                value_type="mixed",
                populated_series_count=len(semantics),
                coverage_ratio=1.0,
            )
            for dimension in ["city", "primary_use", "variable", "capacity"]
        ],
    )


def _event(event_id: str = "event_outage") -> CanonicalEventRecord:
    return CanonicalEventRecord(
        event_id=event_id,
        event_start=datetime(2024, 1, 10),
        summary=(
            "On January 10, 2024, an outage disrupted operations in Boston."
        ),
        event_family="disruption",
        context_role="actual",
        affected_scope_claim="Boston operations",
        source_event_summary_ids=["summary_1"],
        canonical_date_basis="Official incident date.",
        merge_or_split_notes="One outage episode.",
        confidence=0.9,
    )


def _rule(
    *,
    rule_id: str = "rule_1",
    event_id: str = "event_outage",
    scope: str = "dimension_filter",
    filters: list[SemanticProjectionFilter] | None = None,
    combine: str = "all",
) -> SemanticProjectionRule:
    return SemanticProjectionRule(
        rule_id=rule_id,
        event_id=event_id,
        scope=scope,
        filters=filters
        if filters is not None
        else [
            SemanticProjectionFilter(
                dimension="city",
                operator="eq",
                value="Boston",
            )
        ],
        combine=combine,
        basis="The event scope explicitly names the affected series.",
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
    rules: list[SemanticProjectionRule],
    events: list[CanonicalEventRecord] | None = None,
    semantics: list[SeriesSemanticDimensionsRecord] | None = None,
    readiness: DataReadinessReport | None = None,
    strategy: CurationStrategy | None = None,
) -> dict:
    events = events or [_event()]
    semantics = semantics or _semantics()
    readiness = readiness or _readiness(semantics)
    _write_json(
        tmp_path / "curation_strategy.approved.json",
        strategy or _strategy(),
    )
    _write_json(tmp_path / "data_readiness_report.json", readiness)
    _write_jsonl(tmp_path / "canonical_events.jsonl", events)
    _write_jsonl(tmp_path / "series_semantic_dimensions.jsonl", semantics)
    _write_jsonl(tmp_path / "projection_rules.jsonl", rules)
    return project_events(tmp_path)


def test_category_broadcast_and_stable_links(tmp_path) -> None:
    rule = _rule(
        filters=[
            SemanticProjectionFilter(
                dimension="primary_use",
                operator="in",
                value=["education", "office"],
            )
        ]
    )
    result = _run(tmp_path, rules=[rule])
    links = load_projected_links(tmp_path)

    assert result["link_count"] == 3
    assert [link.series_id for link in links] == [
        "boston_office",
        "boston_school",
        "chicago_school",
    ]
    assert all(link.projection_rule_id == "rule_1" for link in links)
    assert all(link.confidence == 0.8 for link in links)


def test_city_and_variable_conjunction(tmp_path) -> None:
    rule = _rule(
        filters=[
            SemanticProjectionFilter(
                dimension="city",
                operator="eq",
                value="Boston",
            ),
            SemanticProjectionFilter(
                dimension="variable",
                operator="in",
                value=["pm25"],
            ),
        ],
        combine="all",
    )
    _run(tmp_path, rules=[rule])

    assert [link.series_id for link in load_projected_links(tmp_path)] == [
        "boston_school"
    ]


def test_any_filters_exists_and_between(tmp_path) -> None:
    rule = _rule(
        filters=[
            SemanticProjectionFilter(
                dimension="capacity",
                operator="between",
                value=[100, 130],
            ),
            SemanticProjectionFilter(
                dimension="variable",
                operator="exists",
            ),
        ],
        combine="any",
    )
    result = _run(tmp_path, rules=[rule])

    assert result["link_count"] == 3


def test_dataset_wide_projection_requires_strategy_authorization(tmp_path) -> None:
    rule = _rule(scope="dataset", filters=[])
    assert _run(tmp_path, rules=[rule])["link_count"] == 3

    with pytest.raises(ValueError, match="not strategy-approved"):
        _run(
            tmp_path,
            rules=[rule],
            strategy=_strategy(allow_dataset_wide=False),
        )


def test_strategy_and_readiness_must_belong_to_the_same_dataset(tmp_path) -> None:
    mismatched = _strategy().model_copy(update={"dataset_id": "other"})

    with pytest.raises(ValueError, match="dataset IDs must match"):
        _run(
            tmp_path,
            rules=[_rule()],
            strategy=mismatched,
        )


def test_illegal_dimension_and_unknown_value_fail_loud(tmp_path) -> None:
    with pytest.raises(ValueError, match="not approved"):
        _run(
            tmp_path,
            rules=[
                _rule(
                    filters=[
                        SemanticProjectionFilter(
                            dimension="provider",
                            operator="eq",
                            value="utility",
                        )
                    ]
                )
            ],
        )

    with pytest.raises(ValueError, match="unknown values"):
        _run(
            tmp_path,
            rules=[
                _rule(
                    filters=[
                        SemanticProjectionFilter(
                            dimension="city",
                            operator="eq",
                            value="Seattle",
                        )
                    ]
                )
            ],
        )


def test_time_policy_excludes_series_outside_event_coverage(tmp_path) -> None:
    semantics = _semantics()
    readiness = _readiness(
        semantics,
        starts={"boston_office": datetime(2024, 1, 20)},
    )
    result = _run(
        tmp_path,
        rules=[_rule()],
        semantics=semantics,
        readiness=readiness,
    )

    assert result["link_count"] == 1
    assert load_projected_links(tmp_path)[0].series_id == "boston_school"


def test_time_policy_preserves_legacy_awareness_and_normalizes_explicit_utc() -> None:
    profile = _readiness(_semantics()).series_profiles[0]
    with pytest.raises(ValueError, match="legacy inferred timeline"):
        _event_overlaps_profile(
            datetime(2024, 1, 10, tzinfo=timezone.utc),
            None,
            profile,
        )

    explicit_profile = profile.model_copy(
        update={
            "start": datetime(2024, 1, 1, 5),
            "end": datetime(2024, 1, 31, 5),
            "has_explicit_timestamps": True,
        }
    )
    assert _event_overlaps_profile(
        datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=-5))),
        None,
        explicit_profile,
    )


def test_zero_match_unknown_event_and_duplicate_event_rule_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="matches no series"):
        _run(
            tmp_path,
            rules=[
                _rule(
                    filters=[
                        SemanticProjectionFilter(
                            dimension="capacity",
                            operator="between",
                            value=[1000, 2000],
                        )
                    ]
                )
            ],
        )

    with pytest.raises(ValueError, match="unknown canonical event"):
        _run(tmp_path, rules=[_rule(event_id="event_missing")])

    with pytest.raises(ValueError, match="at most one projection rule"):
        _run(
            tmp_path,
            rules=[
                _rule(rule_id="rule_1"),
                _rule(rule_id="rule_2"),
            ],
        )


def test_missing_rule_holds_ambiguous_event_without_broadcast(tmp_path) -> None:
    result = _run(tmp_path, rules=[])

    assert result["held_event_ids"] == ["event_outage"]
    assert result["link_count"] == 0
    assert load_projected_links(tmp_path) == []


def test_stored_links_must_match_current_projection_inputs(tmp_path) -> None:
    _run(tmp_path, rules=[_rule()])
    assert len(validate_projected_links(tmp_path)) == 2

    (tmp_path / "projected_event_series_links.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale or does not match"):
        validate_projected_links(tmp_path)

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from world2ts_core.offline_discover.holiday_calendar import country_holiday_dates
from world2ts_core.events import DiscoveredEvent
from world2ts_core.offline_discover.holiday_impact import holiday_impact_id
from world2ts_core.offline_discover.models import (
    AgentEventSelectionRecord,
    CanonicalEventRecord,
    CurationStrategy,
    DataReadinessReport,
    DimensionReadinessProfile,
    QualityGateResult,
    QualityReport,
    SearchTraceRecord,
    SearchWindowStateRecord,
    SemanticProjectionFilter,
    SemanticProjectionRule,
    SemanticReadinessReport,
    SeriesReadinessProfile,
    SeriesSemanticDimensionsRecord,
    StrategyReviewDecision,
    TimeWindowRecord,
)
from world2ts_core.offline_discover.projection import project_events
from world2ts_core.offline_discover.quality_audit import (
    CONTENT_GATE_IDS,
    apply_quality_review,
    build_quality_audit_briefing,
    prepare_quality_review,
    quality_publish_status,
)
from world2ts_core.offline_discover.strategy import strategy_sha256


START = datetime(2024, 1, 1)
END = datetime(2024, 1, 31)
EVENT_DATE = datetime(2024, 1, 10)


def _strategy() -> CurationStrategy:
    return CurationStrategy.model_validate(
        {
            "schema_version": "2.0",
            "dataset_id": "operations",
            "greenfield": True,
            "dataset_understanding": "Daily city-level operational measurements.",
            "event_policy": {
                "allow_weak_context": True,
                "allow_retrospective_context": True,
                "allow_forecasts": False,
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
                        "target_series_ids": ["boston", "chicago"],
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
                "allowed_dimensions": ["city"],
                "allow_dataset_wide": False,
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


def _series_semantics() -> list[SeriesSemanticDimensionsRecord]:
    return [
        SeriesSemanticDimensionsRecord(
            series_id="boston",
            confidence=1.0,
            dimensions={"city": "Boston"},
        ),
        SeriesSemanticDimensionsRecord(
            series_id="chicago",
            confidence=1.0,
            dimensions={"city": "Chicago"},
        ),
    ]


def _readiness() -> DataReadinessReport:
    return DataReadinessReport(
        dataset_id="operations",
        status="ready",
        arrow_files=["data.arrow"],
        series_count=2,
        series_ids=["boston", "chicago"],
        series_id_order_sha256="a" * 64,
        series_profiles=[
            SeriesReadinessProfile(
                series_id=series_id,
                start=START,
                end=END,
                freq="1d",
                length=31,
                missing_rate=0.0,
                has_explicit_timestamps=False,
            )
            for series_id in ["boston", "chicago"]
        ],
        dataset_semantics_coverage=1.0,
        series_semantics_coverage=1.0,
        dimensions=[
            DimensionReadinessProfile(
                dimension="city",
                value_type="string",
                populated_series_count=2,
                coverage_ratio=1.0,
                value_counts={'"Boston"': 1, '"Chicago"': 1},
            )
        ],
    )


def _candidate() -> DiscoveredEvent:
    return DiscoveredEvent(
        event_id="raw_outage",
        event_summary_id="summary_outage",
        source_name="tavily",
        source_kind="web_search",
        title="Boston outage report",
        description="A report described an outage in Boston.",
        evidence_quality="web_snippet",
        timestamp=EVENT_DATE,
        window_id="january",
        structured_payload={"source_hits": ["tavily"]},
    )


def _event() -> CanonicalEventRecord:
    return CanonicalEventRecord(
        event_id="event_outage",
        event_start=EVENT_DATE,
        summary=(
            "On January 10, 2024, an electricity outage disrupted operations "
            "across Boston."
        ),
        event_family="disruption",
        context_role="actual",
        affected_scope_claim="Boston operations",
        source_event_summary_ids=["summary_outage"],
        canonical_date_basis="Official incident date.",
        merge_or_split_notes="One outage episode.",
        confidence=0.9,
    )


def _rule() -> SemanticProjectionRule:
    return SemanticProjectionRule(
        rule_id="rule_outage",
        event_id="event_outage",
        scope="dimension_filter",
        filters=[
            SemanticProjectionFilter(
                dimension="city",
                operator="eq",
                value="Boston",
            )
        ],
        basis="The outage was explicitly limited to Boston.",
        confidence=0.8,
    )


def _write_json(path: Path, value) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else value,
                ensure_ascii=False,
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _time_window() -> TimeWindowRecord:
    return TimeWindowRecord(
        window_id="january",
        period="monthly",
        start=START,
        end=END,
        active_series_ids=["boston", "chicago"],
        active_dimension_values={"city": ["Boston", "Chicago"]},
        outline_ids=["disruption"],
        priority=2.0,
    )


def _holiday_impacts(
    strategy: CurationStrategy,
    window: TimeWindowRecord,
) -> list[dict]:
    impacts = []
    for calendar in strategy.search_policy.holiday_calendars:
        active_series_ids = [
            series_id
            for series_id in calendar.target_series_ids
            if series_id in window.active_series_ids
        ]
        for holiday_date, holiday_name in country_holiday_dates(
            calendar.country_code,
            calendar.subdivision_code,
            window.start.date(),
            window.end.date(),
        ):
            impacts.append(
                {
                    "impact_id": holiday_impact_id(
                        window.window_id,
                        calendar.country_code,
                        calendar.subdivision_code,
                        calendar.timezone,
                        calendar.target_series_ids,
                        holiday_date,
                        holiday_name,
                    ),
                    "window_id": window.window_id,
                    "country_code": calendar.country_code,
                    "subdivision_code": calendar.subdivision_code,
                    "timezone": calendar.timezone,
                    "holiday_date": holiday_date.isoformat(),
                    "holiday_name": holiday_name,
                    "accepted": False,
                    "affected_series_ids": [],
                    "series_impacts": [
                        {
                            "series_id": series_id,
                            "event_count": 0,
                            "baseline_count": 0,
                            "supported": False,
                        }
                        for series_id in active_series_ids
                    ],
                }
            )
    return impacts


def _valid_run(tmp_path: Path) -> Path:
    strategy = _strategy()
    digest = strategy_sha256(strategy)
    _write_json(tmp_path / "curation_strategy.draft.json", strategy)
    _write_json(tmp_path / "curation_strategy.approved.json", strategy)
    _write_json(
        tmp_path / "strategy_review_decision.json",
        StrategyReviewDecision(
            decision="approve",
            reviewed_strategy_sha256=digest,
            reviewed_at=START,
        ),
    )

    readiness = _readiness()
    _write_json(tmp_path / "data_readiness_report.json", readiness)
    _write_json(
        tmp_path / "semantic_readiness_report.json",
        SemanticReadinessReport(
            dataset_id="operations",
            status="ready",
            expected_series_ids=["boston", "chicago"],
            validated_series_ids=["boston", "chicago"],
        ),
    )
    _write_json(
        tmp_path / "agent_dataset_semantics.json",
        {
            "dataset_id": "operations",
            "description": (
                "This dataset contains daily city-level operational measurements "
                "for Boston and Chicago."
            ),
            "domain": "operations",
            "frequency_phrases": ["daily"],
        },
    )
    _write_jsonl(
        tmp_path / "agent_series_semantics.jsonl",
        [
            {
                "series_id": series_id,
                "description": (
                    f"The {series_id} series contains daily operational "
                    f"measurements for {series_id.title()}."
                ),
                "time_range": {
                    "start": START.isoformat(),
                    "end": END.isoformat(),
                },
                "frequency_phrase": "daily",
            }
            for series_id in ["boston", "chicago"]
        ],
    )
    _write_jsonl(
        tmp_path / "series_semantic_dimensions.jsonl",
        _series_semantics(),
    )
    window = _time_window()
    _write_json(tmp_path / "time_windows.json", [window.model_dump(mode="json")])

    _write_jsonl(
        tmp_path / "agent_window_search_plans.jsonl",
        [
            {
                "window_id": "january",
                "prior_event_hypotheses": [
                    {
                        "hypothesis_id": "prior_outage",
                        "event_name": "Possible Boston outage",
                        "date_hint": "January 2024",
                        "relevance_pathway": "outage -> operations",
                        "reasoning": "A local outage could affect the series.",
                        "confidence": 0.6,
                    }
                ],
                "generated_by": "agent_active_review",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "agent_query_intents.jsonl",
        [
            {
                "query_intent_id": "query_1",
                "window_id": "january",
                "stage": "initial",
                "outline_id": "disruption",
                "query_text": "Boston outage January 2024",
                "search_start": START.isoformat(),
                "search_end": END.isoformat(),
                "context_hypothesis": "An outage may affect operations.",
                "relevance_pathway": "outage -> operations",
                "generated_by": "agent_active_review",
            }
        ],
    )
    state = SearchWindowStateRecord(
        window_id="january",
        status="covered",
        attempted_initial_query_intent_ids=["query_1"],
        normalized_queries=["boston outage january 2024"],
        candidate_event_summary_ids=["summary_outage"],
        holiday_analysis_complete=True,
    )
    _write_json(tmp_path / "search_state.json", [state.model_dump(mode="json")])
    _write_jsonl(
        tmp_path / "holiday_impacts.jsonl",
        _holiday_impacts(strategy, window),
    )
    _write_jsonl(
        tmp_path / "search_trace.jsonl",
        [
            SearchTraceRecord(
                trace_id="trace_1",
                window_id="january",
                query_intent_id="query_1",
                stage="initial",
                outline_id="disruption",
                query_text="Boston outage January 2024",
                source="tavily",
                started_at=START,
                elapsed_ms=10,
                returned_count=1,
                event_summary_ids=["summary_outage"],
                selection_decision="accept",
            ),
            SearchTraceRecord(
                trace_id="trace_2",
                window_id="january",
                query_intent_id="query_1",
                stage="initial",
                outline_id="disruption",
                query_text="Boston outage January 2024",
                source="anysearch",
                started_at=START,
                elapsed_ms=10,
                returned_count=0,
                selection_decision="accept",
            ),
        ],
    )
    _write_jsonl(tmp_path / "discovered_events.jsonl", [_candidate()])
    _write_jsonl(
        tmp_path / "agent_event_selections.jsonl",
        [
            AgentEventSelectionRecord(
                window_id="january",
                decision="accept",
                selected_event_summary_ids=["summary_outage"],
                completion_evidence="The outage evidence was reviewed.",
                confidence=0.8,
                selected_by="agent_active_review",
            )
        ],
    )
    _write_jsonl(tmp_path / "canonical_events.jsonl", [_event()])
    _write_jsonl(tmp_path / "projection_rules.jsonl", [_rule()])
    project_events(tmp_path)
    return tmp_path


def _agent_report(
    briefing: dict,
    *,
    gate_status: str = "pass",
    worth_continuing: bool = False,
) -> QualityReport:
    content_gates = [
        QualityGateResult(
            gate_id=gate_id,
            gate_type="content",
            status=gate_status if index == 0 else "pass",
            evidence=[f"Agent reviewed {gate_id}."],
            impact="No blocking issue." if gate_status != "fail" else "Revision required.",
            continue_search_recommended=worth_continuing and index == 0,
            recommended_return_phase=(
                "canonicalization"
                if worth_continuing and index == 0
                else None
            ),
        )
        for index, gate_id in enumerate(CONTENT_GATE_IDS)
    ]
    overall = "fail" if gate_status == "fail" else "warning" if gate_status == "warning" else "pass"
    return QualityReport(
        dataset_id="operations",
        generated_at=EVENT_DATE,
        audit_input_sha256=briefing["audit_input_sha256"],
        overall_status=overall,
        hard_gates=[
            QualityGateResult.model_validate(gate)
            for gate in briefing["hard_gates"]
        ],
        content_gates=content_gates,
        statistics=briefing["statistics"],
        worth_continuing_search=worth_continuing,
        recommendation=(
            "Revise event curation."
            if gate_status == "fail"
            else "The current artifact is ready for user review."
        ),
        recommended_return_phase=(
            "canonicalization" if worth_continuing else None
        ),
    )


def _prepare_report(
    tmp_path: Path,
    *,
    gate_status: str = "pass",
    worth_continuing: bool = False,
) -> tuple[dict, QualityReport]:
    briefing = build_quality_audit_briefing(tmp_path)
    report = _agent_report(
        briefing,
        gate_status=gate_status,
        worth_continuing=worth_continuing,
    )
    report_file = tmp_path / "agent_quality_report.json"
    _write_json(report_file, report)
    packet = prepare_quality_review(
        run_dir=tmp_path,
        quality_report_file=report_file,
    )
    return packet, report


def _decision(
    tmp_path: Path,
    packet: dict,
    decision: str,
    *,
    target_window_ids: list[str] | None = None,
) -> Path:
    path = tmp_path / f"{decision}.json"
    _write_json(
        path,
        {
            "decision": decision,
            "reviewed_quality_report_sha256": packet["quality_report_sha256"],
            "user_instructions": "",
            "target_window_ids": target_window_ids or [],
            "reviewed_at": EVENT_DATE.isoformat(),
        },
    )
    return path


def test_audit_briefing_covers_six_quality_dimensions_and_statistics(tmp_path) -> None:
    _valid_run(tmp_path)
    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "auditing"
    assert briefing["hard_gates"][0]["gate_id"] == "structural_integrity"
    assert briefing["required_content_gate_ids"] == list(CONTENT_GATE_IDS)
    assert briefing["statistics"]["series_count"] == 2
    assert briefing["statistics"]["event_count"] == 1
    assert briefing["statistics"]["link_count"] == 1
    assert briefing["statistics"]["events_per_series"]["zero_event_series_count"] == 1
    assert briefing["statistics"]["outline_coverage"] == {
        "disruption": {"failures": 0, "queries": 2, "returned": 1}
    }
    assert briefing["review_hints"]["zero_event_series_ids"] == ["chicago"]


def test_hard_failure_stops_agent_review_and_cannot_be_overridden(tmp_path) -> None:
    _valid_run(tmp_path)
    (tmp_path / "projected_event_series_links.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "failed"
    assert briefing["hard_gates"][0]["status"] == "fail"
    with pytest.raises(ValueError, match="Hard quality gate failed"):
        prepare_quality_review(
            run_dir=tmp_path,
            quality_report_file=tmp_path / "unused.json",
        )


@pytest.mark.parametrize(
    "failure",
    ["unfinished_search", "missing_trace"],
)
def test_audit_rejects_incomplete_search_evidence(tmp_path, failure: str) -> None:
    _valid_run(tmp_path)
    if failure == "unfinished_search":
        state = json.loads(
            (tmp_path / "search_state.json").read_text(encoding="utf-8")
        )
        state[0]["status"] = "awaiting_selection"
        _write_json(tmp_path / "search_state.json", state)
    else:
        (tmp_path / "search_trace.jsonl").write_text("", encoding="utf-8")

    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "failed"
    assert briefing["blocked_reason"] == "hard_quality_gate_failed"


def test_audit_allows_incomplete_holiday_analysis_flag(tmp_path) -> None:
    _valid_run(tmp_path)
    state = json.loads(
        (tmp_path / "search_state.json").read_text(encoding="utf-8")
    )
    state[0]["holiday_analysis_complete"] = False
    _write_json(tmp_path / "search_state.json", state)

    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "auditing"
    assert briefing["hard_gates"][0]["status"] == "pass"


def test_audit_allows_partial_trace_source_coverage(tmp_path) -> None:
    _valid_run(tmp_path)
    traces = [
        json.loads(line)
        for line in (tmp_path / "search_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _write_jsonl(
        tmp_path / "search_trace.jsonl",
        [trace for trace in traces if trace["source"] != "anysearch"],
    )

    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "auditing"
    assert set(briefing["statistics"]["source_trace"]) == {"tavily"}


def test_audit_does_not_cross_check_trace_source_against_intent(tmp_path) -> None:
    _valid_run(tmp_path)
    intents = [
        json.loads(line)
        for line in (tmp_path / "agent_query_intents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    intents[0]["source_names"] = ["tavily"]
    _write_jsonl(tmp_path / "agent_query_intents.jsonl", intents)

    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "auditing"
    assert briefing["hard_gates"][0]["status"] == "pass"


def test_audit_allows_missing_holiday_impact_evidence(tmp_path) -> None:
    _valid_run(tmp_path)
    impacts = [
        json.loads(line)
        for line in (tmp_path / "holiday_impacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert impacts
    _write_jsonl(tmp_path / "holiday_impacts.jsonl", impacts[1:])

    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "auditing"
    assert briefing["statistics"]["holiday_impact"]["evaluated"] == len(impacts) - 1


def test_audit_allows_incomplete_holiday_series_evidence(tmp_path) -> None:
    _valid_run(tmp_path)
    impacts = [
        json.loads(line)
        for line in (tmp_path / "holiday_impacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert impacts
    impacts[0]["series_impacts"] = impacts[0]["series_impacts"][:1]
    _write_jsonl(tmp_path / "holiday_impacts.jsonl", impacts)

    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "auditing"
    assert briefing["statistics"]["holiday_impact"]["evaluated"] == len(impacts)


def test_audit_interprets_naive_explicit_timestamp_windows_as_utc(tmp_path) -> None:
    _valid_run(tmp_path)

    readiness = json.loads(
        (tmp_path / "data_readiness_report.json").read_text(encoding="utf-8")
    )
    for profile in readiness["series_profiles"]:
        profile["has_explicit_timestamps"] = True
    _write_json(tmp_path / "data_readiness_report.json", readiness)

    windows = json.loads((tmp_path / "time_windows.json").read_text(encoding="utf-8"))
    windows[0]["end"] = "2024-01-15T02:00:00"
    _write_json(tmp_path / "time_windows.json", windows)

    intents = [
        json.loads(line)
        for line in (tmp_path / "agent_query_intents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    intents[0]["search_end"] = "2024-01-15T02:00:00"
    _write_jsonl(tmp_path / "agent_query_intents.jsonl", intents)

    impacts = [
        json.loads(line)
        for line in (tmp_path / "holiday_impacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    _write_jsonl(
        tmp_path / "holiday_impacts.jsonl",
        [impact for impact in impacts if impact["holiday_date"] < "2024-01-15"],
    )

    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "auditing"
    assert briefing["hard_gates"][0]["status"] == "pass"


def test_zero_event_series_policy_is_a_hard_strategy_constraint(tmp_path) -> None:
    _valid_run(tmp_path)
    strategy_payload = json.loads(
        (tmp_path / "curation_strategy.draft.json").read_text(encoding="utf-8")
    )
    strategy_payload["quality_policy"]["zero_event_series_allowed"] = False
    strategy = CurationStrategy.model_validate(strategy_payload)
    digest = strategy_sha256(strategy)
    _write_json(tmp_path / "curation_strategy.draft.json", strategy)
    _write_json(tmp_path / "curation_strategy.approved.json", strategy)
    _write_json(
        tmp_path / "strategy_review_decision.json",
        StrategyReviewDecision(
            decision="approve",
            reviewed_strategy_sha256=digest,
            reviewed_at=START,
        ),
    )

    briefing = build_quality_audit_briefing(tmp_path)

    assert briefing["state"] == "failed"
    assert "does not allow zero-event series" in briefing["hard_gates"][0]["evidence"][0]


def test_agent_report_cannot_override_hard_gates_or_statistics(tmp_path) -> None:
    _valid_run(tmp_path)
    briefing = build_quality_audit_briefing(tmp_path)
    report = _agent_report(briefing).model_dump(mode="json")
    report_file = tmp_path / "agent_quality_report.json"

    report["hard_gates"][0]["status"] = "warning"
    report["overall_status"] = "warning"
    _write_json(report_file, report)
    with pytest.raises(ValueError, match="may not alter deterministic hard gates"):
        prepare_quality_review(
            run_dir=tmp_path,
            quality_report_file=report_file,
        )

    report = _agent_report(briefing).model_dump(mode="json")
    report["statistics"]["link_count"] = 999
    _write_json(report_file, report)
    with pytest.raises(ValueError, match="may not alter deterministic statistics"):
        prepare_quality_review(
            run_dir=tmp_path,
            quality_report_file=report_file,
        )


def test_prepare_quality_review_binds_report_and_keeps_packet_concise(tmp_path) -> None:
    _valid_run(tmp_path)
    packet, _ = _prepare_report(tmp_path)

    assert packet["state"] == "awaiting_quality_review"
    review = (tmp_path / "quality_review.md").read_text(encoding="utf-8")
    assert packet["quality_report_sha256"] in review
    assert "Canonical events: 1" in review
    assert '"statistics"' not in review
    assert quality_publish_status(tmp_path)["can_publish"] is False


def test_warning_can_be_explicitly_approved(tmp_path) -> None:
    _valid_run(tmp_path)
    packet, _ = _prepare_report(tmp_path, gate_status="warning")
    result = apply_quality_review(
        run_dir=tmp_path,
        decision_file=_decision(tmp_path, packet, "approve"),
    )

    assert result["state"] == "approved"
    assert quality_publish_status(tmp_path)["can_publish"] is True


def test_report_or_audit_input_change_invalidates_approval(tmp_path) -> None:
    _valid_run(tmp_path)
    packet, _ = _prepare_report(tmp_path)
    apply_quality_review(
        run_dir=tmp_path,
        decision_file=_decision(tmp_path, packet, "approve"),
    )

    report = json.loads((tmp_path / "quality_report.json").read_text(encoding="utf-8"))
    report["recommendation"] = "Changed after review."
    _write_json(tmp_path / "quality_report.json", report)
    assert quality_publish_status(tmp_path)["blocked_reason"] == "quality_report_stale"

    _prepare_report(tmp_path)
    current_packet = prepare_quality_review(
        run_dir=tmp_path,
        quality_report_file=tmp_path / "agent_quality_report.json",
    )
    apply_quality_review(
        run_dir=tmp_path,
        decision_file=_decision(tmp_path, current_packet, "approve"),
    )
    with (tmp_path / "search_trace.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert quality_publish_status(tmp_path)["blocked_reason"] == "quality_report_stale"


@pytest.mark.parametrize(
    ("decision", "return_phase"),
    [
        ("revise_search", "search"),
        ("revise_events", "canonicalization"),
        ("revise_projection", "projection"),
    ],
)
def test_revision_decisions_route_only_affected_downstream_phases(
    tmp_path,
    decision: str,
    return_phase: str,
) -> None:
    _valid_run(tmp_path)
    packet, _ = _prepare_report(tmp_path)
    targets = ["january"] if decision == "revise_search" else []
    result = apply_quality_review(
        run_dir=tmp_path,
        decision_file=_decision(
            tmp_path,
            packet,
            decision,
            target_window_ids=targets,
        ),
    )

    assert result["state"] == "revision_required"
    assert result["return_phase"] == return_phase
    assert result["invalidated_artifacts"][-2:] == [
        "quality_report.json",
        "quality_review.md",
    ]
    assert quality_publish_status(tmp_path)["can_publish"] is False


def test_revision_requires_changed_inputs_and_a_new_report(tmp_path) -> None:
    _valid_run(tmp_path)
    packet, _ = _prepare_report(tmp_path)
    apply_quality_review(
        run_dir=tmp_path,
        decision_file=_decision(
            tmp_path,
            packet,
            "revise_events",
        ),
    )

    with pytest.raises(ValueError, match="changed audit inputs and a new"):
        apply_quality_review(
            run_dir=tmp_path,
            decision_file=_decision(tmp_path, packet, "approve"),
        )
    with pytest.raises(ValueError, match="affected audit inputs to change"):
        _prepare_report(tmp_path)


def test_stop_and_content_failure_never_authorize_publication(tmp_path) -> None:
    _valid_run(tmp_path)
    packet, _ = _prepare_report(tmp_path)
    stopped = apply_quality_review(
        run_dir=tmp_path,
        decision_file=_decision(tmp_path, packet, "stop"),
    )
    assert stopped["state"] == "stopped"
    assert quality_publish_status(tmp_path)["blocked_reason"] == "user_stopped"

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    _valid_run(failed_dir)
    failed_packet, _ = _prepare_report(
        failed_dir,
        gate_status="fail",
        worth_continuing=True,
    )
    with pytest.raises(ValueError, match="failed content review"):
        apply_quality_review(
            run_dir=failed_dir,
            decision_file=_decision(failed_dir, failed_packet, "approve"),
        )

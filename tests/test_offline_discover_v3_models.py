from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from world2ts_core.offline_discover import models


NOW = datetime(2026, 7, 23)


def _strategy_payload() -> dict:
    return {
        "schema_version": "2.0",
        "dataset_id": "example",
        "greenfield": True,
        "dataset_understanding": "Hourly state-level electricity demand.",
        "event_policy": {
            "allow_weak_context": True,
            "allow_retrospective_context": True,
            "allow_forecasts": True,
            "exclude_target_only_milestones": True,
            "canonical_date_preference": [
                "actual_start",
                "effective_date",
                "announcement_date",
                "publication_date",
            ],
            "duplicate_policy": "merge_same_episode",
            "compound_record_policy": "split_distinct_episodes",
        },
        "search_policy": {
            "default_sources": ["tavily", "anysearch"],
            "top_k": 10,
            "initial_queries_per_outline": 3,
            "max_refinement_queries_per_window": 5,
            "time_window_period": "yearly",
            "stop_policy": "outline_coverage_and_marginal_yield",
            "holiday_calendars": [
                {
                    "country_code": "US",
                    "subdivision_code": "MA",
                    "timezone": "America/New_York",
                    "target_series_ids": ["series_1"],
                }
            ],
        },
        "search_intent_outline": [
            {
                "outline_id": "weather_extremes",
                "event_family": "extreme weather",
                "priority": "high",
                "why_relevant": "Weather can affect grid demand and operations.",
                "context_pathways": ["weather -> operating conditions -> demand context"],
                "target_dimensions": ["state"],
                "time_window_period": "yearly",
                "initial_query_roles": [
                    "broad_safety_net",
                    "official_or_entity_specific",
                    "mechanism_specific",
                ],
                "in_scope_examples": ["heatwave", "storm"],
                "out_of_scope_examples": ["unrelated overseas weather"],
                "completion_evidence": "Major episodes checked for each state and year.",
            }
        ],
        "projection_policy": {
            "allowed_dimensions": ["state"],
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


def test_v3_model_surface_has_no_detected_shift_models_or_fields():
    for removed_name in (
        "AnchorRecord",
        "WindowClusterRecord",
        "WindowAnchorContext",
        "AgentJudgmentRecord",
        "BigQueryEventSummaryRecord",
        "DatasetContext",
        "ExampleIndexEntry",
        "OfflineDiscoverRun",
        "RawCandidateRecord",
        "ResolvedMetadata",
        "SearchPlanRecord",
        "WindowContextRecord",
    ):
        assert not hasattr(models, removed_name)

    assert "anchor_count" not in models.LoadedExample.model_fields
    assert set(models.ExampleEventLink.model_fields) == {
        "event_id",
        "series_id",
        "projection_rule_id",
        "confidence",
    }

    state = models.RunState(
        dataset_id="example",
        run_id="run_1",
        state="prepared",
        updated_at=NOW,
    )
    assert state.offline_discover_run_schema == "3.3"
    with pytest.raises(ValidationError, match="offline_discover_run_schema"):
        models.RunState.model_validate(
            {
                **state.model_dump(mode="json"),
                "offline_discover_run_schema": "3.2",
            }
        )


def test_agent_authored_models_forbid_unknown_fields():
    payload = _strategy_payload()
    payload["base_run"] = "old-run"

    with pytest.raises(ValidationError, match="base_run"):
        models.CurationStrategy.model_validate(payload)

    with pytest.raises(ValidationError, match="canonical_event_dates"):
        models.AgentEventSelectionRecord(
            window_id="2025",
            decision="accept",
            selected_event_summary_ids=["summary_1"],
            selected_by="agent_active_review",
            canonical_event_dates={},
        )


def test_curation_strategy_is_greenfield_and_dimensions_are_declared():
    strategy = models.CurationStrategy.model_validate(_strategy_payload())
    assert strategy.greenfield is True

    old_schema = _strategy_payload()
    old_schema["schema_version"] = "1.0"
    with pytest.raises(ValidationError, match="schema_version"):
        models.CurationStrategy.model_validate(old_schema)

    not_greenfield = _strategy_payload()
    not_greenfield["greenfield"] = False
    with pytest.raises(ValidationError, match="greenfield"):
        models.CurationStrategy.model_validate(not_greenfield)

    unknown_dimension = _strategy_payload()
    unknown_dimension["search_intent_outline"][0]["target_dimensions"] = ["station_id"]
    with pytest.raises(ValidationError, match="not approved"):
        models.CurationStrategy.model_validate(unknown_dimension)


def test_agent_query_intent_requires_explicit_feedback_lineage():
    with pytest.raises(ValidationError, match="failed-query feedback"):
        models.AgentQueryIntentRecord(
            query_intent_id="q2",
            window_id="2025",
            stage="window_refinement",
            query_text="Boston storm 2025",
            search_start=NOW,
            search_end=NOW,
            context_hypothesis="A storm may disrupt operations.",
            relevance_pathway="storm -> disruption -> operating context",
            feedback_query_intent_ids=["q1"],
            refinement_reason="The broad query returned commentary only.",
            generated_by="agent_active_review",
        )

    with pytest.raises(ValidationError, match="agent_active_review"):
        models.AgentQueryIntentRecord(
            query_intent_id="q1",
            window_id="2025",
            stage="initial",
            outline_id="weather",
            query_text="Boston storm 2025",
            search_start=NOW,
            search_end=NOW,
            context_hypothesis="A storm may disrupt operations.",
            relevance_pathway="storm -> disruption -> operating context",
            generated_by="agent_fallback",
        )


def test_selection_canonicalization_and_projection_are_separate_contracts():
    selection = models.AgentEventSelectionRecord(
        window_id="2025",
        decision="accept",
        selected_event_summary_ids=["summary_1"],
        rejected_event_summary_ids=["summary_2"],
        rejected_rationales={"summary_2": "Wrong geography."},
        selected_by="agent_active_review",
    )
    event = models.CanonicalEventRecord(
        event_id="event_1",
        event_start=NOW,
        summary="A storm disrupted Boston transport operations on July 23, 2026.",
        event_family="extreme weather",
        context_role="actual",
        affected_scope_claim="Boston transport services",
        source_event_summary_ids=["summary_1"],
        canonical_date_basis="Official incident date.",
        merge_or_split_notes="One episode retained.",
        confidence=0.8,
    )
    rule = models.SemanticProjectionRule(
        rule_id="rule_1",
        event_id=event.event_id,
        scope="dimension_filter",
        filters=[
            models.SemanticProjectionFilter(
                dimension="city", operator="in", value=["Boston"]
            )
        ],
        basis="The source explicitly scopes the disruption to Boston.",
        confidence=0.8,
    )
    link = models.ExampleEventLink(
        event_id=event.event_id,
        series_id="series_1",
        projection_rule_id=rule.rule_id,
        confidence=0.8,
    )

    assert selection.selected_event_summary_ids == ["summary_1"]
    assert link.event_id == event.event_id


def test_readiness_and_time_window_validate_inventory_and_order():
    report = models.DataReadinessReport(
        dataset_id="example",
        status="ready",
        arrow_files=["data.arrow"],
        series_count=1,
        series_ids=["series_1"],
        series_id_order_sha256="a" * 64,
        series_profiles=[
            models.SeriesReadinessProfile(
                series_id="series_1",
                start=NOW,
                end=NOW,
                freq="1h",
                length=1,
                missing_rate=0.0,
                has_explicit_timestamps=False,
            )
        ],
        dataset_semantics_coverage=1.0,
        series_semantics_coverage=1.0,
    )
    window = models.TimeWindowRecord(
        window_id="2026",
        period="yearly",
        start=NOW,
        end=NOW,
        active_series_ids=["series_1"],
        active_dimension_values={"city": ["Boston", "Boston"]},
        outline_ids=["weather"],
        priority=1.0,
    )

    assert report.status == "ready"
    assert window.active_dimension_values == {"city": ["Boston"]}


def test_review_decisions_require_content_hashes_and_revision_targets():
    strategy_decision = models.StrategyReviewDecision(
        decision="approve",
        reviewed_strategy_sha256="A" * 64,
        reviewed_at=NOW,
    )
    assert strategy_decision.reviewed_strategy_sha256 == "a" * 64

    with pytest.raises(ValidationError, match="target_window_ids"):
        models.QualityReviewDecision(
            decision="revise_search",
            reviewed_quality_report_sha256="b" * 64,
            reviewed_at=NOW,
        )

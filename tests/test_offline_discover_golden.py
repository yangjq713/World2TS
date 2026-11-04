from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

from world2ts_core.offline_discover.holiday_calendar import country_holiday_dates
from world2ts_core.offline_discover.canonicalization import validate_canonical_events
from world2ts_core.offline_discover.holiday_impact import holiday_impact_id
from world2ts_core.offline_discover.models import QualityReport
from world2ts_core.offline_discover.projection import project_events
from world2ts_core.offline_discover.publication import publish_minimal
from world2ts_core.offline_discover.quality_audit import (
    CONTENT_GATE_IDS,
    apply_quality_review,
    build_quality_audit_briefing,
    prepare_quality_review,
)
from world2ts_core.offline_discover.strategy import (
    apply_strategy_review,
    prepare_strategy_review,
)


FIXTURE = Path(__file__).parent / "fixtures" / "offline_discover_golden.json"
START = datetime(2013, 1, 1)
END = datetime(2018, 1, 1)


def _write_json(path: Path, value) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _strategy(case: dict) -> dict:
    return {
        "schema_version": "2.0",
        "dataset_id": case["dataset_id"],
        "greenfield": True,
        "dataset_understanding": case["description"],
        "event_policy": {
            "allow_weak_context": True,
            "allow_retrospective_context": False,
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
            "time_window_period": "yearly",
            "stop_policy": "outline coverage",
            "holiday_calendars": [
                {
                    "country_code": "US",
                    "timezone": "UTC",
                    "target_series_ids": [
                        series["series_id"] for series in case["series"]
                    ],
                }
            ],
        },
        "search_intent_outline": [
            {
                "outline_id": case["outline_id"],
                "event_family": case["event_family"],
                "priority": "high",
                "why_relevant": "The event family provides external operating context.",
                "context_pathways": ["external event -> represented series"],
                "target_dimensions": case["allowed_dimensions"],
                "time_window_period": "yearly",
                "initial_query_roles": ["broad", "official"],
                "in_scope_examples": [case["event_family"]],
                "out_of_scope_examples": ["target-only milestone"],
                "completion_evidence": "The frozen candidate set was reviewed.",
            }
        ],
        "projection_policy": {
            "allowed_dimensions": case["allowed_dimensions"],
            "allow_dataset_wide": case["allow_dataset_wide"],
            "ambiguous_scope_action": "hold_for_agent_review",
            "link_time_policy": "event_must_overlap_series_coverage",
        },
        "quality_policy": {
            "zero_event_series_allowed": True,
            "holiday_share_is_diagnostic_only": True,
            "require_user_final_approval": True,
        },
    }


def _write_readiness(run_dir: Path, case: dict) -> None:
    series_ids = [series["series_id"] for series in case["series"]]
    dimensions = []
    for dimension in case["allowed_dimensions"]:
        values = [
            series["dimensions"][dimension]
            for series in case["series"]
            if dimension in series["dimensions"]
        ]
        counts = Counter(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            for value in values
        )
        dimensions.append(
            {
                "dimension": dimension,
                "value_type": "string",
                "populated_series_count": len(values),
                "coverage_ratio": len(values) / len(series_ids),
                "value_counts": dict(sorted(counts.items())),
            }
        )
    _write_json(
        run_dir / "data_readiness_report.json",
        {
            "dataset_id": case["dataset_id"],
            "status": "ready",
            "arrow_files": ["frozen.arrow"],
            "series_count": len(series_ids),
            "series_ids": series_ids,
            "series_id_order_sha256": hashlib.sha256(
                "\n".join(series_ids).encode()
            ).hexdigest(),
            "series_profiles": [
                {
                    "series_id": series_id,
                    "start": START.isoformat(),
                    "end": END.isoformat(),
                    "freq": case["freq"],
                    "length": 43825,
                    "missing_rate": 0.0,
                    "has_explicit_timestamps": False,
                }
                for series_id in series_ids
            ],
            "dataset_semantics_coverage": 1.0,
            "series_semantics_coverage": 1.0,
            "dimensions": dimensions,
        },
    )
    _write_json(
        run_dir / "semantic_readiness_report.json",
        {
            "dataset_id": case["dataset_id"],
            "status": "ready",
            "expected_series_ids": series_ids,
            "validated_series_ids": series_ids,
        },
    )
    _write_json(
        run_dir / "agent_dataset_semantics.json",
        {
            "dataset_id": case["dataset_id"],
            "description": case["description"],
            "domain": case["domain"],
            "frequency_phrases": [case["frequency_phrase"]],
        },
    )
    _write_jsonl(
        run_dir / "agent_series_semantics.jsonl",
        [
            {
                "series_id": series["series_id"],
                "description": (
                    f"This {case['frequency_phrase']} series represents "
                    f"{series['series_id']} with verified semantic dimensions "
                    f"for {case['dataset_id']}."
                ),
                "time_range": {
                    "start": START.isoformat(),
                    "end": END.isoformat(),
                },
                "frequency_phrase": case["frequency_phrase"],
            }
            for series in case["series"]
        ],
    )
    _write_jsonl(
        run_dir / "series_semantic_dimensions.jsonl",
        [
            {
                "series_id": series["series_id"],
                "confidence": 1.0,
                "dimensions": series["dimensions"],
            }
            for series in case["series"]
        ],
    )


def _approve_strategy(run_dir: Path, case: dict) -> dict:
    strategy = _strategy(case)
    _write_json(run_dir / "curation_strategy.draft.json", strategy)
    packet = prepare_strategy_review(
        strategy_file=run_dir / "curation_strategy.draft.json",
        data_readiness_report_file=run_dir / "data_readiness_report.json",
        semantic_readiness_report_file=run_dir / "semantic_readiness_report.json",
        output_dir=run_dir,
    )
    assert packet["blocked_reason"] == "user_strategy_review_required"
    decision = run_dir / "strategy_review_decision.json"
    _write_json(
        decision,
        {
            "decision": "approve",
            "reviewed_strategy_sha256": packet["strategy_sha256"],
            "reviewed_at": START.isoformat(),
        },
    )
    assert apply_strategy_review(run_dir=run_dir, decision_file=decision)[
        "state"
    ] == "strategy_approved"
    return strategy


def _write_frozen_agent_artifacts(run_dir: Path, case: dict) -> None:
    query_id = f"{case['case_id']}_query"
    source_ids = [
        candidate["event_summary_id"] for candidate in case["candidates"]
    ]
    series_ids = [series["series_id"] for series in case["series"]]
    _write_json(
        run_dir / "time_windows.json",
        [
            {
                "window_id": "golden_window",
                "period": "yearly",
                "start": START.isoformat(),
                "end": END.isoformat(),
                "active_series_ids": series_ids,
                "outline_ids": [case["outline_id"]],
                "priority": float(len(series_ids)),
            }
        ],
    )
    _write_jsonl(
        run_dir / "agent_window_search_plans.jsonl",
        [
            {
                "window_id": "golden_window",
                "prior_event_hypotheses": [
                    {
                        "hypothesis_id": f"{case['case_id']}_prior",
                        "event_name": case["event_family"],
                        "date_hint": "Frozen golden window",
                        "relevance_pathway": "external event -> represented series",
                        "reasoning": "The frozen case supplies a known relevant event family.",
                        "confidence": 0.8,
                    }
                ],
                "generated_by": "agent_active_review",
            }
        ],
    )
    _write_jsonl(
        run_dir / "agent_query_intents.jsonl",
        [
            {
                "query_intent_id": query_id,
                "window_id": "golden_window",
                "stage": "initial",
                "outline_id": case["outline_id"],
                "query_text": case["case_id"].replace("_", " "),
                "search_start": START.isoformat(),
                "search_end": END.isoformat(),
                "top_k": 10,
                "context_hypothesis": "A frozen candidate may add context.",
                "relevance_pathway": "external event -> represented series",
                "generated_by": "agent_active_review",
            }
        ],
    )
    _write_json(
        run_dir / "search_state.json",
        [
            {
                "window_id": "golden_window",
                "status": "covered",
                "attempted_initial_query_intent_ids": [query_id],
                "attempted_refinement_query_intent_ids": [],
                "normalized_queries": [case["case_id"].replace("_", " ")],
                "candidate_event_summary_ids": source_ids,
                "holiday_analysis_complete": True,
            }
        ],
    )
    _write_jsonl(
        run_dir / "holiday_impacts.jsonl",
        [
            {
                "impact_id": holiday_impact_id(
                    "golden_window",
                    "US",
                    "",
                    "UTC",
                    series_ids,
                    holiday_date,
                    holiday_name,
                ),
                "window_id": "golden_window",
                "country_code": "US",
                "subdivision_code": "",
                "timezone": "UTC",
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
                    for series_id in series_ids
                ],
            }
            for holiday_date, holiday_name in country_holiday_dates(
                "US",
                "",
                START.date(),
                END.date(),
            )
        ],
    )
    candidates = [
        {
            **candidate,
            "window_id": "golden_window",
            "query_intent_id": query_id,
        }
        for candidate in case["candidates"]
    ]
    _write_jsonl(run_dir / "discovered_events.jsonl", candidates)
    _write_jsonl(
        run_dir / "search_trace.jsonl",
        [
            {
                "trace_id": f"{case['case_id']}_{source}",
                "window_id": "golden_window",
                "query_intent_id": query_id,
                "stage": "initial",
                "outline_id": case["outline_id"],
                "query_text": case["case_id"].replace("_", " "),
                "source": source,
                "started_at": START.isoformat(),
                "elapsed_ms": 1,
                "returned_count": sum(
                    candidate["source_name"] == source
                    for candidate in case["candidates"]
                ),
                "event_summary_ids": [
                    candidate["event_summary_id"]
                    for candidate in case["candidates"]
                    if candidate["source_name"] == source
                ],
                "selection_decision": "accept",
            }
            for source in ("tavily", "anysearch")
        ],
    )
    _write_jsonl(
        run_dir / "agent_event_selections.jsonl",
        [
            {
                "window_id": "golden_window",
                "decision": "accept",
                **case["selection"],
                "confidence": 0.95,
                "selected_by": "agent_active_review",
            }
        ],
    )
    _write_jsonl(run_dir / "canonical_events.jsonl", case["canonical_events"])
    _write_jsonl(run_dir / "projection_rules.jsonl", case["projection_rules"])


def _approve_quality(run_dir: Path, case: dict) -> dict:
    briefing = build_quality_audit_briefing(run_dir)
    assert briefing["state"] == "auditing", briefing
    assert all(gate["status"] == "pass" for gate in briefing["hard_gates"])
    report = QualityReport(
        dataset_id=case["dataset_id"],
        generated_at=END,
        audit_input_sha256=briefing["audit_input_sha256"],
        overall_status="pass",
        hard_gates=briefing["hard_gates"],
        content_gates=[
            {
                "gate_id": gate_id,
                "gate_type": "content",
                "status": "pass",
                "evidence": ["The frozen golden case satisfies this content gate."],
                "impact": "No revision is required.",
                "continue_search_recommended": False,
            }
            for gate_id in CONTENT_GATE_IDS
        ],
        statistics=briefing["statistics"],
        worth_continuing_search=False,
        recommendation="Approve this frozen golden case.",
    )
    report_file = run_dir / "agent_quality_report.json"
    _write_json(report_file, report)
    packet = prepare_quality_review(
        run_dir=run_dir,
        quality_report_file=report_file,
    )
    assert packet["blocked_reason"] == "user_quality_review_required"
    with pytest.raises(ValueError, match="user_quality_review_required"):
        publish_minimal(run_dir)
    decision = run_dir / "quality_review_decision.json"
    _write_json(
        decision,
        {
            "decision": "approve",
            "reviewed_quality_report_sha256": packet["quality_report_sha256"],
            "reviewed_at": END.isoformat(),
        },
    )
    assert apply_quality_review(run_dir=run_dir, decision_file=decision)[
        "state"
    ] == "approved"
    return briefing


def test_four_frozen_golden_cases_complete_both_review_gates(tmp_path) -> None:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    matched_events = 0
    canonical_events = 0
    correct_dates = 0
    duplicate_episodes = 0
    illegal_links = 0
    full_broadcasts = 0
    authorized_full_broadcasts = 0
    self_contained = 0
    summaries = 0
    covered_outlines = 0
    outline_count = 0

    for case in cases:
        run_dir = tmp_path / case["case_id"]
        run_dir.mkdir()
        _write_readiness(run_dir, case)
        strategy = _approve_strategy(run_dir, case)
        _write_frozen_agent_artifacts(run_dir, case)

        canonical = validate_canonical_events(run_dir)
        duplicate_episodes += len(canonical["duplicate_hints"])
        projection = project_events(run_dir)
        actual_events = {
            event["event_id"]: event
            for event in case["canonical_events"]
        }
        expected_dates = case["expected_event_dates"]
        canonical_events += len(actual_events)
        matched_events += len(set(actual_events) & set(expected_dates))
        correct_dates += sum(
            actual_events[event_id]["event_start"] == expected_date
            for event_id, expected_date in expected_dates.items()
            if event_id in actual_events
        )

        actual_links = {
            (row["event_id"], row["series_id"])
            for row in (
                json.loads(line)
                for line in (run_dir / "projected_event_series_links.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        expected_links = {tuple(link) for link in case["expected_links"]}
        illegal_links += len(actual_links - expected_links)
        assert actual_links == expected_links
        assert projection["link_count"] == len(expected_links)

        links_per_event = Counter(event_id for event_id, _ in actual_links)
        rules = {
            rule["event_id"]: rule for rule in case["projection_rules"]
        }
        for event_id, count in links_per_event.items():
            if count != len(case["series"]):
                continue
            full_broadcasts += 1
            authorized_full_broadcasts += (
                strategy["projection_policy"]["allow_dataset_wide"]
                and rules[event_id]["scope"] == "dataset"
            )

        for event_id, terms in case["summary_terms"].items():
            summary = actual_events[event_id]["summary"].casefold()
            summaries += 1
            self_contained += all(term.casefold() in summary for term in terms)

        briefing = _approve_quality(run_dir, case)
        covered_outlines += sum(
            statuses.get("queries", 0) > 0
            and statuses.get("failures", 0) == 0
            for statuses in briefing["statistics"]["outline_coverage"].values()
        )
        outline_count += len(briefing["statistics"]["outline_coverage"])
        published = publish_minimal(run_dir)
        assert published["state"] == "published"
        assert published["num_events"] == len(actual_events)

    metrics = {
        "case_count": len(cases),
        "canonical_event_precision": matched_events / canonical_events,
        "canonical_date_accuracy": correct_dates / len(
            [
                expected
                for case in cases
                for expected in case["expected_event_dates"]
            ]
        ),
        "duplicate_episode_count": duplicate_episodes,
        "illegal_cross_dimension_link_count": illegal_links,
        "full_broadcast_authorization_rate": (
            authorized_full_broadcasts / full_broadcasts
        ),
        "summary_self_containment_rate": self_contained / summaries,
        "outline_coverage": covered_outlines / outline_count,
    }
    _write_json(tmp_path / "golden_quality_metrics.json", metrics)

    assert metrics == {
        "case_count": 4,
        "canonical_event_precision": 1.0,
        "canonical_date_accuracy": 1.0,
        "duplicate_episode_count": 0,
        "illegal_cross_dimension_link_count": 0,
        "full_broadcast_authorization_rate": 1.0,
        "summary_self_containment_rate": 1.0,
        "outline_coverage": 1.0,
    }

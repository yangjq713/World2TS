from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from world2ts_core.offline_discover.models import (
    DataReadinessReport,
    DimensionReadinessProfile,
    SemanticReadinessReport,
    SeriesReadinessProfile,
)
from world2ts_core.offline_discover.strategy import (
    APPROVED_FILENAME,
    DRAFT_FILENAME,
    USER_REVIEW_REQUIRED,
    USER_STOPPED,
    apply_strategy_review,
    prepare_strategy_review,
    strategy_search_status,
)


NOW = datetime(2026, 7, 23)


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _strategy() -> dict:
    return {
        "schema_version": "2.0",
        "dataset_id": "example",
        "greenfield": True,
        "dataset_understanding": "Hourly state-level electricity demand with explicit regional semantics.",
        "event_policy": {
            "allow_weak_context": True,
            "allow_retrospective_context": True,
            "allow_forecasts": False,
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
                    "country_code": "AU",
                    "subdivision_code": "NSW",
                    "timezone": "Australia/Sydney",
                    "target_series_ids": ["series_1"],
                }
            ],
        },
        "search_intent_outline": [
            {
                "outline_id": "weather_extremes",
                "event_family": "extreme weather",
                "priority": "high",
                "why_relevant": "Weather can affect grid demand and operating context.",
                "context_pathways": ["weather -> operations -> demand context"],
                "target_dimensions": ["state"],
                "time_window_period": "yearly",
                "initial_query_roles": ["broad", "official", "mechanism"],
                "in_scope_examples": ["heatwave", "storm"],
                "out_of_scope_examples": ["unrelated overseas weather"],
                "completion_evidence": "Major episodes checked for every active state and year.",
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


def _readiness_files(tmp_path: Path) -> tuple[Path, Path]:
    data = DataReadinessReport(
        dataset_id="example",
        status="ready",
        arrow_files=["data.arrow"],
        series_count=1,
        series_ids=["series_1"],
        series_id_order_sha256="a" * 64,
        series_profiles=[
            SeriesReadinessProfile(
                series_id="series_1",
                start=NOW,
                end=NOW,
                freq="H",
                length=1,
                missing_rate=0.0,
                has_explicit_timestamps=False,
            )
        ],
        dataset_semantics_coverage=1.0,
        series_semantics_coverage=1.0,
        dimensions=[
            DimensionReadinessProfile(
                dimension="state",
                value_type="string",
                populated_series_count=1,
                coverage_ratio=1.0,
                value_counts={'"NSW"': 1},
            )
        ],
    )
    semantic = SemanticReadinessReport(
        dataset_id="example",
        status="ready",
        expected_series_ids=["series_1"],
        validated_series_ids=["series_1"],
    )
    return (
        _write_json(tmp_path / "data_readiness_report.json", data.model_dump(mode="json")),
        _write_json(
            tmp_path / "semantic_readiness_report.json", semantic.model_dump(mode="json")
        ),
    )


def _prepare(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    data_file, semantic_file = _readiness_files(tmp_path)
    strategy_file = _write_json(tmp_path / "agent_strategy.json", _strategy())
    run_dir = tmp_path / "run"
    packet = prepare_strategy_review(
        strategy_file=strategy_file,
        data_readiness_report_file=data_file,
        semantic_readiness_report_file=semantic_file,
        output_dir=run_dir,
    )
    return run_dir, packet


def test_prepare_strategy_review_validates_and_writes_concise_packet(tmp_path: Path) -> None:
    run_dir, packet = _prepare(tmp_path)

    assert packet["state"] == "awaiting_strategy_review"
    assert packet["blocked_reason"] == USER_REVIEW_REQUIRED
    assert len(packet["strategy_sha256"]) == 64
    review = (run_dir / "strategy_review.md").read_text(encoding="utf-8")
    assert "extreme weather" in review
    assert "3 initial per outline, 5 window-level refinements" in review
    assert "AU/NSW" in review
    assert "Australia/Sydney" in review
    assert "series_1" in review
    assert packet["strategy_sha256"] in review
    assert '"search_intent_outline"' not in review


def test_strategy_rejects_dimensions_absent_from_readiness(tmp_path: Path) -> None:
    data_file, semantic_file = _readiness_files(tmp_path)
    strategy = _strategy()
    strategy["projection_policy"]["allowed_dimensions"] = ["station_id"]
    strategy["search_intent_outline"][0]["target_dimensions"] = ["station_id"]

    with pytest.raises(ValueError, match="dimensions absent from readiness"):
        prepare_strategy_review(
            strategy_file=_write_json(tmp_path / "strategy.json", strategy),
            data_readiness_report_file=data_file,
            semantic_readiness_report_file=semantic_file,
            output_dir=tmp_path / "run",
        )


def test_strategy_rejects_unknown_holiday_series_scope(tmp_path: Path) -> None:
    data_file, semantic_file = _readiness_files(tmp_path)
    strategy = _strategy()
    strategy["search_policy"]["holiday_calendars"][0][
        "target_series_ids"
    ] = ["missing_series"]

    with pytest.raises(ValueError, match="series absent from readiness"):
        prepare_strategy_review(
            strategy_file=_write_json(tmp_path / "strategy.json", strategy),
            data_readiness_report_file=data_file,
            semantic_readiness_report_file=semantic_file,
            output_dir=tmp_path / "run",
        )


def test_strategy_rejects_unknown_holiday_timezone(tmp_path: Path) -> None:
    data_file, semantic_file = _readiness_files(tmp_path)
    strategy = _strategy()
    strategy["search_policy"]["holiday_calendars"][0]["timezone"] = (
        "Mars/Olympus_Mons"
    )

    with pytest.raises(ValueError, match="unknown IANA timezone"):
        prepare_strategy_review(
            strategy_file=_write_json(tmp_path / "strategy.json", strategy),
            data_readiness_report_file=data_file,
            semantic_readiness_report_file=semantic_file,
            output_dir=tmp_path / "run",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda strategy: strategy["search_policy"].update(
                default_sources=["unapproved_source"]
            ),
            "default_sources",
        ),
        (
            lambda strategy: strategy["search_policy"].update(
                initial_queries_per_outline=0,
            ),
            "initial_queries_per_outline",
        ),
        (lambda strategy: strategy["search_policy"].update(time_window_period="decade"), "time_window_period"),
    ],
)
def test_strategy_rejects_invalid_sources_budgets_and_periods(
    tmp_path: Path, mutate, message: str
) -> None:
    data_file, semantic_file = _readiness_files(tmp_path)
    strategy = _strategy()
    mutate(strategy)

    with pytest.raises(ValueError, match=message):
        prepare_strategy_review(
            strategy_file=_write_json(tmp_path / "strategy.json", strategy),
            data_readiness_report_file=data_file,
            semantic_readiness_report_file=semantic_file,
            output_dir=tmp_path / "run",
        )


@pytest.mark.parametrize(
    ("decision", "state", "blocked_reason", "approved"),
    [
        ("approve", "strategy_approved", "", True),
        ("revise", "strategy_revision_required", USER_REVIEW_REQUIRED, False),
        ("stop", "stopped", USER_STOPPED, False),
    ],
)
def test_strategy_review_decisions_control_search_gate(
    tmp_path: Path,
    decision: str,
    state: str,
    blocked_reason: str,
    approved: bool,
) -> None:
    run_dir, packet = _prepare(tmp_path)
    decision_file = _write_json(
        tmp_path / "decision.json",
        {
            "decision": decision,
            "reviewed_strategy_sha256": packet["strategy_sha256"],
            "user_instructions": "",
            "reviewed_at": NOW.isoformat(),
        },
    )

    result = apply_strategy_review(run_dir=run_dir, decision_file=decision_file)
    gate = strategy_search_status(run_dir)

    assert result["state"] == state
    assert result["blocked_reason"] == blocked_reason
    assert gate["approved"] is approved
    assert gate["blocked_reason"] == blocked_reason
    assert (run_dir / APPROVED_FILENAME).exists() is approved


def test_strategy_change_invalidates_old_approval(tmp_path: Path) -> None:
    run_dir, packet = _prepare(tmp_path)
    decision_file = _write_json(
        tmp_path / "decision.json",
        {
            "decision": "approve",
            "reviewed_strategy_sha256": packet["strategy_sha256"],
            "user_instructions": "",
            "reviewed_at": NOW.isoformat(),
        },
    )
    apply_strategy_review(run_dir=run_dir, decision_file=decision_file)

    changed = json.loads((run_dir / DRAFT_FILENAME).read_text(encoding="utf-8"))
    changed["dataset_understanding"] = "The Agent revised the dataset interpretation."
    _write_json(run_dir / DRAFT_FILENAME, changed)

    gate = strategy_search_status(run_dir)
    assert gate["approved"] is False
    assert gate["blocked_reason"] == USER_REVIEW_REQUIRED
    with pytest.raises(ValueError, match="does not match the current draft"):
        apply_strategy_review(run_dir=run_dir, decision_file=decision_file)

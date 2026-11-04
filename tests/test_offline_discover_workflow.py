from __future__ import annotations

import json
from datetime import datetime

import pytest

from world2ts_core.offline_discover import workflow
from world2ts_core.offline_discover.models import (
    AgentEventSelectionRecord,
    RunState,
    SearchWindowStateRecord,
)
from world2ts_core.offline_discover.search import _apply_selections
from world2ts_core.offline_discover.workflow import run_phase
from tests.test_offline_discover_readiness import (
    _metadata,
    _rows,
    _write_arrow,
    _write_json,
)
from tests.test_offline_discover_v3_models import _strategy_payload


def _prepared_run(tmp_path):
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(["A"]))
    metadata = _write_json(tmp_path / "metadata.json", _metadata(["A"]))
    run_dir = tmp_path / "run"
    result = run_phase(
        phase="prepare",
        run_dir=run_dir,
        dataset_id="readiness_fixture",
        arrow_files=[arrow],
        metadata_file=metadata,
    )
    return run_dir, result


def _write_agent_semantics(run_dir):
    _write_json(
        run_dir / "agent_dataset_semantics.json",
        {
            "dataset_id": "readiness_fixture",
            "description": "This hourly dataset records electricity demand.",
            "domain": "energy",
            "frequency_phrases": ["hourly"],
        },
    )
    (run_dir / "agent_series_semantics.jsonl").write_text(
        json.dumps(
            {
                "series_id": "A",
                "description": "This hourly series records electricity demand for region A.",
                "time_range": {
                    "start": "2020-01-01T00:00:00",
                    "end": "2020-01-01T02:00:00",
                },
                "frequency_phrase": "hourly",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_state(run_dir, state, next_agent_action=""):
    run_dir.mkdir(exist_ok=True)
    _write_json(
        run_dir / "run_state.json",
        RunState(
            dataset_id="readiness_fixture",
            run_id=run_dir.name,
            state=state,
            updated_at=datetime(2026, 7, 24),
            next_agent_action=next_agent_action,
        ).model_dump(mode="json"),
    )


def test_prepare_and_semantics_use_explicit_state_and_history(tmp_path) -> None:
    run_dir, prepared = _prepared_run(tmp_path)

    assert prepared["state"] == "prepared"
    assert prepared["blocked_reason"] == "agent_semantics_required"
    assert prepared["next_agent_action"] == "write_agent_semantics"
    assert set(prepared["artifact_paths"]) == {
        "data_readiness_report.json",
        "series_semantic_dimensions.jsonl",
        "agent_semantic_briefing.json",
    }

    _write_agent_semantics(run_dir)
    semantic = run_phase(phase="validate-semantics", run_dir=run_dir)
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    history = (run_dir / "run_history.jsonl").read_text(encoding="utf-8").splitlines()

    assert semantic["next_agent_action"] == "write_curation_strategy"
    assert state["state"] == "prepared"
    assert state["blocked_reason"] == "agent_strategy_required"
    assert len(history) == 2
    assert json.loads(history[-1])["previous_state"] == "prepared"


def test_strategy_review_approval_builds_windows_and_intent_briefing(tmp_path) -> None:
    run_dir, _ = _prepared_run(tmp_path)
    _write_agent_semantics(run_dir)
    run_phase(phase="validate-semantics", run_dir=run_dir)

    strategy = _strategy_payload()
    strategy["dataset_id"] = "readiness_fixture"
    strategy["search_intent_outline"][0]["target_dimensions"] = ["region"]
    strategy["projection_policy"]["allowed_dimensions"] = ["region"]
    strategy["search_policy"]["holiday_calendars"][0].update(
        {
            "timezone": "America/New_York",
            "target_series_ids": ["A"],
        }
    )
    _write_json(run_dir / "curation_strategy.draft.json", strategy)

    review = run_phase(phase="validate-strategy", run_dir=run_dir)
    assert review["state"] == "awaiting_strategy_review"
    assert review["blocked_reason"] == "user_strategy_review_required"

    _write_json(
        run_dir / "strategy_review_decision.json",
        {
            "decision": "approve",
            "reviewed_strategy_sha256": review["strategy_sha256"],
            "reviewed_at": datetime(2026, 7, 24).isoformat(),
        },
    )
    approved = run_phase(phase="validate-strategy", run_dir=run_dir)
    briefing = run_phase(phase="intents", run_dir=run_dir)

    assert approved["state"] == "strategy_approved"
    assert approved["time_window_count"] == 1
    assert (run_dir / "time_windows.json").is_file()
    assert briefing["state"] == "searching"
    assert briefing["blocked_reason"] == "agent_search_inputs_required"
    assert briefing["search_window_count"] == 1


def test_prepare_rejects_nonempty_run_directory(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "old.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Greenfield prepare"):
        run_phase(
            phase="prepare",
            run_dir=run_dir,
            dataset_id="readiness_fixture",
            arrow_files=[tmp_path / "data.arrow"],
            metadata_file=tmp_path / "metadata.json",
        )


def test_prepare_records_failed_readiness_before_raising(tmp_path) -> None:
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(["A"]))
    metadata = _write_json(tmp_path / "metadata.json", _metadata(["B"]))
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="Arrow example IDs do not match"):
        run_phase(
            phase="prepare",
            run_dir=run_dir,
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata,
        )

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    history = [
        json.loads(line)
        for line in (run_dir / "run_history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert state["state"] == "failed"
    assert state["blocked_reason"] == "data_readiness_failed"
    assert state["next_agent_action"] == "fix_inputs"
    assert history[-1]["next_state"] == "failed"


def test_agent_loop_terminal_state_routes_to_canonicalization(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_state(run_dir, "searching")
    monkeypatch.setattr(
        workflow,
        "run_search_iteration",
        lambda *args, **kwargs: {
            "blocked_reason": "",
            "next_agent_action": "",
            "search_windows": [{"status": "covered"}],
        },
    )

    result = run_phase(phase="agent-loop", run_dir=run_dir)

    assert result["state"] == "curating"
    assert result["blocked_reason"] == "agent_canonical_events_required"
    assert result["next_agent_action"] == "write_canonical_events"
    assert "search_trace.jsonl" in result["artifact_paths"]


def test_post_search_phases_preserve_machine_readable_contract(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_state(run_dir, "curating")
    monkeypatch.setattr(
        workflow,
        "validate_canonical_events",
        lambda run_dir: {"status": "valid", "canonical_event_count": 1},
    )
    canonical = run_phase(phase="validate-canonical-events", run_dir=run_dir)

    monkeypatch.setattr(
        workflow,
        "project_events",
        lambda run_dir: {"status": "projected", "link_count": 1},
    )
    projected = run_phase(phase="project", run_dir=run_dir)

    monkeypatch.setattr(
        workflow,
        "build_quality_audit_briefing",
        lambda run_dir: {
            "state": "auditing",
            "audit_input_sha256": "a" * 64,
            "hard_gates": [{"status": "pass"}],
        },
    )
    briefing = run_phase(phase="audit", run_dir=run_dir)
    monkeypatch.setattr(
        workflow,
        "prepare_quality_review",
        lambda **kwargs: {
            "state": "awaiting_quality_review",
            "blocked_reason": "user_quality_review_required",
            "next_agent_action": "request_quality_review",
            "quality_report_sha256": "b" * 64,
        },
    )
    review = run_phase(phase="audit", run_dir=run_dir)
    monkeypatch.setattr(
        workflow,
        "apply_quality_review",
        lambda **kwargs: {
            "state": "approved",
            "blocked_reason": "",
            "next_agent_action": "publish",
            "quality_report_sha256": "b" * 64,
        },
    )
    approved = run_phase(phase="audit", run_dir=run_dir)
    monkeypatch.setattr(
        workflow,
        "publish_minimal",
        lambda run_dir: {
            "state": "published",
            "run_dir": str(run_dir),
            "num_series": 1,
        },
    )
    published = run_phase(phase="publish", run_dir=run_dir)

    for result in (
        canonical,
        projected,
        briefing,
        review,
        approved,
        published,
    ):
        assert {
            "state",
            "blocked_reason",
            "next_agent_action",
            "artifact_paths",
        } <= set(result)
    assert canonical["next_agent_action"] == "write_projection_rules"
    assert projected["next_agent_action"] == "run_quality_audit"
    assert briefing["next_agent_action"] == "write_quality_report"
    assert review["blocked_reason"] == "user_quality_review_required"
    assert approved["next_agent_action"] == "publish"
    assert published["next_agent_action"] == "complete"


def test_workflow_does_not_call_publisher_before_quality_approval(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_state(run_dir, "awaiting_quality_review")
    calls = 0

    def publish(run_dir):
        nonlocal calls
        calls += 1
        return {"state": "published"}

    monkeypatch.setattr(workflow, "publish_minimal", publish)

    with pytest.raises(ValueError, match="expected one of approved"):
        run_phase(phase="publish", run_dir=run_dir)

    assert calls == 0
    assert json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))[
        "state"
    ] == "awaiting_quality_review"


def test_domain_failure_is_recorded_and_retries_only_the_same_phase(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_state(run_dir, "curating")
    monkeypatch.setattr(
        workflow,
        "validate_canonical_events",
        lambda run_dir: (_ for _ in ()).throw(
            ValueError("canonical artifact is invalid")
        ),
    )

    with pytest.raises(ValueError, match="canonical artifact is invalid"):
        run_phase(phase="validate-canonical-events", run_dir=run_dir)

    failed = json.loads(
        (run_dir / "run_state.json").read_text(encoding="utf-8")
    )
    assert failed["state"] == "failed"
    assert failed["blocked_reason"] == "validate_canonical_events_failed"
    assert failed["next_agent_action"] == "retry_validate_canonical_events"
    with pytest.raises(ValueError, match="expected one of projecting"):
        run_phase(phase="project", run_dir=run_dir)

    monkeypatch.setattr(
        workflow,
        "validate_canonical_events",
        lambda run_dir: {"status": "valid", "canonical_event_count": 1},
    )
    retried = run_phase(
        phase="validate-canonical-events",
        run_dir=run_dir,
    )
    assert retried["state"] == "projecting"


def test_quality_search_revision_preserves_selections_and_reopens_only_targets(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_state(run_dir, "awaiting_quality_review")
    selected_ids = {
        "window_1": "selected_before_revision_1",
        "window_2": "selected_before_revision_2",
    }
    states = [
        SearchWindowStateRecord(
            window_id=window_id,
            status="covered",
            candidate_event_summary_ids=[selected_ids[window_id]],
            reviewed_candidate_event_summary_ids=[selected_ids[window_id]],
        )
        for window_id in ("window_1", "window_2")
    ]
    _write_json(
        run_dir / "search_state.json",
        [state.model_dump(mode="json") for state in states],
    )
    selections = [
        AgentEventSelectionRecord(
            window_id=window_id,
            decision="complete",
            selected_event_summary_ids=[selected_ids[window_id]],
            selected_by="agent_active_review",
        )
        for window_id in ("window_1", "window_2")
    ]
    (run_dir / "agent_event_selections.jsonl").write_text(
        "".join(selection.model_dump_json() + "\n" for selection in selections),
        encoding="utf-8",
    )
    for filename in (
        "canonical_events.jsonl",
        "projection_rules.jsonl",
        "projected_event_series_links.jsonl",
        "quality_report.json",
        "quality_review.md",
        "search_trace.jsonl",
    ):
        (run_dir / filename).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "apply_quality_review",
        lambda **kwargs: {
            "state": "revision_required",
            "blocked_reason": "revise_search",
            "next_agent_action": "revise_search",
            "return_phase": "search",
            "target_window_ids": ["window_1"],
            "invalidated_artifacts": [],
            "quality_report_sha256": "b" * 64,
        },
    )

    result = run_phase(phase="audit", run_dir=run_dir)
    revised_states = json.loads(
        (run_dir / "search_state.json").read_text(encoding="utf-8")
    )
    remaining_selections = [
        json.loads(line)
        for line in (run_dir / "agent_event_selections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert result["state"] == "revision_required"
    assert revised_states[0]["status"] == "awaiting_refinement"
    assert revised_states[0]["next_agent_action"] == "write_window_refined_query"
    assert revised_states[1]["status"] == "covered"
    assert [item["window_id"] for item in remaining_selections] == [
        "window_1",
        "window_2",
    ]
    assert remaining_selections[0]["selected_event_summary_ids"] == [
        "selected_before_revision_1"
    ]
    assert (run_dir / "search_trace.jsonl").exists()
    assert not (run_dir / "canonical_events.jsonl").exists()

    reopened_states = [
        SearchWindowStateRecord.model_validate(item) for item in revised_states
    ]
    preserved_selections = [
        AgentEventSelectionRecord.model_validate(item)
        for item in remaining_selections
    ]
    _apply_selections(
        reopened_states,
        preserved_selections,
        [],
        max_refinements=2,
    )
    assert reopened_states[0].status == "awaiting_refinement"
    assert reopened_states[1].status == "covered"

    new_candidate_id = "selected_from_new_refinement"
    reopened_states[0].candidate_event_summary_ids.append(new_candidate_id)
    reopened_states[0].status = "awaiting_selection"
    new_selection = AgentEventSelectionRecord(
        window_id="window_1",
        decision="complete",
        selected_event_summary_ids=[new_candidate_id],
        selected_by="agent_active_review",
    )
    _apply_selections(
        reopened_states,
        [*preserved_selections, new_selection],
        [],
        max_refinements=2,
    )
    assert reopened_states[0].status == "covered"
    assert reopened_states[0].reviewed_candidate_event_summary_ids == [
        "selected_before_revision_1",
        "selected_from_new_refinement",
    ]

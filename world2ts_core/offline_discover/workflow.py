"""Explicit offline-discover phase orchestration and run-state tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .canonicalization import validate_canonical_events
from .models import (
    CurationStrategy,
    DataReadinessReport,
    RunState,
    SearchWindowStateRecord,
    SeriesSemanticDimensionsRecord,
    TimeWindowRecord,
)
from .projection import project_events
from .publication import publish_minimal
from .quality_audit import (
    QUALITY_DECISION_FILENAME,
    QUALITY_REPORT_FILENAME,
    apply_quality_review,
    build_quality_audit_briefing,
    prepare_quality_review,
)
from .query_intents import build_agent_query_briefing
from .readiness import prepare_data_readiness, validate_semantic_readiness
from .search import run_search_iteration
from .strategy import (
    APPROVED_FILENAME,
    DECISION_FILENAME as STRATEGY_DECISION_FILENAME,
    DRAFT_FILENAME as STRATEGY_DRAFT_FILENAME,
    apply_strategy_review,
    prepare_strategy_review,
)
from .time_windows import build_time_windows

PHASES = (
    "prepare",
    "validate-semantics",
    "validate-strategy",
    "intents",
    "agent-loop",
    "validate-canonical-events",
    "project",
    "audit",
    "publish",
)
RUN_STATE_FILENAME = "run_state.json"
RUN_HISTORY_FILENAME = "run_history.jsonl"
_SEARCH_TERMINAL = {"covered", "exhausted", "no_event_found"}


class _InvalidTransition(ValueError):
    pass


def run_phase(
    *,
    phase: str,
    run_dir: Path,
    dataset_id: str = "",
    arrow_files: Sequence[Path] = (),
    metadata_file: Path | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Execute one explicit phase using run-local Agent artifacts."""
    if phase not in PHASES:
        raise ValueError("--phase must be one of: " + ", ".join(PHASES))
    run_dir = run_dir.expanduser()
    if phase == "prepare":
        return _prepare(run_dir, dataset_id, arrow_files, metadata_file)

    current = _load_state(run_dir)
    handlers = {
        "validate-semantics": _validate_semantics,
        "validate-strategy": _validate_strategy,
        "intents": _intents,
        "validate-canonical-events": _validate_canonical,
        "project": _project,
        "audit": _audit,
        "publish": _publish,
    }
    try:
        if phase == "agent-loop":
            return _agent_loop(run_dir, current, retry_failed)
        return handlers[phase](run_dir, current)
    except _InvalidTransition:
        raise
    except (OSError, ValueError) as exc:
        _finish(
            run_dir,
            current,
            trigger=phase,
            state="failed",
            reason=str(exc),
            blocked_reason=phase.replace("-", "_") + "_failed",
            next_agent_action=_retry_action(phase),
            filenames=tuple(current.artifact_sha256),
        )
        raise


def _prepare(
    run_dir: Path,
    dataset_id: str,
    arrow_files: Sequence[Path],
    metadata_file: Path | None,
) -> dict[str, Any]:
    if not dataset_id.strip():
        raise ValueError("--phase prepare requires --dataset-id")
    if not arrow_files:
        raise ValueError("--phase prepare requires at least one --arrow-file")
    if metadata_file is None:
        raise ValueError("--phase prepare requires --offline-metadata-file")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Greenfield prepare requires an empty run directory")

    try:
        report = prepare_data_readiness(
            dataset_id=dataset_id,
            arrow_files=arrow_files,
            metadata_file=metadata_file,
            output_dir=run_dir,
        )
    except (OSError, ValueError) as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        _finish(
            run_dir,
            None,
            dataset_id=dataset_id,
            trigger="prepare",
            state="failed",
            reason=str(exc),
            blocked_reason="data_readiness_failed",
            next_agent_action="fix_inputs",
            filenames=(
                "data_readiness_report.json",
                "series_semantic_dimensions.jsonl",
            ),
        )
        raise
    return _finish(
        run_dir,
        None,
        dataset_id=report.dataset_id,
        trigger="prepare",
        state="prepared",
        reason="Arrow and strict OfflineMetadata readiness passed.",
        blocked_reason="agent_semantics_required",
        next_agent_action="write_agent_semantics",
        filenames=(
            "data_readiness_report.json",
            "series_semantic_dimensions.jsonl",
            "agent_semantic_briefing.json",
        ),
        series_count=report.series_count,
    )


def _validate_semantics(run_dir: Path, current: RunState) -> dict[str, Any]:
    _require_state(current, "prepared", retry_phase="validate-semantics")
    if current.next_agent_action not in {
        "write_agent_semantics",
        "write_curation_strategy",
        _retry_action("validate-semantics"),
    }:
        raise _InvalidTransition("Run is not awaiting Agent semantics")
    readiness = _read_model(
        run_dir / "data_readiness_report.json",
        DataReadinessReport,
    )
    report = validate_semantic_readiness(
        data_readiness=readiness,
        agent_dataset_semantics_file=run_dir / "agent_dataset_semantics.json",
        agent_series_semantics_file=run_dir / "agent_series_semantics.jsonl",
        output_dir=run_dir,
    )
    return _finish(
        run_dir,
        current,
        trigger="validate-semantics",
        state="prepared",
        reason="Agent dataset and series semantics passed strict readiness.",
        blocked_reason="agent_strategy_required",
        next_agent_action="write_curation_strategy",
        filenames=(
            "agent_dataset_semantics.json",
            "agent_series_semantics.jsonl",
            "semantic_readiness_report.json",
        ),
        validated_series_count=len(report.validated_series_ids),
    )


def _validate_strategy(run_dir: Path, current: RunState) -> dict[str, Any]:
    if current.state in {"prepared", "strategy_revision_required"} or _retrying(
        current,
        "validate-strategy",
    ):
        if current.state == "prepared" and current.next_agent_action != "write_curation_strategy":
            raise _InvalidTransition("Validate Agent semantics before strategy")
        result = prepare_strategy_review(
            strategy_file=run_dir / STRATEGY_DRAFT_FILENAME,
            data_readiness_report_file=run_dir / "data_readiness_report.json",
            semantic_readiness_report_file=run_dir / "semantic_readiness_report.json",
            output_dir=run_dir,
        )
        return _finish(
            run_dir,
            current,
            trigger="validate-strategy",
            state="awaiting_strategy_review",
            reason="Strategy draft validated and review packet generated.",
            blocked_reason=result["blocked_reason"],
            next_agent_action="request_strategy_review",
            filenames=(STRATEGY_DRAFT_FILENAME, "strategy_review.md"),
            strategy_sha256=result["strategy_sha256"],
        )

    _require_state(current, "awaiting_strategy_review")
    result = apply_strategy_review(
        run_dir=run_dir,
        decision_file=run_dir / STRATEGY_DECISION_FILENAME,
    )
    filenames = [
        STRATEGY_DRAFT_FILENAME,
        "strategy_review.md",
        STRATEGY_DECISION_FILENAME,
    ]
    next_action = {
        "strategy_approved": "write_agent_search_inputs",
        "strategy_revision_required": "revise_strategy",
        "stopped": "stop",
    }[result["state"]]
    extra: dict[str, Any] = {"strategy_sha256": result["strategy_sha256"]}
    if result["state"] == "strategy_approved":
        extra["time_window_count"] = len(_write_time_windows(run_dir))
        filenames.extend((APPROVED_FILENAME, "time_windows.json"))
    return _finish(
        run_dir,
        current,
        trigger="validate-strategy",
        state=result["state"],
        reason="User strategy review decision applied.",
        blocked_reason=result["blocked_reason"],
        next_agent_action=next_action,
        filenames=filenames,
        **extra,
    )


def _intents(run_dir: Path, current: RunState) -> dict[str, Any]:
    _require_state(
        current,
        "strategy_approved",
        "searching",
        retry_phase="intents",
    )
    strategy = _read_model(run_dir / APPROVED_FILENAME, CurationStrategy)
    windows = _read_models(run_dir / "time_windows.json", TimeWindowRecord)
    semantics = _read_models(
        run_dir / "series_semantic_dimensions.jsonl",
        SeriesSemanticDimensionsRecord,
    )
    briefing = build_agent_query_briefing(strategy, windows, semantics)
    _write_json(run_dir / "agent_query_briefing.json", briefing)
    return _finish(
        run_dir,
        current,
        trigger="intents",
        state="searching",
        reason="Agent query briefing generated from the approved strategy.",
        blocked_reason="agent_search_inputs_required",
        next_agent_action="write_agent_search_inputs",
        filenames=(
            "agent_query_briefing.json",
            "agent_window_search_plans.jsonl",
            "agent_query_intents.jsonl",
        ),
        search_window_count=len(briefing["windows"]),
    )


def _agent_loop(
    run_dir: Path,
    current: RunState,
    retry_failed: bool,
) -> dict[str, Any]:
    _require_state(
        current,
        "strategy_approved",
        "searching",
        "revision_required",
        retry_phase="agent-loop",
    )
    if current.state == "failed" and not retry_failed:
        raise _InvalidTransition(
            "Retrying a failed search requires --retry-failed"
        )
    if current.state == "revision_required" and current.next_agent_action != "revise_search":
        raise _InvalidTransition(
            "Quality review did not route this run back to search"
        )

    result = run_search_iteration(run_dir, retry_failed=retry_failed)
    statuses = {
        window["status"] for window in result.get("search_windows", [])
    }
    if statuses and statuses <= _SEARCH_TERMINAL:
        state, blocked, action = (
            "curating",
            "agent_canonical_events_required",
            "write_canonical_events",
        )
    elif "failed" in statuses:
        state, blocked, action = (
            "failed",
            "search_source_failure",
            "retry_agent_loop",
        )
    elif result.get("blocked_reason") == "user_strategy_review_required":
        state = "awaiting_strategy_review"
        blocked = result["blocked_reason"]
        action = result["next_agent_action"]
    else:
        state = "searching"
        blocked = result.get("blocked_reason", "")
        action = result.get("next_agent_action", "")
    extra = {
        key: value
        for key, value in result.items()
        if key not in {"blocked_reason", "next_agent_action"}
    }
    return _finish(
        run_dir,
        current,
        trigger="agent-loop",
        state=state,
        reason="Feedback-driven search iteration completed.",
        blocked_reason=blocked,
        next_agent_action=action,
        filenames=(
            "agent_window_search_plans.jsonl",
            "agent_query_intents.jsonl",
            "search_state.json",
            "search_trace.jsonl",
            "holiday_impacts.jsonl",
            "discovered_events.jsonl",
            "agent_event_selections.jsonl",
        ),
        **extra,
    )


def _validate_canonical(run_dir: Path, current: RunState) -> dict[str, Any]:
    _require_state(
        current,
        "curating",
        "revision_required",
        retry_phase="validate-canonical-events",
    )
    if (
        current.state == "revision_required"
        and current.next_agent_action != "revise_canonicalization"
    ):
        raise _InvalidTransition(
            "Quality review did not route this run back to canonicalization"
        )
    result = validate_canonical_events(run_dir)
    return _finish(
        run_dir,
        current,
        trigger="validate-canonical-events",
        state="projecting",
        reason="Agent canonical events passed current-run validation.",
        blocked_reason="agent_projection_rules_required",
        next_agent_action="write_projection_rules",
        filenames=("canonical_events.jsonl",),
        **result,
    )


def _project(run_dir: Path, current: RunState) -> dict[str, Any]:
    _require_state(
        current,
        "projecting",
        "revision_required",
        retry_phase="project",
    )
    if current.state == "revision_required" and current.next_agent_action != "revise_projection":
        raise _InvalidTransition(
            "Quality review did not route this run back to projection"
        )
    result = project_events(run_dir)
    return _finish(
        run_dir,
        current,
        trigger="project",
        state="auditing",
        reason="Projection rules validated and stable links generated.",
        blocked_reason="",
        next_agent_action="run_quality_audit",
        filenames=(
            "projection_rules.jsonl",
            "projected_event_series_links.jsonl",
        ),
        **result,
    )


def _audit(run_dir: Path, current: RunState) -> dict[str, Any]:
    if _retrying(current, "audit") or (
        current.state == "auditing"
        and current.next_agent_action == "run_quality_audit"
    ) or (
        current.state == "failed"
        and current.next_agent_action == "fix_hard_gate"
    ):
        briefing = build_quality_audit_briefing(run_dir)
        if briefing["state"] == "failed":
            return _finish(
                run_dir,
                current,
                trigger="audit",
                state="failed",
                reason="Deterministic hard quality gate failed.",
                blocked_reason="hard_quality_gate_failed",
                next_agent_action="fix_hard_gate",
                filenames=("quality_audit_briefing.json",),
                hard_gates=briefing["hard_gates"],
            )
        return _finish(
            run_dir,
            current,
            trigger="audit",
            state="auditing",
            reason="Deterministic quality audit briefing generated.",
            blocked_reason="agent_quality_report_required",
            next_agent_action="write_quality_report",
            filenames=("quality_audit_briefing.json",),
            audit_input_sha256=briefing["audit_input_sha256"],
            hard_gates=briefing["hard_gates"],
        )

    if current.state == "auditing" and current.next_agent_action == "write_quality_report":
        result = prepare_quality_review(
            run_dir=run_dir,
            quality_report_file=run_dir / QUALITY_REPORT_FILENAME,
        )
        return _finish(
            run_dir,
            current,
            trigger="audit",
            state=result["state"],
            reason="Agent quality report validated and user packet generated.",
            blocked_reason=result["blocked_reason"],
            next_agent_action=result["next_agent_action"],
            filenames=(
                "quality_audit_briefing.json",
                QUALITY_REPORT_FILENAME,
                "quality_review.md",
            ),
            quality_report_sha256=result["quality_report_sha256"],
        )

    _require_state(current, "awaiting_quality_review")
    result = apply_quality_review(
        run_dir=run_dir,
        decision_file=run_dir / QUALITY_DECISION_FILENAME,
    )
    if result["state"] == "revision_required":
        result["invalidated_artifacts"] = _apply_revision(run_dir, result)
    extra = {
        key: value
        for key, value in result.items()
        if key not in {"state", "blocked_reason", "next_agent_action"}
    }
    return _finish(
        run_dir,
        current,
        trigger="audit",
        state=result["state"],
        reason="User quality review decision applied.",
        blocked_reason=result["blocked_reason"],
        next_agent_action=result["next_agent_action"],
        filenames=(
            QUALITY_REPORT_FILENAME,
            "quality_review.md",
            QUALITY_DECISION_FILENAME,
        ),
        **extra,
    )


def _publish(run_dir: Path, current: RunState) -> dict[str, Any]:
    _require_state(current, "approved", retry_phase="publish")
    result = publish_minimal(run_dir)
    result.update(
        {
            "blocked_reason": "",
            "next_agent_action": "complete",
            "artifact_paths": _paths(
                run_dir,
                (
                    "dataset_info.json",
                    "unified_event_series.jsonl",
                    "unified_events.jsonl",
                ),
            ),
        }
    )
    return result


def _apply_revision(run_dir: Path, result: dict[str, Any]) -> list[str]:
    return_phase = result["return_phase"]
    if return_phase == "search":
        targets = set(result["target_window_ids"])
        states = _read_models(
            run_dir / "search_state.json",
            SearchWindowStateRecord,
        )
        known = {state.window_id for state in states}
        unknown = sorted(targets - known)
        if unknown:
            raise ValueError(
                "Quality search revision references unknown time windows: "
                + ", ".join(unknown)
            )
        revised = [
            state.model_copy(
                update={
                    "status": "awaiting_refinement",
                    "next_agent_action": "write_window_refined_query",
                }
            )
            if state.window_id in targets
            else state
            for state in states
        ]
        _write_json(
            run_dir / "search_state.json",
            [state.model_dump(mode="json") for state in revised],
        )
        invalidated = (
            "canonical_events.jsonl",
            "projection_rules.jsonl",
            "projected_event_series_links.jsonl",
            QUALITY_REPORT_FILENAME,
            "quality_review.md",
        )
    elif return_phase == "canonicalization":
        invalidated = (
            "canonical_events.jsonl",
            "projection_rules.jsonl",
            "projected_event_series_links.jsonl",
            QUALITY_REPORT_FILENAME,
            "quality_review.md",
        )
    else:
        invalidated = (
            "projection_rules.jsonl",
            "projected_event_series_links.jsonl",
            QUALITY_REPORT_FILENAME,
            "quality_review.md",
        )
    for filename in invalidated:
        path = run_dir / filename
        if path.exists():
            path.unlink()
    return list(invalidated)


def _write_time_windows(run_dir: Path) -> list[TimeWindowRecord]:
    readiness = _read_model(
        run_dir / "data_readiness_report.json",
        DataReadinessReport,
    )
    semantics = _read_models(
        run_dir / "series_semantic_dimensions.jsonl",
        SeriesSemanticDimensionsRecord,
    )
    strategy = _read_model(run_dir / APPROVED_FILENAME, CurationStrategy)
    windows = build_time_windows(readiness, semantics, strategy)
    _write_json(
        run_dir / "time_windows.json",
        [window.model_dump(mode="json") for window in windows],
    )
    return windows


def _finish(
    run_dir: Path,
    current: RunState | None,
    *,
    trigger: str,
    state: str,
    reason: str,
    blocked_reason: str,
    next_agent_action: str,
    filenames: Sequence[str],
    dataset_id: str = "",
    **extra: Any,
) -> dict[str, Any]:
    artifacts = _paths(run_dir, filenames)
    timestamp = datetime.now(timezone.utc)
    hashes = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in artifacts.items()
        if Path(path).is_file()
    }
    run_state = RunState(
        dataset_id=dataset_id or current.dataset_id,
        run_id=run_dir.name,
        state=state,
        updated_at=timestamp,
        reason=reason,
        blocked_reason=blocked_reason,
        next_agent_action=next_agent_action,
        artifact_sha256=hashes,
    )
    _write_json(run_dir / RUN_STATE_FILENAME, run_state.model_dump(mode="json"))
    history = {
        "timestamp": timestamp.isoformat(),
        "previous_state": current.state if current else None,
        "next_state": state,
        "trigger": trigger,
        "input_sha256": hashes,
        "reason": reason,
    }
    with (run_dir / RUN_HISTORY_FILENAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(history, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "state": state,
        "blocked_reason": blocked_reason,
        "next_agent_action": next_agent_action,
        "run_dir": str(run_dir),
        "artifact_paths": artifacts,
        **extra,
    }


def _load_state(run_dir: Path) -> RunState:
    path = run_dir / RUN_STATE_FILENAME
    if not path.exists():
        raise ValueError(f"Missing v3 run state: {path}")
    return _read_model(path, RunState)


def _require_state(
    current: RunState,
    *allowed: str,
    retry_phase: str = "",
) -> None:
    if retry_phase and _retrying(current, retry_phase):
        return
    if current.state not in allowed:
        raise _InvalidTransition(
            f"Phase is invalid while run state is {current.state!r}; "
            f"expected one of {', '.join(allowed)}"
        )


def _retry_action(phase: str) -> str:
    return "retry_" + phase.replace("-", "_")


def _retrying(current: RunState, phase: str) -> bool:
    return (
        current.state == "failed"
        and current.next_agent_action == _retry_action(phase)
    )


def _paths(run_dir: Path, filenames: Sequence[str]) -> dict[str, str]:
    return {name: str(run_dir / name) for name in filenames}


def _read_model(path: Path, model):
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _read_models(path: Path, model):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    payloads = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    return [model.model_validate(item) for item in payloads]


def _read_optional_models(path: Path, model):
    return _read_models(path, model) if path.exists() else []


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, models: Sequence[Any]) -> None:
    path.write_text(
        "".join(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for model in models
        ),
        encoding="utf-8",
    )

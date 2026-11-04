"""Publish the strict three-file minimal contract after quality approval."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from world2ts_core.minimal_semantics import validate_agent_authored_semantics

from .canonicalization import load_canonical_events
from .models import DataReadinessReport, RunState
from .projection import validate_projected_links
from .quality_audit import quality_publish_status
from .readiness import validate_current_agent_semantics

DATASET_INFO_FILENAME = "dataset_info.json"
EVENT_SERIES_FILENAME = "unified_event_series.jsonl"
EVENTS_FILENAME = "unified_events.jsonl"
LEGACY_SERIES_FILENAME = "unified_event_series_samples.jsonl"
RUN_STATE_FILENAME = "run_state.json"
RUN_HISTORY_FILENAME = "run_history.jsonl"

_PUBLIC_FILES = (
    DATASET_INFO_FILENAME,
    EVENT_SERIES_FILENAME,
    EVENTS_FILENAME,
)
_DATASET_KEYS = {"dataset_id", "description", "domain"}
_SERIES_KEYS = {"series_id", "description", "time_range", "event_links"}
_EVENT_KEYS = {"event_id", "timestamp", "summary"}
_LINK_KEYS = {"event_id"}


def publish_minimal(run_dir: Path) -> dict[str, Any]:
    """Validate and transactionally publish the current approved run."""

    gate = quality_publish_status(run_dir)
    if not gate["can_publish"]:
        raise ValueError(
            "Minimal publication is blocked: "
            f"{gate['blocked_reason']} ({gate['next_agent_action']})"
        )

    state_path = run_dir / RUN_STATE_FILENAME
    previous_state = (
        _read_model(state_path, RunState).state
        if state_path.exists()
        else None
    )
    readiness = _read_model(
        run_dir / "data_readiness_report.json",
        DataReadinessReport,
    )
    dataset, series_by_id = validate_agent_authored_semantics(
        agent_dataset_semantics_file=run_dir / "agent_dataset_semantics.json",
        agent_series_semantics_file=run_dir / "agent_series_semantics.jsonl",
    )
    validate_current_agent_semantics(dataset, series_by_id, readiness)

    events = load_canonical_events(run_dir)
    links = validate_projected_links(run_dir)
    event_by_id = {event.event_id: event for event in events}
    referenced_event_ids = {link.event_id for link in links}
    missing = sorted(referenced_event_ids - set(event_by_id))
    if missing:
        raise ValueError(
            "Projected links reference unknown canonical events: "
            + ", ".join(missing)
        )

    dataset_row = {
        "dataset_id": dataset["dataset_id"],
        "description": dataset["description"],
        "domain": dataset["domain"],
    }
    links_by_series: dict[str, list[str]] = defaultdict(list)
    for link in links:
        links_by_series[link.series_id].append(link.event_id)
    series_rows = [
        {
            "series_id": series_id,
            "description": series_by_id[series_id]["description"],
            "time_range": series_by_id[series_id]["time_range"],
            "event_links": [
                {"event_id": event_id}
                for event_id in sorted(set(links_by_series[series_id]))
            ],
        }
        for series_id in readiness.series_ids
    ]
    event_rows = [
        {
            "event_id": event.event_id,
            "timestamp": event.event_start.isoformat(),
            "summary": event.summary,
        }
        for event in sorted(
            (
                event_by_id[event_id]
                for event_id in referenced_event_ids
            ),
            key=lambda item: (item.event_start, item.event_id),
        )
    ]
    _validate_public_contract(
        dataset_row,
        series_rows,
        event_rows,
        readiness.series_ids,
    )

    with tempfile.TemporaryDirectory(
        prefix=".publication-",
        dir=run_dir,
    ) as temporary:
        staged_dir = Path(temporary)
        _write_json(staged_dir / DATASET_INFO_FILENAME, dataset_row)
        _write_jsonl(staged_dir / EVENT_SERIES_FILENAME, series_rows)
        _write_jsonl(staged_dir / EVENTS_FILENAME, event_rows)
        hashes = {
            filename: _file_sha256(staged_dir / filename)
            for filename in _PUBLIC_FILES
        }
        state = RunState(
            dataset_id=readiness.dataset_id,
            run_id=run_dir.name,
            state="published",
            updated_at=datetime.now(timezone.utc),
            reason="Minimal three-file contract published after user quality approval.",
            artifact_sha256=hashes,
        )
        _write_model(staged_dir / RUN_STATE_FILENAME, state)
        history_path = staged_dir / RUN_HISTORY_FILENAME
        existing_history = run_dir / RUN_HISTORY_FILENAME
        prior = (
            existing_history.read_text(encoding="utf-8")
            if existing_history.exists()
            else ""
        )
        history_path.write_text(
            prior
            + json.dumps(
                {
                    "timestamp": state.updated_at.isoformat(),
                    "previous_state": previous_state,
                    "next_state": "published",
                    "trigger": "publish",
                    "input_sha256": hashes,
                    "reason": state.reason,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _replace_transaction(
            run_dir,
            staged_dir,
            [*_PUBLIC_FILES, RUN_STATE_FILENAME, RUN_HISTORY_FILENAME],
            delete=[LEGACY_SERIES_FILENAME],
        )

    return {
        "state": "published",
        "dataset_id": readiness.dataset_id,
        "run_dir": str(run_dir),
        "dataset_info_path": str(run_dir / DATASET_INFO_FILENAME),
        "event_series_path": str(run_dir / EVENT_SERIES_FILENAME),
        "events_path": str(run_dir / EVENTS_FILENAME),
        "num_series": len(series_rows),
        "num_events": len(event_rows),
        "num_event_links": len(links),
        "quality_report_sha256": gate["quality_report_sha256"],
    }


def _validate_public_contract(
    dataset: dict[str, Any],
    series: list[dict[str, Any]],
    events: list[dict[str, Any]],
    expected_series_ids: list[str],
) -> None:
    if set(dataset) != _DATASET_KEYS:
        raise AssertionError("Internal dataset_info payload violates exact schema")
    if [row.get("series_id") for row in series] != expected_series_ids:
        raise ValueError("Public series rows do not preserve readiness series order")
    if any(set(row) != _SERIES_KEYS for row in series):
        raise AssertionError("Internal event-series payload violates exact schema")
    if any(set(row) != _EVENT_KEYS for row in events):
        raise AssertionError("Internal event payload violates exact schema")

    event_ids = [row["event_id"] for row in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Public event IDs must be unique")
    known_events = set(event_ids)
    for row in series:
        link_ids = []
        for link in row["event_links"]:
            if set(link) != _LINK_KEYS:
                raise AssertionError("Internal event link violates exact schema")
            if link["event_id"] not in known_events:
                raise ValueError("Public link references an unknown event")
            link_ids.append(link["event_id"])
        if len(link_ids) != len(set(link_ids)):
            raise ValueError(
                f"Public series {row['series_id']!r} contains duplicate event links"
            )


def _replace_transaction(
    run_dir: Path,
    staged_dir: Path,
    filenames: list[str],
    *,
    delete: list[str],
) -> None:
    backup_dir = staged_dir / "backup"
    backup_dir.mkdir()
    backed_up = []
    installed = []
    try:
        for filename in [*filenames, *delete]:
            target = run_dir / filename
            if target.exists():
                os.replace(target, backup_dir / filename)
                backed_up.append(filename)
        for filename in filenames:
            os.replace(staged_dir / filename, run_dir / filename)
            installed.append(filename)
    except Exception:
        for filename in installed:
            (run_dir / filename).unlink(missing_ok=True)
        for filename in backed_up:
            os.replace(backup_dir / filename, run_dir / filename)
        raise


def _read_model(path: Path, model):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}")
    return model.model_validate(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_model(path: Path, model) -> None:
    _write_json(path, model.model_dump(mode="json"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

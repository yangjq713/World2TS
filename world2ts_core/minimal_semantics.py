"""Strict publication export for agent-authored minimal semantics.

The exporter deliberately does not infer or generate semantic descriptions.
Agents provide complete dataset- and series-level text; this module validates
that input and mechanically joins it to the existing event links.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DATASET_INFO_FILENAME = "dataset_info.json"
EVENT_SERIES_FILENAME = "unified_event_series.jsonl"
LEGACY_EVENT_SERIES_FILENAME = "unified_event_series_samples.jsonl"
EVENTS_FILENAME = "unified_events.jsonl"

_AGENT_DATASET_KEYS = {
    "dataset_id",
    "description",
    "domain",
    "frequency_phrases",
}
_AGENT_SERIES_KEYS = {
    "series_id",
    "description",
    "time_range",
    "frequency_phrase",
}
_TIME_RANGE_KEYS = {"start", "end"}
_PUBLIC_DATASET_KEYS = {"dataset_id", "description", "domain"}
_PUBLIC_SERIES_KEYS = {"series_id", "description", "time_range", "event_links"}
_PUBLIC_EVENT_KEYS = {"event_id", "timestamp", "summary"}
_PUBLIC_LINK_KEYS = {"event_id"}
_TERMINAL_PUNCTUATION = (".", "!", "?", "。", "！", "？")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys do not match contract: missing={missing} extra={extra}")


def _validate_description(description: Any, *, label: str, frequency_phrases: Iterable[str]) -> str:
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{label} description must be a non-empty string")
    text = description.strip()
    if len(text) < 30 or not text.endswith(_TERMINAL_PUNCTUATION):
        raise ValueError(f"{label} description must be a complete natural-language sentence")
    lowered = text.casefold()
    for phrase in frequency_phrases:
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError(f"{label} frequency phrase must be a non-empty string")
        if phrase.strip().casefold() not in lowered:
            raise ValueError(
                f"{label} description does not contain agent-declared frequency phrase: {phrase!r}"
            )
    forbidden = ("todo", "<unknown>", "<description>", "n/a description")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"{label} description contains placeholder text")
    return text


def _parse_time_range(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} time_range must be an object")
    _require_exact_keys(value, _TIME_RANGE_KEYS, f"{label} time_range")
    start_text = value.get("start")
    end_text = value.get("end")
    if not isinstance(start_text, str) or not isinstance(end_text, str):
        raise ValueError(f"{label} time_range start/end must be ISO-8601 strings")
    try:
        start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} time_range is not valid ISO-8601") from exc
    if end < start:
        raise ValueError(f"{label} time_range end precedes start")
    return {"start": start_text, "end": end_text}


def _load_existing_series(run_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    current = run_dir / EVENT_SERIES_FILENAME
    legacy = run_dir / LEGACY_EVENT_SERIES_FILENAME
    if current.exists() and legacy.exists():
        raise ValueError(
            f"Both {EVENT_SERIES_FILENAME} and {LEGACY_EVENT_SERIES_FILENAME} exist in {run_dir}"
        )
    source = current if current.exists() else legacy
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {EVENT_SERIES_FILENAME} or {LEGACY_EVENT_SERIES_FILENAME} in {run_dir}"
        )
    rows = _read_jsonl(source)
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        allowed = _PUBLIC_SERIES_KEYS if source == current else {"series_id", "event_links"}
        _require_exact_keys(row, allowed, f"series row {index}")
        series_id = row.get("series_id")
        if not isinstance(series_id, str) or not series_id:
            raise ValueError(f"series row {index} has an invalid series_id")
        if series_id in seen:
            raise ValueError(f"Duplicate series_id in {source}: {series_id}")
        seen.add(series_id)
        links = row.get("event_links")
        if not isinstance(links, list):
            raise ValueError(f"series {series_id} event_links must be a list")
        link_ids: set[str] = set()
        for link in links:
            if not isinstance(link, dict):
                raise ValueError(f"series {series_id} contains a non-object event link")
            _require_exact_keys(link, _PUBLIC_LINK_KEYS, f"series {series_id} event link")
            event_id = link.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(f"series {series_id} contains an invalid event_id")
            if event_id in link_ids:
                raise ValueError(f"series {series_id} contains duplicate event_id {event_id}")
            link_ids.add(event_id)
    return source, rows


def _validate_events(run_dir: Path) -> set[str]:
    path = run_dir / EVENTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing {EVENTS_FILENAME} in {run_dir}")
    event_ids: set[str] = set()
    for index, row in enumerate(_read_jsonl(path), 1):
        _require_exact_keys(row, _PUBLIC_EVENT_KEYS, f"event row {index}")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"event row {index} has an invalid event_id")
        if event_id in event_ids:
            raise ValueError(f"Duplicate event_id in {path}: {event_id}")
        event_ids.add(event_id)
        if not isinstance(row.get("timestamp"), str) or not row["timestamp"]:
            raise ValueError(f"event {event_id} has an invalid timestamp")
        if not isinstance(row.get("summary"), str) or not row["summary"].strip():
            raise ValueError(f"event {event_id} has an empty summary")
    return event_ids


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def validate_agent_authored_semantics(
    *,
    agent_dataset_semantics_file: Path,
    agent_series_semantics_file: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Strictly load Agent semantics without inferring missing content."""

    agent_dataset = _read_json(agent_dataset_semantics_file.expanduser())
    _require_exact_keys(agent_dataset, _AGENT_DATASET_KEYS, "agent dataset semantics")
    dataset_id = agent_dataset.get("dataset_id")
    domain = agent_dataset.get("domain")
    frequency_phrases = agent_dataset.get("frequency_phrases")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("agent dataset semantics dataset_id must be non-empty")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("agent dataset semantics domain must be non-empty")
    if not isinstance(frequency_phrases, list) or not frequency_phrases:
        raise ValueError("agent dataset semantics frequency_phrases must be a non-empty list")
    validated_dataset = {
        "dataset_id": dataset_id.strip(),
        "description": _validate_description(
            agent_dataset.get("description"),
            label=f"dataset {dataset_id}",
            frequency_phrases=frequency_phrases,
        ),
        "domain": domain.strip(),
        "frequency_phrases": frequency_phrases,
    }

    semantic_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_read_jsonl(agent_series_semantics_file.expanduser()), 1):
        _require_exact_keys(row, _AGENT_SERIES_KEYS, f"agent series semantics row {index}")
        series_id = row.get("series_id")
        if not isinstance(series_id, str) or not series_id:
            raise ValueError(f"agent series semantics row {index} has an invalid series_id")
        if series_id in semantic_by_id:
            raise ValueError(f"Duplicate agent series semantics for {series_id}")
        frequency_phrase = row.get("frequency_phrase")
        if not isinstance(frequency_phrase, str) or not frequency_phrase.strip():
            raise ValueError(f"agent series semantics {series_id} has no frequency_phrase")
        semantic_by_id[series_id] = {
            "series_id": series_id,
            "description": _validate_description(
                row.get("description"),
                label=f"series {series_id}",
                frequency_phrases=[frequency_phrase],
            ),
            "time_range": _parse_time_range(row.get("time_range"), label=f"series {series_id}"),
            "frequency_phrase": frequency_phrase.strip(),
        }
    return validated_dataset, semantic_by_id


def export_agent_authored_minimal_semantics(
    *,
    run_dir: Path,
    agent_dataset_semantics_file: Path,
    agent_series_semantics_file: Path,
    remove_legacy_series_file: bool = True,
) -> dict[str, Any]:
    """Validate agent-authored semantics and publish the three-file contract.

    There is intentionally no semantic fallback. Missing, incomplete, or
    contradictory agent inputs fail before any publication file is changed.
    """

    run_dir = run_dir.expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    source_path, existing_rows = _load_existing_series(run_dir)
    event_ids = _validate_events(run_dir)

    agent_dataset, semantic_by_id = validate_agent_authored_semantics(
        agent_dataset_semantics_file=agent_dataset_semantics_file,
        agent_series_semantics_file=agent_series_semantics_file,
    )
    dataset_id = agent_dataset["dataset_id"]

    existing_ids = {row["series_id"] for row in existing_rows}
    semantic_ids = set(semantic_by_id)
    if semantic_ids != existing_ids:
        raise ValueError(
            "Agent series semantics IDs do not match publication series IDs: "
            f"missing={sorted(existing_ids - semantic_ids)} extra={sorted(semantic_ids - existing_ids)}"
        )

    unresolved_event_ids = {
        link["event_id"]
        for row in existing_rows
        for link in row["event_links"]
        if link["event_id"] not in event_ids
    }
    if unresolved_event_ids:
        raise ValueError(f"Event links reference unknown events: {sorted(unresolved_event_ids)}")

    public_dataset = {
        "dataset_id": dataset_id,
        "description": agent_dataset["description"],
        "domain": agent_dataset["domain"],
    }
    _require_exact_keys(public_dataset, _PUBLIC_DATASET_KEYS, "public dataset info")

    public_series = []
    for existing in existing_rows:
        semantic = semantic_by_id[existing["series_id"]]
        public_series.append(
            {
                "series_id": existing["series_id"],
                "description": semantic["description"],
                "time_range": semantic["time_range"],
                "event_links": existing["event_links"],
            }
        )
    if any(set(row) != _PUBLIC_SERIES_KEYS for row in public_series):
        raise AssertionError("Internal error: public series payload violates exact schema")

    dataset_output = run_dir / DATASET_INFO_FILENAME
    series_output = run_dir / EVENT_SERIES_FILENAME
    _atomic_write_json(dataset_output, public_dataset)
    _atomic_write_jsonl(series_output, public_series)

    legacy = run_dir / LEGACY_EVENT_SERIES_FILENAME
    if remove_legacy_series_file and legacy.exists():
        legacy.unlink()

    return {
        "dataset_id": dataset_id,
        "run_dir": str(run_dir),
        "source_series_path": str(source_path),
        "dataset_info_path": str(dataset_output),
        "event_series_path": str(series_output),
        "events_path": str(run_dir / EVENTS_FILENAME),
        "num_series": len(public_series),
        "num_events": len(event_ids),
        "num_event_links": sum(len(row["event_links"]) for row in public_series),
        "legacy_series_file_removed": remove_legacy_series_file and not legacy.exists(),
    }

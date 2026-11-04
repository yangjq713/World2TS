"""Validate Agent-authored canonical events for the current run."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from world2ts_core.events import DiscoveredEvent

from .models import (
    AgentEventSelectionRecord,
    CanonicalEventRecord,
    CurationStrategy,
)
from .strategy import APPROVED_FILENAME

CANONICAL_EVENTS_FILENAME = "canonical_events.jsonl"
DISCOVERED_EVENTS_FILENAME = "discovered_events.jsonl"
SELECTIONS_FILENAME = "agent_event_selections.jsonl"

_SUMMARY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def validate_canonical_events(run_dir: Path) -> dict[str, Any]:
    """Validate the current run's canonical event artifact.

    Validation is deliberately limited to deterministic boundary checks.
    Duplicate episodes are reported for Agent review and never auto-merged.
    """

    strategy = _read_model(run_dir / APPROVED_FILENAME, CurationStrategy)
    candidates = _read_jsonl(
        run_dir / DISCOVERED_EVENTS_FILENAME,
        DiscoveredEvent,
    )
    selections = _read_jsonl(
        run_dir / SELECTIONS_FILENAME,
        AgentEventSelectionRecord,
    )
    events = load_canonical_events(run_dir)

    candidate_by_id = _candidate_index(candidates)
    selected_source_ids = _selected_source_ids(selections, candidate_by_id)
    _validate_events(events, candidate_by_id, selected_source_ids, strategy)

    referenced = {
        source_id
        for event in events
        for source_id in event.source_event_summary_ids
    }
    selected = set(selected_source_ids)

    return {
        "status": "valid",
        "canonical_event_count": len(events),
        "selected_source_event_count": len(selected),
        "uncanonicalized_selected_source_event_ids": sorted(selected - referenced),
        "duplicate_hints": suspected_duplicate_episodes(events, candidate_by_id),
    }


def load_canonical_events(run_dir: Path) -> list[CanonicalEventRecord]:
    """Load the strict Agent-authored canonical event JSONL."""

    return _read_jsonl(
        run_dir / CANONICAL_EVENTS_FILENAME,
        CanonicalEventRecord,
    )


def suspected_duplicate_episodes(
    events: list[CanonicalEventRecord],
    candidate_by_id: dict[str, DiscoveredEvent] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic duplicate hints without merging any events."""

    candidates = candidate_by_id or {}
    hints: list[dict[str, Any]] = []
    for left, right in combinations(sorted(events, key=lambda item: item.event_id), 2):
        shared_sources = sorted(
            set(left.source_event_summary_ids) & set(right.source_event_summary_ids)
        )
        reasons = []
        if shared_sources:
            reasons.append("shared_source_event")
        if (
            left.event_family.casefold() == right.event_family.casefold()
            and _date_gap_days(left, right) <= 1
            and _token_overlap(left.summary, right.summary) >= 0.5
        ):
            reasons.append("same_family_near_date_similar_summary")
        if not reasons:
            continue

        source_ids = set(left.source_event_summary_ids) | set(
            right.source_event_summary_ids
        )
        source_names = sorted(
            {
                candidates[source_id].source_name
                for source_id in source_ids
                if source_id in candidates
            }
        )
        window_ids = sorted(
            {
                candidates[source_id].window_id
                for source_id in source_ids
                if source_id in candidates and candidates[source_id].window_id
            }
        )
        hints.append(
            {
                "event_ids": [left.event_id, right.event_id],
                "reasons": reasons,
                "shared_source_event_summary_ids": shared_sources,
                "source_names": source_names,
                "window_ids": window_ids,
            }
        )
    return hints


def _candidate_index(candidates: list[DiscoveredEvent]) -> dict[str, DiscoveredEvent]:
    indexed: dict[str, DiscoveredEvent] = {}
    for candidate in candidates:
        source_id = candidate.event_summary_id
        if not source_id:
            raise ValueError("Current-run discovered event is missing event_summary_id")
        if source_id in indexed:
            raise ValueError(f"Duplicate discovered event_summary_id: {source_id}")
        indexed[source_id] = candidate
    return indexed


def _selected_source_ids(
    selections: list[AgentEventSelectionRecord],
    candidate_by_id: dict[str, DiscoveredEvent],
) -> set[str]:
    selected: set[str] = set()
    for selection in selections:
        for source_id in selection.selected_event_summary_ids:
            if source_id not in candidate_by_id:
                raise ValueError(
                    f"Selection references event outside the current run: {source_id}"
                )
            selected.add(source_id)
    return selected


def _validate_events(
    events: list[CanonicalEventRecord],
    candidate_by_id: dict[str, DiscoveredEvent],
    selected_source_ids: set[str],
    strategy: CurationStrategy,
) -> None:
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("canonical event_id values must be unique")

    for event in events:
        sources = []
        for source_id in event.source_event_summary_ids:
            candidate = candidate_by_id.get(source_id)
            if candidate is None:
                raise ValueError(
                    f"Canonical event references event outside the current run: {source_id}"
                )
            if source_id not in selected_source_ids:
                raise ValueError(
                    f"Canonical event references an unselected candidate: {source_id}"
                )
            sources.append(candidate)
        if event.context_role == "forecast" and not strategy.event_policy.allow_forecasts:
            raise ValueError("Canonical forecast is disallowed by the approved strategy")
        if (
            event.context_role == "retrospective"
            and not strategy.event_policy.allow_retrospective_context
        ):
            raise ValueError(
                "Canonical retrospective context is disallowed by the approved strategy"
            )
        if (
            strategy.event_policy.exclude_target_only_milestones
            and sources
            and all(_is_target_only(source) for source in sources)
        ):
            raise ValueError(
                f"Canonical event {event.event_id!r} is supported only by "
                "target-derived milestones"
            )
        if any(
            _normalized_text(event.summary)
            in {
                _normalized_text(source.title),
                _normalized_text(source.description),
            }
            for source in sources
        ):
            raise ValueError(
                f"Canonical event {event.event_id!r} summary must rewrite "
                "retrieved titles and snippets"
            )


def _is_target_only(candidate: DiscoveredEvent) -> bool:
    payload = candidate.structured_payload or {}
    return payload.get("target_only_milestone") is True


def _date_gap_days(left: CanonicalEventRecord, right: CanonicalEventRecord) -> int:
    left_end = (left.event_end or left.event_start).date()
    right_end = (right.event_end or right.event_start).date()
    left_start = left.event_start.date()
    right_start = right.event_start.date()
    if left_start <= right_end and right_start <= left_end:
        return 0
    if left_end < right_start:
        return (right_start - left_end).days
    return (left_start - right_end).days


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _summary_tokens(left)
    right_tokens = _summary_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _summary_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in _SUMMARY_STOPWORDS
    }


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _read_model(path: Path, model):
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path, model):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        raise ValueError(f"{path.name} must be JSONL, not a JSON array")
    return [
        model.model_validate(json.loads(line))
        for line in text.splitlines()
        if line.strip()
    ]

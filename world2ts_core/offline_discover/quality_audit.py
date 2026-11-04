"""Deterministic quality audit and the final user review gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from world2ts_core.events import DiscoveredEvent
from world2ts_core.minimal_semantics import validate_agent_authored_semantics

from .canonicalization import validate_canonical_events
from .models import (
    AgentEventSelectionRecord,
    CanonicalEventRecord,
    CurationStrategy,
    DataReadinessReport,
    ExampleEventLink,
    HolidayImpactRecord,
    QualityGateResult,
    QualityReport,
    QualityReviewDecision,
    SearchTraceRecord,
    SearchWindowStateRecord,
    SemanticProjectionRule,
    SemanticReadinessReport,
    SeriesSemanticDimensionsRecord,
)
from .projection import load_projection_rules, validate_projected_links
from .readiness import validate_current_agent_semantics
from .strategy import (
    APPROVED_FILENAME,
    DECISION_FILENAME as STRATEGY_DECISION_FILENAME,
    DRAFT_FILENAME as STRATEGY_DRAFT_FILENAME,
    strategy_search_status,
)

AUDIT_BRIEFING_FILENAME = "quality_audit_briefing.json"
QUALITY_REPORT_FILENAME = "quality_report.json"
QUALITY_REVIEW_FILENAME = "quality_review.md"
QUALITY_DECISION_FILENAME = "quality_review_decision.json"

HARD_GATE_ID = "structural_integrity"
CONTENT_GATE_IDS = (
    "event_text_self_containment",
    "temporal_correctness",
    "contextual_incremental_value",
    "series_event_mapping_quality",
    "coverage_and_diversity",
)
USER_REVIEW_REQUIRED = "user_quality_review_required"
QUALITY_REPORT_STALE = "quality_report_stale"
HARD_GATE_FAILED = "hard_quality_gate_failed"

_AUDIT_INPUT_FILENAMES = (
    STRATEGY_DRAFT_FILENAME,
    STRATEGY_DECISION_FILENAME,
    APPROVED_FILENAME,
    "data_readiness_report.json",
    "semantic_readiness_report.json",
    "agent_dataset_semantics.json",
    "agent_series_semantics.jsonl",
    "series_semantic_dimensions.jsonl",
    "agent_window_search_plans.jsonl",
    "agent_query_intents.jsonl",
    "search_state.json",
    "search_trace.jsonl",
    "holiday_impacts.jsonl",
    "discovered_events.jsonl",
    "agent_event_selections.jsonl",
    "canonical_events.jsonl",
    "projection_rules.jsonl",
    "projected_event_series_links.jsonl",
)
_ACCEPTABLE_SEARCH_STATUSES = {"covered", "exhausted", "no_event_found"}
_REVISION_ROUTES = {
    "revise_search": (
        "search",
        [
            "search_state.json",
            "search_trace.jsonl",
            "discovered_events.jsonl",
            "agent_event_selections.jsonl",
            "canonical_events.jsonl",
            "projection_rules.jsonl",
            "projected_event_series_links.jsonl",
            QUALITY_REPORT_FILENAME,
            QUALITY_REVIEW_FILENAME,
        ],
    ),
    "revise_events": (
        "canonicalization",
        [
            "canonical_events.jsonl",
            "projection_rules.jsonl",
            "projected_event_series_links.jsonl",
            QUALITY_REPORT_FILENAME,
            QUALITY_REVIEW_FILENAME,
        ],
    ),
    "revise_projection": (
        "projection",
        [
            "projection_rules.jsonl",
            "projected_event_series_links.jsonl",
            QUALITY_REPORT_FILENAME,
            QUALITY_REVIEW_FILENAME,
        ],
    ),
}


def build_quality_audit_briefing(run_dir: Path) -> dict[str, Any]:
    """Run hard checks and write the deterministic Agent audit briefing."""

    try:
        snapshot = _audit_snapshot(run_dir)
    except (OSError, ValueError) as exc:
        hard_gate = QualityGateResult(
            gate_id=HARD_GATE_ID,
            gate_type="hard",
            status="fail",
            evidence=[str(exc)],
            impact="Quality review and publication are blocked.",
            continue_search_recommended=False,
        )
        briefing = {
            "state": "failed",
            "blocked_reason": HARD_GATE_FAILED,
            "next_agent_action": "fix_hard_gate",
            "hard_gates": [hard_gate.model_dump(mode="json")],
            "required_content_gate_ids": list(CONTENT_GATE_IDS),
            "statistics": {},
            "review_hints": {},
        }
    else:
        hard_gate = QualityGateResult(
            gate_id=HARD_GATE_ID,
            gate_type="hard",
            status="pass",
            evidence=[
                "All required artifacts parse and their dataset, series, event, "
                "rule, link, search-state, and Agent-semantics references agree."
            ],
            impact="The artifact set is structurally eligible for content review.",
            continue_search_recommended=False,
        )
        briefing = {
            "state": "auditing",
            "blocked_reason": "agent_quality_report_required",
            "next_agent_action": "write_quality_report",
            "audit_input_sha256": snapshot["audit_input_sha256"],
            "hard_gates": [hard_gate.model_dump(mode="json")],
            "required_content_gate_ids": list(CONTENT_GATE_IDS),
            "content_gate_instructions": {
                "event_text_self_containment": (
                    "Judge whether every summary independently states when, where, "
                    "and what happened without relying on a URL or snippet."
                ),
                "temporal_correctness": (
                    "Judge canonical dates and ranges against the event text and evidence."
                ),
                "contextual_incremental_value": (
                    "Reject target-only milestones and target-curve restatements."
                ),
                "series_event_mapping_quality": (
                    "Judge whether each projection scope matches affected series semantics."
                ),
                "coverage_and_diversity": (
                    "Judge family, year, outline, zero-event, holiday, and broadcast coverage."
                ),
            },
            "report_requirements": (
                "For every content gate provide pass/warning/fail, evidence, impact, "
                "whether more search is worthwhile, and a recommended return phase "
                "when revision is needed."
            ),
            "statistics": snapshot["statistics"],
            "review_hints": snapshot["review_hints"],
        }

    _write_json(run_dir / AUDIT_BRIEFING_FILENAME, briefing)
    return briefing


def prepare_quality_review(
    *,
    run_dir: Path,
    quality_report_file: Path,
) -> dict[str, Any]:
    """Validate an Agent quality report and create the concise user packet."""

    briefing = build_quality_audit_briefing(run_dir)
    if briefing["state"] == "failed":
        raise ValueError(
            "Hard quality gate failed: "
            + briefing["hard_gates"][0]["evidence"][0]
        )

    report = _read_model(quality_report_file, QualityReport)
    strategy = _read_model(run_dir / APPROVED_FILENAME, CurationStrategy)
    if report.dataset_id != strategy.dataset_id:
        raise ValueError("Quality report and approved strategy dataset IDs must match")
    if report.audit_input_sha256 != briefing["audit_input_sha256"]:
        raise ValueError("Quality report is not bound to the current audit inputs")

    expected_hard_gates = [
        QualityGateResult.model_validate(item)
        for item in briefing["hard_gates"]
    ]
    if report.hard_gates != expected_hard_gates:
        raise ValueError("Agent quality report may not alter deterministic hard gates")
    content_gate_ids = [gate.gate_id for gate in report.content_gates]
    if set(content_gate_ids) != set(CONTENT_GATE_IDS) or len(content_gate_ids) != len(
        CONTENT_GATE_IDS
    ):
        raise ValueError(
            "Quality report must contain exactly the five required content gates"
        )
    if report.statistics != briefing["statistics"]:
        raise ValueError("Agent quality report may not alter deterministic statistics")
    if report.worth_continuing_search and report.recommended_return_phase is None:
        raise ValueError(
            "Reports recommending more work must name a recommended return phase"
        )
    _require_revision_input_change(run_dir, report)

    _write_model(run_dir / QUALITY_REPORT_FILENAME, report)
    digest = quality_report_sha256(report)
    review = _render_quality_review(report, digest)
    (run_dir / QUALITY_REVIEW_FILENAME).write_text(review, encoding="utf-8")
    return {
        "state": "awaiting_quality_review",
        "blocked_reason": USER_REVIEW_REQUIRED,
        "next_agent_action": "request_quality_review",
        "quality_report_sha256": digest,
        "quality_report_path": str(run_dir / QUALITY_REPORT_FILENAME),
        "review_path": str(run_dir / QUALITY_REVIEW_FILENAME),
    }


def apply_quality_review(
    *,
    run_dir: Path,
    decision_file: Path,
) -> dict[str, Any]:
    """Bind the user's final decision to the current quality report."""

    report = _read_model(run_dir / QUALITY_REPORT_FILENAME, QualityReport)
    _require_current_audit_inputs(run_dir, report)
    decision = _read_model(decision_file, QualityReviewDecision)
    digest = quality_report_sha256(report)
    if decision.reviewed_quality_report_sha256 != digest:
        raise ValueError(
            "Quality review decision hash does not match the current report"
        )
    if any(gate.status != "pass" for gate in report.hard_gates):
        raise ValueError("Hard quality-gate failures cannot be overridden")
    if decision.decision == "approve" and report.overall_status == "fail":
        raise ValueError("A failed content review cannot be approved")
    existing_decision_path = run_dir / QUALITY_DECISION_FILENAME
    if existing_decision_path.exists():
        existing = _read_model(existing_decision_path, QualityReviewDecision)
        if (
            existing.decision in _REVISION_ROUTES
            and decision.decision == "approve"
            and existing.reviewed_quality_report_sha256 == digest
        ):
            raise ValueError(
                "A revision decision requires changed audit inputs and a new "
                "quality report before approval"
            )

    _write_model(run_dir / QUALITY_DECISION_FILENAME, decision)
    if decision.decision == "approve":
        return {
            "state": "approved",
            "blocked_reason": "",
            "next_agent_action": "publish",
            "quality_report_sha256": digest,
        }
    if decision.decision == "stop":
        return {
            "state": "stopped",
            "blocked_reason": "user_stopped",
            "next_agent_action": "stop",
            "quality_report_sha256": digest,
        }

    return_phase, invalidated = _REVISION_ROUTES[decision.decision]
    return {
        "state": "revision_required",
        "blocked_reason": decision.decision,
        "next_agent_action": f"revise_{return_phase}",
        "return_phase": return_phase,
        "target_window_ids": decision.target_window_ids,
        "invalidated_artifacts": invalidated,
        "quality_report_sha256": digest,
    }


def quality_publish_status(run_dir: Path) -> dict[str, Any]:
    """Return the hard gate that every publication path must enforce."""

    def blocked(reason: str, action: str) -> dict[str, Any]:
        return {
            "approved": False,
            "can_publish": False,
            "blocked_reason": reason,
            "next_agent_action": action,
        }

    report_path = run_dir / QUALITY_REPORT_FILENAME
    if not report_path.exists():
        return blocked(USER_REVIEW_REQUIRED, "run_quality_audit")
    try:
        report = _read_model(report_path, QualityReport)
        _require_current_audit_inputs(run_dir, report)
    except (OSError, ValueError):
        return blocked(QUALITY_REPORT_STALE, "rerun_quality_audit")
    if any(gate.status != "pass" for gate in report.hard_gates):
        return blocked(HARD_GATE_FAILED, "fix_hard_gate")

    decision_path = run_dir / QUALITY_DECISION_FILENAME
    if not decision_path.exists():
        return blocked(USER_REVIEW_REQUIRED, "request_quality_review")
    decision = _read_model(decision_path, QualityReviewDecision)
    if decision.reviewed_quality_report_sha256 != quality_report_sha256(report):
        return blocked(QUALITY_REPORT_STALE, "request_quality_review")
    if decision.decision == "stop":
        return blocked("user_stopped", "stop")
    if decision.decision != "approve":
        return blocked(decision.decision, f"revise_{_REVISION_ROUTES[decision.decision][0]}")
    if report.overall_status == "fail":
        return blocked(HARD_GATE_FAILED, "revise_quality_failures")
    return {
        "approved": True,
        "can_publish": True,
        "blocked_reason": "",
        "next_agent_action": "publish",
        "quality_report_sha256": quality_report_sha256(report),
    }


def quality_report_sha256(report: QualityReport) -> str:
    """Hash one normalized quality report."""

    canonical = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_input_sha256(run_dir: Path) -> str:
    """Hash every internal artifact that the quality report audits."""

    digest = hashlib.sha256()
    for filename in _AUDIT_INPUT_FILENAMES:
        path = run_dir / filename
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _audit_snapshot(run_dir: Path) -> dict[str, Any]:
    gate = strategy_search_status(run_dir)
    if not gate["can_search"]:
        raise ValueError("Current strategy does not have a valid user approval")

    strategy = _read_model(run_dir / APPROVED_FILENAME, CurationStrategy)
    readiness = _read_model(
        run_dir / "data_readiness_report.json",
        DataReadinessReport,
    )
    semantic_readiness = _read_model(
        run_dir / "semantic_readiness_report.json",
        SemanticReadinessReport,
    )
    if readiness.status != "ready" or semantic_readiness.status != "ready":
        raise ValueError("Data and semantic readiness must both be ready")
    if not (
        strategy.dataset_id
        == readiness.dataset_id
        == semantic_readiness.dataset_id
    ):
        raise ValueError("Strategy and readiness dataset IDs must match")

    agent_dataset, agent_series = validate_agent_authored_semantics(
        agent_dataset_semantics_file=run_dir / "agent_dataset_semantics.json",
        agent_series_semantics_file=run_dir / "agent_series_semantics.jsonl",
    )
    validate_current_agent_semantics(
        agent_dataset,
        agent_series,
        readiness,
    )
    series_semantics = _read_jsonl(
        run_dir / "series_semantic_dimensions.jsonl",
        SeriesSemanticDimensionsRecord,
    )
    if [series.series_id for series in series_semantics] != readiness.series_ids:
        raise ValueError("Series dimensions do not match readiness series order")

    search_states = _read_records(
        run_dir / "search_state.json",
        SearchWindowStateRecord,
    )
    unfinished = [
        state.window_id
        for state in search_states
        if state.status not in _ACCEPTABLE_SEARCH_STATUSES
    ]
    if unfinished:
        raise ValueError(
            "Quality audit requires terminal successful search windows: "
            + ", ".join(unfinished)
        )
    traces = _read_jsonl(
        run_dir / "search_trace.jsonl",
        SearchTraceRecord,
    )
    _validate_trace_coverage(search_states, traces)
    holiday_impacts = _read_jsonl(
        run_dir / "holiday_impacts.jsonl",
        HolidayImpactRecord,
    )

    canonical_result = validate_canonical_events(run_dir)
    events = _read_jsonl(
        run_dir / "canonical_events.jsonl",
        CanonicalEventRecord,
    )
    rules = load_projection_rules(run_dir)
    links = validate_projected_links(run_dir)
    if (
        not strategy.quality_policy.zero_event_series_allowed
        and {link.series_id for link in links} != set(readiness.series_ids)
    ):
        raise ValueError(
            "Approved strategy does not allow zero-event series"
        )
    candidates = _read_jsonl(
        run_dir / "discovered_events.jsonl",
        DiscoveredEvent,
    )
    selections = _read_jsonl(
        run_dir / "agent_event_selections.jsonl",
        AgentEventSelectionRecord,
    )

    return {
        "audit_input_sha256": audit_input_sha256(run_dir),
        "statistics": _quality_statistics(
            strategy=strategy,
            readiness=readiness,
            series_semantics=series_semantics,
            search_states=search_states,
            traces=traces,
            holiday_impacts=holiday_impacts,
            events=events,
            rules=rules,
            links=links,
            candidates=candidates,
            selections=selections,
        ),
        "review_hints": _review_hints(
            events,
            canonical_result["duplicate_hints"],
            canonical_result["uncanonicalized_selected_source_event_ids"],
            rules,
            readiness.series_ids,
            links,
        ),
    }


def _validate_trace_coverage(
    states: list[SearchWindowStateRecord],
    traces: list[SearchTraceRecord],
) -> None:
    traced_queries = {trace.query_intent_id for trace in traces}
    missing = sorted(
        {
            query_id
            for state in states
            for query_id in (
                *state.attempted_initial_query_intent_ids,
                *state.attempted_refinement_query_intent_ids,
            )
            if query_id not in traced_queries
        }
    )
    if missing:
        raise ValueError(
            "Search trace is missing attempted query IDs: " + ", ".join(missing)
        )


def _quality_statistics(
    *,
    strategy: CurationStrategy,
    readiness: DataReadinessReport,
    series_semantics: list[SeriesSemanticDimensionsRecord],
    search_states: list[SearchWindowStateRecord],
    traces: list[SearchTraceRecord],
    holiday_impacts: list[HolidayImpactRecord],
    events: list[CanonicalEventRecord],
    rules: list[SemanticProjectionRule],
    links: list[ExampleEventLink],
    candidates: list[DiscoveredEvent],
    selections: list[AgentEventSelectionRecord],
) -> dict[str, Any]:
    links_by_series = Counter(link.series_id for link in links)
    links_by_event = Counter(link.event_id for link in links)
    counts = [links_by_series[series_id] for series_id in readiness.series_ids]
    family = defaultdict(lambda: {"events": 0, "links": 0})
    year = defaultdict(lambda: {"events": 0, "links": 0})
    for event in events:
        family[event.event_family]["events"] += 1
        family[event.event_family]["links"] += links_by_event[event.event_id]
        year[str(event.event_start.year)]["events"] += 1
        year[str(event.event_start.year)]["links"] += links_by_event[event.event_id]

    rule_by_event = {rule.event_id: rule for rule in rules}
    broadcast = [
        {
            "event_id": event.event_id,
            "series_count": links_by_event[event.event_id],
            "series_ratio": round(
                links_by_event[event.event_id] / len(readiness.series_ids),
                6,
            )
            if readiness.series_ids
            else 0.0,
        }
        for event in sorted(events, key=lambda item: item.event_id)
    ]
    outline = defaultdict(Counter)
    for trace in traces:
        if trace.outline_id:
            outline[trace.outline_id]["queries"] += 1
            outline[trace.outline_id]["returned"] += trace.returned_count
            outline[trace.outline_id]["failures"] += bool(trace.failure)
    source = defaultdict(lambda: {"queries": 0, "returned": 0, "failures": 0})
    for trace in traces:
        source[trace.source]["queries"] += 1
        source[trace.source]["returned"] += trace.returned_count
        source[trace.source]["failures"] += bool(trace.failure)

    source_use_count = Counter(
        source_id
        for event in events
        for source_id in event.source_event_summary_ids
    )
    return {
        "series_count": len(readiness.series_ids),
        "event_count": len(events),
        "link_count": len(links),
        "events_per_series": {
            "min": min(counts, default=0),
            "mean": round(sum(counts) / len(counts), 6) if counts else 0.0,
            "max": max(counts, default=0),
            "zero_event_series_count": sum(count == 0 for count in counts),
        },
        "event_family": dict(sorted(family.items())),
        "year": dict(sorted(year.items())),
        "broadcast": broadcast,
        "dataset_wide_event_ids": sorted(
            event_id
            for event_id, rule in rule_by_event.items()
            if rule.scope == "dataset"
        ),
        "search_window_status": dict(
            sorted(Counter(state.status for state in search_states).items())
        ),
        "outline_coverage": {
            outline_id: dict(sorted(statuses.items()))
            for outline_id, statuses in sorted(outline.items())
        },
        "dimension_coverage": _dimension_coverage(
            strategy,
            series_semantics,
            links_by_series,
        ),
        "source_trace": {
            source_name: values
            for source_name, values in sorted(source.items())
        },
        "holiday_impact": {
            "evaluated": len(holiday_impacts),
            "accepted": sum(item.accepted for item in holiday_impacts),
            "rejected": sum(not item.accepted for item in holiday_impacts),
        },
        "candidate_count": len(candidates),
        "selection_count": len(selections),
        "canonical_merge_count": sum(
            len(event.source_event_summary_ids) > 1 for event in events
        ),
        "canonical_split_source_count": sum(
            count > 1 for count in source_use_count.values()
        ),
        "holiday_event_count": sum(
            _is_holiday_family(event.event_family) for event in events
        ),
        "holiday_link_count": sum(
            links_by_event[event.event_id]
            for event in events
            if _is_holiday_family(event.event_family)
        ),
    }


def _dimension_coverage(
    strategy: CurationStrategy,
    series_semantics: list[SeriesSemanticDimensionsRecord],
    links_by_series: Counter,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dimension in strategy.projection_policy.allowed_dimensions:
        values = defaultdict(lambda: {"series": 0, "links": 0})
        for series in series_semantics:
            if dimension not in series.dimensions:
                continue
            raw = series.dimensions[dimension]
            for value in raw if isinstance(raw, list) else [raw]:
                label = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                values[label]["series"] += 1
                values[label]["links"] += links_by_series[series.series_id]
        output[dimension] = dict(sorted(values.items()))
    return output


def _review_hints(
    events: list[CanonicalEventRecord],
    duplicate_hints: list[dict[str, Any]],
    uncanonicalized_source_ids: list[str],
    rules: list[SemanticProjectionRule],
    series_ids: list[str],
    links: list[ExampleEventLink],
) -> dict[str, Any]:
    links_by_event = Counter(link.event_id for link in links)
    valid_years = {
        event.event_id: {
            str(event.event_start.year),
            str((event.event_end or event.event_start).year),
        }
        for event in events
    }
    return {
        "empty_summary_event_ids": [
            event.event_id for event in events if not event.summary.strip()
        ],
        "title_or_truncated_summary_event_ids": [
            event.event_id
            for event in events
            if len(event.summary.split()) < 8
            or not event.summary.endswith((".", "!", "?", "。", "！", "？"))
            or event.summary.endswith(("...", "…"))
        ],
        "date_summary_mismatch_event_ids": [
            event.event_id
            for event in events
            if (years := set(re.findall(r"\b(?:19|20)\d{2}\b", event.summary)))
            and not (years & valid_years[event.event_id])
        ],
        "target_milestone_event_ids": [
            event.event_id
            for event in events
            if re.search(
                r"\b(record high|record low|milestone|reached|surpassed|topped)\b",
                event.summary,
                flags=re.IGNORECASE,
            )
        ],
        "suspected_duplicate_events": duplicate_hints,
        "uncanonicalized_selected_source_event_ids": uncanonicalized_source_ids,
        "dataset_wide_event_ids": sorted(
            rule.event_id for rule in rules if rule.scope == "dataset"
        ),
        "zero_event_series_ids": sorted(
            series_id
            for series_id in series_ids
            if not any(link.series_id == series_id for link in links)
        ),
        "full_broadcast_event_ids": sorted(
            event.event_id
            for event in events
            if series_ids and links_by_event[event.event_id] == len(series_ids)
        ),
    }


def _render_quality_review(report: QualityReport, digest: str) -> str:
    gate_lines = "\n".join(
        f"- **{gate.gate_id}**: {gate.status} — {gate.impact}"
        for gate in [*report.hard_gates, *report.content_gates]
    )
    stats = report.statistics
    event_counts = stats.get("events_per_series", {})
    return f"""# Offline discovery quality review: {report.dataset_id}

Quality report SHA-256: `{digest}`

## Verdict

- Overall: **{report.overall_status}**
- Worth continuing search: {"yes" if report.worth_continuing_search else "no"}
- Recommendation: {report.recommendation}

## Scope

- Series: {stats.get("series_count", 0)}
- Canonical events: {stats.get("event_count", 0)}
- Event-series links: {stats.get("link_count", 0)}
- Events per series: min {event_counts.get("min", 0)}, mean {event_counts.get("mean", 0)}, max {event_counts.get("max", 0)}
- Zero-event series: {event_counts.get("zero_event_series_count", 0)}

## Gates

{gate_lines}

Detailed evidence and statistics: [`quality_report.json`](quality_report.json)

Choose `approve`, `revise_search`, `revise_events`, `revise_projection`, or `stop`.
"""


def _require_current_audit_inputs(
    run_dir: Path,
    report: QualityReport,
) -> None:
    if report.audit_input_sha256 != audit_input_sha256(run_dir):
        raise ValueError("Quality report is stale relative to current audit inputs")


def _require_revision_input_change(
    run_dir: Path,
    report: QualityReport,
) -> None:
    decision_path = run_dir / QUALITY_DECISION_FILENAME
    previous_report_path = run_dir / QUALITY_REPORT_FILENAME
    if not decision_path.exists() or not previous_report_path.exists():
        return
    decision = _read_model(decision_path, QualityReviewDecision)
    if decision.decision not in _REVISION_ROUTES:
        return
    previous = _read_model(previous_report_path, QualityReport)
    if previous.audit_input_sha256 == report.audit_input_sha256:
        raise ValueError(
            "Revision requires affected audit inputs to change before a new review"
        )


def _is_holiday_family(event_family: str) -> bool:
    lowered = event_family.casefold()
    return "holiday" in lowered or "observance" in lowered


def _read_model(path: Path, model):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}")
    return model.model_validate(value)


def _read_records(path: Path, model):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError(f"Expected one JSON array at {path}")
    return [model.model_validate(value) for value in values]


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_model(path: Path, model) -> None:
    _write_json(path, model.model_dump(mode="json"))

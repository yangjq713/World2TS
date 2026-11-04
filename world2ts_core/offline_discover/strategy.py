"""Strict Agent strategy validation, review, hashing, and approval freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from world2ts_core.offline_discover.holiday_calendar import country_holiday_dates

from .models import (
    CurationStrategy,
    DataReadinessReport,
    SemanticReadinessReport,
    StrategyReviewDecision,
)


DRAFT_FILENAME = "curation_strategy.draft.json"
REVIEW_FILENAME = "strategy_review.md"
DECISION_FILENAME = "strategy_review_decision.json"
APPROVED_FILENAME = "curation_strategy.approved.json"
USER_REVIEW_REQUIRED = "user_strategy_review_required"
USER_STOPPED = "user_stopped"

ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Expected one JSON object at {path}")
    try:
        return model_type.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid {model_type.__name__} at {path}: {exc}") from exc


def _write_model(path: Path, model: BaseModel) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def strategy_sha256(strategy: CurationStrategy) -> str:
    canonical = json.dumps(
        strategy.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_readiness(
    strategy: CurationStrategy,
    data_readiness: DataReadinessReport,
    semantic_readiness: SemanticReadinessReport,
) -> None:
    if data_readiness.status != "ready" or semantic_readiness.status != "ready":
        raise ValueError("Curation strategy requires ready data and Agent semantics")
    if strategy.dataset_id != data_readiness.dataset_id:
        raise ValueError(
            f"Strategy dataset_id {strategy.dataset_id!r} does not match "
            f"readiness dataset_id {data_readiness.dataset_id!r}"
        )
    if semantic_readiness.dataset_id != data_readiness.dataset_id:
        raise ValueError("Semantic and data readiness dataset IDs do not match")
    if semantic_readiness.expected_series_ids != data_readiness.series_ids:
        raise ValueError("Semantic readiness is not bound to the current data series order")

    available_dimensions = {profile.dimension for profile in data_readiness.dimensions}
    requested_dimensions = set(strategy.projection_policy.allowed_dimensions)
    unknown = sorted(requested_dimensions - available_dimensions)
    if unknown:
        raise ValueError(f"Strategy uses dimensions absent from readiness: {unknown}")
    for calendar in strategy.search_policy.holiday_calendars:
        unknown_series_ids = sorted(
            set(calendar.target_series_ids) - set(data_readiness.series_ids)
        )
        if unknown_series_ids:
            raise ValueError(
                "Holiday calendar references series absent from readiness: "
                + ", ".join(unknown_series_ids)
            )
        try:
            country_holiday_dates(
                calendar.country_code,
                calendar.subdivision_code,
                data_readiness.series_profiles[0].start.date(),
                data_readiness.series_profiles[0].start.date(),
            )
        except (KeyError, NotImplementedError) as exc:
            scope = "/".join(
                part
                for part in (
                    calendar.country_code,
                    calendar.subdivision_code,
                )
                if part
            )
            raise ValueError(f"Unsupported holiday calendar: {scope}") from exc


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _render_review(
    strategy: CurationStrategy,
    data_readiness: DataReadinessReport,
    digest: str,
) -> str:
    outlines = "\n".join(
        f"- **{outline.event_family}** ({outline.priority}): {outline.why_relevant}"
        for outline in strategy.search_intent_outline
    )
    periods = ", ".join(
        sorted({outline.time_window_period for outline in strategy.search_intent_outline})
    )
    dimensions = ", ".join(strategy.projection_policy.allowed_dimensions) or "none"
    holiday_calendars = ", ".join(
        (
            "/".join(
                part
                for part in (calendar.country_code, calendar.subdivision_code)
                if part
            )
            + f" [{calendar.timezone}] -> "
            + ", ".join(calendar.target_series_ids)
        )
        for calendar in strategy.search_policy.holiday_calendars
    )
    risk_flags = []
    if strategy.event_policy.allow_weak_context:
        risk_flags.append("weak contextual links are allowed")
    if strategy.event_policy.allow_forecasts:
        risk_flags.append("forecasts are allowed")
    if strategy.event_policy.allow_retrospective_context:
        risk_flags.append("retrospective context is allowed")
    if strategy.projection_policy.allow_dataset_wide:
        risk_flags.append("dataset-wide projection is allowed")
    if strategy.quality_policy.zero_event_series_allowed:
        risk_flags.append("zero-event series are acceptable")
    risks = "; ".join(risk_flags) or "no optional scope broadening is enabled"

    return f"""# Curation strategy review: {strategy.dataset_id}

Strategy SHA-256: `{digest}`

## Dataset understanding

{strategy.dataset_understanding}

Verified readiness covers {data_readiness.series_count} series and dimensions: {dimensions}.

## Event-family coverage

{outlines}

## Search plan

- Sources: {", ".join(strategy.search_policy.default_sources)}
- Structured holiday calendars: {holiday_calendars}
- Windows: {periods}; default {strategy.search_policy.time_window_period}
- Query budget: {strategy.search_policy.initial_queries_per_outline} initial per outline, {strategy.search_policy.max_refinement_queries_per_window} window-level refinements; top-k {strategy.search_policy.top_k}
- Stop policy: {strategy.search_policy.stop_policy}

## Event and projection policy

- Weak context: {_yes_no(strategy.event_policy.allow_weak_context)}; forecasts: {_yes_no(strategy.event_policy.allow_forecasts)}; retrospective context: {_yes_no(strategy.event_policy.allow_retrospective_context)}
- Target-only milestones excluded: {_yes_no(strategy.event_policy.exclude_target_only_milestones)}
- Canonical date order: {", ".join(strategy.event_policy.canonical_date_preference)}
- Approved projection dimensions: {dimensions}; dataset-wide projection: {_yes_no(strategy.projection_policy.allow_dataset_wide)}
- Ambiguous scope: {strategy.projection_policy.ambiguous_scope_action}
- Zero-event series allowed: {_yes_no(strategy.quality_policy.zero_event_series_allowed)}; final user approval required: yes

## Review focus

Main risk flags: {risks}.

Recommendation: approve only if these event families, budgets, and scope-broadening choices match the intended curation; otherwise request a revision.
"""


def prepare_strategy_review(
    *,
    strategy_file: Path,
    data_readiness_report_file: Path,
    semantic_readiness_report_file: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Validate an Agent draft and write its normalized review packet."""

    strategy = _load_model(strategy_file, CurationStrategy)
    data_readiness = _load_model(data_readiness_report_file, DataReadinessReport)
    semantic_readiness = _load_model(
        semantic_readiness_report_file, SemanticReadinessReport
    )
    _validate_readiness(strategy, data_readiness, semantic_readiness)
    digest = strategy_sha256(strategy)

    output_dir.mkdir(parents=True, exist_ok=True)
    draft_path = output_dir / DRAFT_FILENAME
    review_path = output_dir / REVIEW_FILENAME
    _write_model(draft_path, strategy)
    review_path.write_text(_render_review(strategy, data_readiness, digest), encoding="utf-8")
    return {
        "state": "awaiting_strategy_review",
        "blocked_reason": USER_REVIEW_REQUIRED,
        "strategy_sha256": digest,
        "draft_path": str(draft_path),
        "review_path": str(review_path),
    }


def apply_strategy_review(*, run_dir: Path, decision_file: Path) -> dict[str, str]:
    """Bind a user decision to the current draft and freeze approved content."""

    strategy = _load_model(run_dir / DRAFT_FILENAME, CurationStrategy)
    decision = _load_model(decision_file, StrategyReviewDecision)
    digest = strategy_sha256(strategy)
    if decision.reviewed_strategy_sha256 != digest:
        raise ValueError("Strategy review decision hash does not match the current draft")

    _write_model(run_dir / DECISION_FILENAME, decision)
    if decision.decision == "approve":
        _write_model(run_dir / APPROVED_FILENAME, strategy)
        return {
            "state": "strategy_approved",
            "blocked_reason": "",
            "strategy_sha256": digest,
        }
    if decision.decision == "revise":
        return {
            "state": "strategy_revision_required",
            "blocked_reason": USER_REVIEW_REQUIRED,
            "strategy_sha256": digest,
        }
    return {
        "state": "stopped",
        "blocked_reason": USER_STOPPED,
        "strategy_sha256": digest,
    }


def strategy_search_status(run_dir: Path) -> dict[str, str | bool]:
    """Return the hard gate that every search phase must check before adapters."""

    def blocked(digest: str, reason: str = USER_REVIEW_REQUIRED) -> dict[str, str | bool]:
        return {
            "approved": False,
            "can_search": False,
            "blocked_reason": reason,
            "next_agent_action": (
                "stop" if reason == USER_STOPPED else "request_strategy_review"
            ),
            "strategy_sha256": digest,
        }

    draft_path = run_dir / DRAFT_FILENAME
    if not draft_path.exists():
        return blocked("")

    strategy = _load_model(draft_path, CurationStrategy)
    digest = strategy_sha256(strategy)
    decision_path = run_dir / DECISION_FILENAME
    if not decision_path.exists():
        return blocked(digest)

    decision = _load_model(decision_path, StrategyReviewDecision)
    if decision.decision == "stop":
        return blocked(digest, USER_STOPPED)
    if decision.decision != "approve" or decision.reviewed_strategy_sha256 != digest:
        return blocked(digest)

    approved_path = run_dir / APPROVED_FILENAME
    if not approved_path.exists():
        return blocked(digest)
    approved = _load_model(approved_path, CurationStrategy)
    approved_matches = strategy_sha256(approved) == digest
    return {
        "approved": approved_matches,
        "can_search": approved_matches,
        "blocked_reason": "" if approved_matches else USER_REVIEW_REQUIRED,
        "next_agent_action": "" if approved_matches else "request_strategy_review",
        "strategy_sha256": digest,
    }

"""Strict v3 contracts for the Greenfield, time-only offline workflow."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .schema import DimensionMap, DimensionValue, normalize_dimension_map


RunStatus = Literal[
    "prepared",
    "awaiting_strategy_review",
    "strategy_revision_required",
    "strategy_approved",
    "searching",
    "curating",
    "projecting",
    "auditing",
    "awaiting_quality_review",
    "revision_required",
    "approved",
    "published",
    "failed",
    "stopped",
]
TimeWindowPeriod = Literal[
    "weekly",
    "biweekly",
    "monthly",
    "quarterly",
    "semiannual",
    "yearly",
]
SearchSourceName = Literal["tavily", "anysearch"]
GateStatus = Literal["pass", "warning", "fail"]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictArtifactModel(BaseModel):
    """Reject misspelled or stale fields in v3 artifacts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RunState(StrictArtifactModel):
    offline_discover_run_schema: Literal["3.3"] = "3.3"
    dataset_id: NonEmptyStr
    run_id: NonEmptyStr
    state: RunStatus
    updated_at: datetime
    reason: str = ""
    blocked_reason: str = ""
    next_agent_action: str = ""
    artifact_sha256: dict[str, str] = Field(default_factory=dict)

    @field_validator("artifact_sha256")
    @classmethod
    def validate_artifact_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            name: _validated_sha256(digest, f"artifact_sha256[{name!r}]")
            for name, digest in value.items()
        }


class LoadedExample(StrictArtifactModel):
    example_id: str
    start: datetime
    freq: str
    target: list[float] = Field(default_factory=list)
    timestamps: Optional[list[datetime]] = None


class SeriesReadinessProfile(StrictArtifactModel):
    series_id: str
    start: datetime
    end: datetime
    freq: str
    length: int = Field(ge=0)
    missing_rate: float = Field(ge=0.0, le=1.0)
    has_explicit_timestamps: bool

    @model_validator(mode="after")
    def validate_order(self) -> "SeriesReadinessProfile":
        if self.end < self.start:
            raise ValueError("series end must not be before start")
        return self


class SeriesSemanticDimensionsRecord(StrictArtifactModel):
    series_id: NonEmptyStr
    entity_label: str = ""
    description: str = ""
    location: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    dimensions: DimensionMap = Field(default_factory=dict)

    @field_validator("dimensions", mode="before")
    @classmethod
    def normalize_dimensions(cls, value: object) -> object:
        return normalize_dimension_map(value)


class MetadataConsistencyCheck(StrictArtifactModel):
    check_id: str
    status: Literal["pass", "fail"]
    evidence: str


class DimensionReadinessProfile(StrictArtifactModel):
    dimension: str
    value_type: Literal[
        "string",
        "integer",
        "number",
        "boolean",
        "string_list",
        "integer_list",
        "number_list",
        "boolean_list",
        "mixed",
    ]
    populated_series_count: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    value_counts: dict[str, int] = Field(default_factory=dict)


class DataReadinessReport(StrictArtifactModel):
    schema_version: Literal["1.1"] = "1.1"
    dataset_id: NonEmptyStr
    status: Literal["ready", "blocked"]
    arrow_files: list[str] = Field(min_length=1)
    series_count: int = Field(ge=0)
    series_ids: list[str] = Field(default_factory=list)
    series_id_order_sha256: str
    series_profiles: list[SeriesReadinessProfile] = Field(default_factory=list)
    metadata_consistency: list[MetadataConsistencyCheck] = Field(default_factory=list)
    dataset_semantics_coverage: float = Field(ge=0.0, le=1.0)
    series_semantics_coverage: float = Field(ge=0.0, le=1.0)
    dimensions: list[DimensionReadinessProfile] = Field(default_factory=list)
    unrecoverable_series_ids: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)

    @field_validator("series_id_order_sha256")
    @classmethod
    def validate_order_hash(cls, value: str) -> str:
        return _validated_sha256(value, "series_id_order_sha256")

    @model_validator(mode="after")
    def validate_series_inventory(self) -> "DataReadinessReport":
        if self.series_count != len(self.series_ids):
            raise ValueError("series_count must equal the number of series_ids")
        if len(set(self.series_ids)) != len(self.series_ids):
            raise ValueError("series_ids must be unique")
        profile_ids = [profile.series_id for profile in self.series_profiles]
        if profile_ids != self.series_ids:
            raise ValueError("series_profiles must follow the physical series_ids order")
        if self.status == "blocked" and not self.blocking_reasons:
            raise ValueError("blocked readiness requires at least one blocking reason")
        if self.status == "ready" and self.blocking_reasons:
            raise ValueError("ready readiness must not include blocking reasons")
        return self


class SemanticReadinessReport(StrictArtifactModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: NonEmptyStr
    status: Literal["ready", "blocked"]
    expected_series_ids: list[NonEmptyStr] = Field(default_factory=list)
    validated_series_ids: list[NonEmptyStr] = Field(default_factory=list)
    checks: list[MetadataConsistencyCheck] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> "SemanticReadinessReport":
        if self.status == "blocked" and not self.blocking_reasons:
            raise ValueError("blocked semantic readiness requires a reason")
        if self.status == "ready":
            if self.blocking_reasons:
                raise ValueError("ready semantic readiness must not include blocking reasons")
            if self.validated_series_ids != self.expected_series_ids:
                raise ValueError("ready semantic readiness requires exact ordered series coverage")
        return self


class EventPolicy(StrictArtifactModel):
    allow_weak_context: bool
    allow_retrospective_context: bool
    allow_forecasts: bool
    exclude_target_only_milestones: bool
    canonical_date_preference: list[
        Literal["actual_start", "effective_date", "announcement_date", "publication_date"]
    ] = Field(min_length=1)
    duplicate_policy: Literal["merge_same_episode"]
    compound_record_policy: Literal["split_distinct_episodes"]


class HolidayCalendarSpec(StrictArtifactModel):
    country_code: NonEmptyStr
    subdivision_code: str = ""
    timezone: NonEmptyStr
    target_series_ids: list[NonEmptyStr] = Field(min_length=1)

    @field_validator("country_code")
    @classmethod
    def normalize_country_code(cls, value: str) -> str:
        return value.upper()

    @field_validator("subdivision_code")
    @classmethod
    def normalize_subdivision_code(cls, value: str) -> str:
        return value.upper()

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value!r}") from exc
        return value

    @field_validator("target_series_ids")
    @classmethod
    def validate_target_series_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("holiday calendar target_series_ids must be unique")
        return value


class SearchPolicy(StrictArtifactModel):
    default_sources: list[SearchSourceName] = Field(min_length=1)
    top_k: int = Field(ge=1)
    initial_queries_per_outline: int = Field(ge=1)
    max_refinement_queries_per_window: int = Field(ge=1)
    time_window_period: TimeWindowPeriod
    stop_policy: NonEmptyStr
    holiday_calendars: list[HolidayCalendarSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources_and_calendars(self) -> "SearchPolicy":
        if len(set(self.default_sources)) != len(self.default_sources):
            raise ValueError("default_sources must not contain duplicates")
        calendars = [
            (
                item.country_code,
                item.subdivision_code,
                item.timezone,
                tuple(sorted(item.target_series_ids)),
            )
            for item in self.holiday_calendars
        ]
        if len(set(calendars)) != len(calendars):
            raise ValueError("holiday_calendars must not contain duplicates")
        return self


class SearchIntentOutline(StrictArtifactModel):
    outline_id: NonEmptyStr
    event_family: NonEmptyStr
    priority: Literal["high", "medium", "low"]
    why_relevant: NonEmptyStr
    context_pathways: list[NonEmptyStr] = Field(min_length=1)
    target_dimensions: list[NonEmptyStr] = Field(default_factory=list)
    time_window_period: TimeWindowPeriod
    initial_query_roles: list[NonEmptyStr] = Field(min_length=1)
    in_scope_examples: list[NonEmptyStr] = Field(default_factory=list)
    out_of_scope_examples: list[NonEmptyStr] = Field(default_factory=list)
    completion_evidence: NonEmptyStr


class ProjectionPolicy(StrictArtifactModel):
    allowed_dimensions: list[NonEmptyStr] = Field(default_factory=list)
    allow_dataset_wide: bool
    ambiguous_scope_action: Literal["hold_for_agent_review"]
    link_time_policy: Literal["event_must_overlap_series_coverage"]


class QualityPolicy(StrictArtifactModel):
    zero_event_series_allowed: bool
    holiday_share_is_diagnostic_only: bool
    require_user_final_approval: Literal[True]


class CurationStrategy(StrictArtifactModel):
    schema_version: Literal["2.0"]
    dataset_id: NonEmptyStr
    greenfield: Literal[True]
    dataset_understanding: NonEmptyStr
    event_policy: EventPolicy
    search_policy: SearchPolicy
    search_intent_outline: list[SearchIntentOutline] = Field(min_length=1)
    projection_policy: ProjectionPolicy
    quality_policy: QualityPolicy

    @model_validator(mode="after")
    def validate_outline_identity_and_dimensions(self) -> "CurationStrategy":
        outline_ids = [outline.outline_id for outline in self.search_intent_outline]
        if len(set(outline_ids)) != len(outline_ids):
            raise ValueError("search intent outline IDs must be unique")
        if len(set(self.projection_policy.allowed_dimensions)) != len(
            self.projection_policy.allowed_dimensions
        ):
            raise ValueError("allowed projection dimensions must be unique")
        for outline in self.search_intent_outline:
            if len(set(outline.target_dimensions)) != len(outline.target_dimensions):
                raise ValueError(f"outline {outline.outline_id!r} repeats target dimensions")
            if len(set(outline.initial_query_roles)) != len(outline.initial_query_roles):
                raise ValueError(f"outline {outline.outline_id!r} repeats initial query roles")
        allowed = set(self.projection_policy.allowed_dimensions)
        unknown = sorted(
            {
                dimension
                for outline in self.search_intent_outline
                for dimension in outline.target_dimensions
                if dimension not in allowed
            }
        )
        if unknown:
            raise ValueError(f"outline target dimensions are not approved: {unknown}")
        return self


class StrategyReviewDecision(StrictArtifactModel):
    decision: Literal["approve", "revise", "stop"]
    reviewed_strategy_sha256: str
    user_instructions: str = ""
    reviewed_at: datetime

    @field_validator("reviewed_strategy_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validated_sha256(value, "reviewed_strategy_sha256")


class TimeWindowRecord(StrictArtifactModel):
    window_id: NonEmptyStr
    period: TimeWindowPeriod
    start: datetime
    end: datetime
    active_series_ids: list[NonEmptyStr] = Field(min_length=1)
    active_dimension_values: DimensionMap = Field(default_factory=dict)
    outline_ids: list[NonEmptyStr] = Field(min_length=1)
    priority: float = Field(default=0.0, ge=0.0)

    @field_validator("active_dimension_values", mode="before")
    @classmethod
    def normalize_dimensions(cls, value: object) -> object:
        return normalize_dimension_map(value)

    @model_validator(mode="after")
    def validate_time_window(self) -> "TimeWindowRecord":
        if self.end < self.start:
            raise ValueError("time window end must not be before start")
        if len(set(self.active_series_ids)) != len(self.active_series_ids):
            raise ValueError("active_series_ids must be unique")
        if len(set(self.outline_ids)) != len(self.outline_ids):
            raise ValueError("outline_ids must be unique")
        return self


class AgentPriorEventHypothesis(StrictArtifactModel):
    hypothesis_id: NonEmptyStr
    event_name: NonEmptyStr
    date_hint: NonEmptyStr
    relevance_pathway: NonEmptyStr
    reasoning: NonEmptyStr
    confidence: float = Field(ge=0.0, le=1.0)


class AgentWindowSearchPlanRecord(StrictArtifactModel):
    window_id: NonEmptyStr
    prior_event_hypotheses: list[AgentPriorEventHypothesis] = Field(min_length=1)
    generated_by: Literal["agent_active_review"]

    @model_validator(mode="after")
    def validate_hypothesis_ids(self) -> "AgentWindowSearchPlanRecord":
        ids = [item.hypothesis_id for item in self.prior_event_hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError("prior event hypothesis IDs must be unique within a window")
        return self


class AgentQueryIntentRecord(StrictArtifactModel):
    query_intent_id: NonEmptyStr
    window_id: NonEmptyStr
    stage: Literal["initial", "window_refinement"]
    outline_id: Optional[NonEmptyStr] = None
    query_text: NonEmptyStr
    search_start: datetime
    search_end: datetime
    top_k: int = Field(default=10, ge=1)
    context_hypothesis: NonEmptyStr
    relevance_pathway: NonEmptyStr
    source_names: Optional[list[SearchSourceName]] = None
    feedback_query_intent_ids: list[NonEmptyStr] = Field(default_factory=list)
    prior_event_hypothesis_ids: list[NonEmptyStr] = Field(default_factory=list)
    refinement_reason: str = ""
    failed_query_feedback: list[str] = Field(default_factory=list)
    generated_by: Literal["agent_active_review"]

    @model_validator(mode="after")
    def validate_query_lineage(self) -> "AgentQueryIntentRecord":
        if self.search_end < self.search_start:
            raise ValueError("search_end must not be before search_start")
        is_refinement = self.stage == "window_refinement"
        if is_refinement:
            if self.outline_id is not None:
                raise ValueError("window refinements must not be restricted to an outline")
            if (
                not self.feedback_query_intent_ids
                or not self.refinement_reason
                or not self.failed_query_feedback
            ):
                raise ValueError(
                    "window refinements require query feedback, a reason, and failed-query feedback"
                )
        elif self.outline_id is None:
            raise ValueError("initial queries require outline_id")
        elif (
            self.feedback_query_intent_ids
            or self.prior_event_hypothesis_ids
            or self.refinement_reason
            or self.failed_query_feedback
        ):
            raise ValueError("initial queries must not contain refinement feedback")
        if len(set(self.feedback_query_intent_ids)) != len(
            self.feedback_query_intent_ids
        ):
            raise ValueError("feedback query intent IDs must not contain duplicates")
        if len(set(self.prior_event_hypothesis_ids)) != len(
            self.prior_event_hypothesis_ids
        ):
            raise ValueError("prior event hypothesis IDs must not contain duplicates")
        if self.source_names is not None and len(set(self.source_names)) != len(self.source_names):
            raise ValueError("source_names must not contain duplicates")
        return self


class SearchWindowStateRecord(StrictArtifactModel):
    window_id: NonEmptyStr
    status: Literal[
        "pending",
        "awaiting_selection",
        "awaiting_refinement",
        "covered",
        "exhausted",
        "no_event_found",
        "failed",
        "stopped",
    ] = "pending"
    attempted_initial_query_intent_ids: list[NonEmptyStr] = Field(default_factory=list)
    attempted_refinement_query_intent_ids: list[NonEmptyStr] = Field(
        default_factory=list
    )
    normalized_queries: list[NonEmptyStr] = Field(default_factory=list)
    candidate_event_summary_ids: list[NonEmptyStr] = Field(default_factory=list)
    reviewed_candidate_event_summary_ids: list[NonEmptyStr] = Field(
        default_factory=list
    )
    failure_reasons: list[NonEmptyStr] = Field(default_factory=list)
    holiday_analysis_complete: bool = False
    next_agent_action: str = ""

    @model_validator(mode="after")
    def validate_search_state(self) -> "SearchWindowStateRecord":
        attempted = [
            *self.attempted_initial_query_intent_ids,
            *self.attempted_refinement_query_intent_ids,
        ]
        if len(set(attempted)) != len(attempted):
            raise ValueError("attempted query intent IDs must be unique")
        if len(set(self.normalized_queries)) != len(self.normalized_queries):
            raise ValueError("normalized queries must be unique")
        if not set(self.reviewed_candidate_event_summary_ids) <= set(
            self.candidate_event_summary_ids
        ):
            raise ValueError("reviewed candidates must belong to the search window")
        return self


class SearchTraceRecord(StrictArtifactModel):
    trace_id: NonEmptyStr
    window_id: NonEmptyStr
    query_intent_id: NonEmptyStr
    stage: Literal["initial", "window_refinement"]
    outline_id: Optional[NonEmptyStr] = None
    query_text: NonEmptyStr
    source: SearchSourceName
    started_at: datetime
    elapsed_ms: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    failure: str = ""
    event_summary_ids: list[str] = Field(default_factory=list)
    selection_decision: str = ""
    rejected_rationales: dict[str, str] = Field(default_factory=dict)
    feedback_query_intent_ids: list[str] = Field(default_factory=list)


class HolidaySeriesImpact(StrictArtifactModel):
    series_id: NonEmptyStr
    event_count: int = Field(ge=0)
    baseline_count: int = Field(ge=0)
    event_mean: Optional[float] = None
    baseline_median: Optional[float] = None
    pct_difference: Optional[float] = None
    robust_z: Optional[float] = None
    supported: bool


class HolidayImpactRecord(StrictArtifactModel):
    impact_id: NonEmptyStr
    window_id: NonEmptyStr
    country_code: NonEmptyStr
    subdivision_code: str = ""
    timezone: NonEmptyStr
    holiday_date: date
    holiday_name: NonEmptyStr
    accepted: bool
    affected_series_ids: list[NonEmptyStr] = Field(default_factory=list)
    series_impacts: list[HolidaySeriesImpact] = Field(default_factory=list)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value!r}") from exc
        return value

    @model_validator(mode="after")
    def validate_affected_series(self) -> "HolidayImpactRecord":
        supported = [
            item.series_id for item in self.series_impacts if item.supported
        ]
        if self.affected_series_ids != supported:
            raise ValueError(
                "affected_series_ids must equal supported series impact order"
            )
        if self.accepted != bool(supported):
            raise ValueError("accepted holiday impacts require supported series")
        return self


class AgentEventSelectionRecord(StrictArtifactModel):
    """Agent judgment over retrieved candidates only."""

    window_id: NonEmptyStr
    decision: Literal["accept", "reject", "continue_search", "complete"]
    selected_event_summary_ids: list[str] = Field(default_factory=list)
    rejected_event_summary_ids: list[str] = Field(default_factory=list)
    rejected_rationales: dict[str, str] = Field(default_factory=dict)
    supporting_evidence: list[str] = Field(default_factory=list)
    completion_evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    selected_by: Literal["agent_active_review"]

    @model_validator(mode="after")
    def validate_candidate_decision(self) -> "AgentEventSelectionRecord":
        selected = set(self.selected_event_summary_ids)
        rejected = set(self.rejected_event_summary_ids)
        if len(selected) != len(self.selected_event_summary_ids):
            raise ValueError("selected candidate IDs must be unique")
        if len(rejected) != len(self.rejected_event_summary_ids):
            raise ValueError("rejected candidate IDs must be unique")
        if selected & rejected:
            raise ValueError("a candidate cannot be both selected and rejected")
        if self.decision == "accept" and not selected:
            raise ValueError("accept decisions require selected candidates")
        if selected and self.decision not in {
            "accept",
            "continue_search",
            "complete",
        }:
            raise ValueError(
                "selected candidates are invalid for reject decisions"
            )
        if set(self.rejected_rationales) - rejected:
            raise ValueError("rejected rationales must reference rejected candidates")
        return self


class CanonicalEventRecord(StrictArtifactModel):
    event_id: NonEmptyStr
    event_start: datetime
    event_end: Optional[datetime] = None
    summary: NonEmptyStr
    event_family: NonEmptyStr
    context_role: Literal["actual", "forecast", "decision", "retrospective", "concurrent"]
    affected_scope_claim: NonEmptyStr
    source_event_summary_ids: list[NonEmptyStr] = Field(min_length=1)
    canonical_date_basis: NonEmptyStr
    merge_or_split_notes: NonEmptyStr
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_event_dates(self) -> "CanonicalEventRecord":
        if self.event_end is not None and self.event_end < self.event_start:
            raise ValueError("event_end must not be before event_start")
        if len(set(self.source_event_summary_ids)) != len(self.source_event_summary_ids):
            raise ValueError("source_event_summary_ids must be unique")
        return self


class SemanticProjectionFilter(StrictArtifactModel):
    dimension: NonEmptyStr
    operator: Literal["eq", "in", "between", "exists"]
    value: Optional[DimensionValue] = None

    @model_validator(mode="after")
    def validate_operator_value(self) -> "SemanticProjectionFilter":
        if self.operator == "exists" and self.value is not None:
            raise ValueError("exists filters must omit value")
        if self.operator != "exists" and self.value is None:
            raise ValueError(f"{self.operator} filters require value")
        if self.operator == "in" and not isinstance(self.value, list):
            raise ValueError("in filters require a list value")
        if self.operator == "eq" and isinstance(self.value, list):
            raise ValueError("eq filters require a scalar value")
        if self.operator == "between":
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError("between filters require exactly two values")
            if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in self.value):
                raise ValueError("between filters require numeric values")
        return self


class SemanticProjectionRule(StrictArtifactModel):
    rule_id: NonEmptyStr
    event_id: NonEmptyStr
    scope: Literal["dataset", "dimension_filter"]
    filters: list[SemanticProjectionFilter] = Field(default_factory=list)
    combine: Literal["all", "any"] = "all"
    basis: NonEmptyStr
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_scope(self) -> "SemanticProjectionRule":
        if self.scope == "dataset" and self.filters:
            raise ValueError("dataset-wide rules must not include filters")
        if self.scope == "dimension_filter" and not self.filters:
            raise ValueError("dimension-filter rules require filters")
        return self


class ExampleEventLink(StrictArtifactModel):
    event_id: NonEmptyStr
    series_id: NonEmptyStr
    projection_rule_id: NonEmptyStr
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class QualityGateResult(StrictArtifactModel):
    gate_id: NonEmptyStr
    gate_type: Literal["hard", "content"]
    status: GateStatus
    evidence: list[NonEmptyStr] = Field(min_length=1)
    impact: NonEmptyStr
    continue_search_recommended: bool
    recommended_return_phase: Optional[Literal["search", "canonicalization", "projection"]] = None


class QualityReport(StrictArtifactModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: NonEmptyStr
    generated_at: datetime
    audit_input_sha256: str
    overall_status: GateStatus
    hard_gates: list[QualityGateResult] = Field(default_factory=list)
    content_gates: list[QualityGateResult] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    worth_continuing_search: bool
    recommendation: NonEmptyStr
    recommended_return_phase: Optional[Literal["search", "canonicalization", "projection"]] = None

    @field_validator("audit_input_sha256")
    @classmethod
    def validate_audit_hash(cls, value: str) -> str:
        return _validated_sha256(value, "audit_input_sha256")

    @model_validator(mode="after")
    def validate_gate_partition(self) -> "QualityReport":
        if any(gate.gate_type != "hard" for gate in self.hard_gates):
            raise ValueError("hard_gates may only contain hard gate results")
        if any(gate.gate_type != "content" for gate in self.content_gates):
            raise ValueError("content_gates may only contain content gate results")
        gates = [*self.hard_gates, *self.content_gates]
        gate_ids = [gate.gate_id for gate in gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("quality gate IDs must be unique")
        expected_status = (
            "fail"
            if any(gate.status == "fail" for gate in gates)
            else "warning"
            if any(gate.status == "warning" for gate in gates)
            else "pass"
        )
        if self.overall_status != expected_status:
            raise ValueError(
                f"overall_status must reflect gate statuses: {expected_status}"
            )
        return self


class QualityReviewDecision(StrictArtifactModel):
    decision: Literal[
        "approve", "revise_search", "revise_events", "revise_projection", "stop"
    ]
    reviewed_quality_report_sha256: str
    user_instructions: str = ""
    target_window_ids: list[str] = Field(default_factory=list)
    reviewed_at: datetime

    @field_validator("reviewed_quality_report_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validated_sha256(value, "reviewed_quality_report_sha256")

    @model_validator(mode="after")
    def validate_revision_target(self) -> "QualityReviewDecision":
        if self.decision == "revise_search" and not self.target_window_ids:
            raise ValueError("revise_search requires target_window_ids")
        if self.decision != "revise_search" and self.target_window_ids:
            raise ValueError("target_window_ids are only valid for revise_search")
        return self


def _validated_sha256(value: str, field_name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return normalized

"""Strict input schema for offline-discover dataset metadata.

Original LOTSA metadata serializes per-series semantics as ``attributes``.
The parsed model exposes normalized ``dimensions`` internally so readiness,
strategy, and projection share one semantic surface without changing the wire
contract.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)


class StrictSchemaModel(BaseModel):
    """Base class for fail-loud metadata input validation."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


DimensionScalar = Union[StrictStr, StrictInt, StrictFloat, StrictBool]


def _normalize_dimension_scalar(value: object) -> object:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("dimension string values must not be empty")
        return normalized
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("dimension numeric values must be finite")
        return value
    raise ValueError(
        "dimension values must be a string, integer, finite number, boolean, "
        "or a non-empty homogeneous list of those scalar types"
    )


def normalize_dimension_value(value: object) -> object:
    """Normalize one dimension value without silently coercing its type.

    Strings are stripped, list order is retained, and duplicate list members
    are removed. Numeric lists may mix integers and floats and are normalized
    to floats; all other lists must contain a single scalar type.
    """

    if not isinstance(value, list):
        return _normalize_dimension_scalar(value)
    if not value:
        raise ValueError("dimension list values must not be empty")

    normalized = [_normalize_dimension_scalar(item) for item in value]
    kinds = {
        "bool"
        if isinstance(item, bool)
        else "int"
        if isinstance(item, int)
        else "float"
        if isinstance(item, float)
        else "str"
        for item in normalized
    }
    if len(kinds) > 1:
        if kinds <= {"int", "float"}:
            normalized = [float(item) for item in normalized]
        else:
            raise ValueError("dimension list values must have one scalar type")

    unique: list[object] = []
    seen: set[tuple[type[object], object]] = set()
    for item in normalized:
        key = (type(item), item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


DimensionValue = Annotated[
    Union[DimensionScalar, list[DimensionScalar]],
    BeforeValidator(normalize_dimension_value),
]
DimensionMap = dict[StrictStr, DimensionValue]


def normalize_dimension_map(value: object) -> object:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("dimensions must be an object keyed by dimension name")

    normalized: dict[str, object] = {}
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str):
            raise ValueError("dimension names must be strings")
        name = raw_name.strip()
        if not name:
            raise ValueError("dimension names must not be empty")
        if name in normalized:
            raise ValueError(f"duplicate dimension name after normalization: {name!r}")
        normalized[name] = raw_value
    return normalized


class TimeRange(StrictSchemaModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_order(self) -> "TimeRange":
        if self.end < self.start:
            raise ValueError("time_range.end must not be before time_range.start")
        return self


class DatasetSemantics(StrictSchemaModel):
    dataset_name: str
    category: str
    subcategory: str
    geo: str
    providers: list[str] = Field(default_factory=list)
    time_granularity: str
    time_range: TimeRange
    num_series: int = Field(ge=0)
    description_short: str
    description_long: str
    data_quality: dict[str, str] = Field(default_factory=dict, alias="_data_quality")


class MetricField(StrictSchemaModel):
    name: str
    description: str = ""
    unit: str = ""


class ItemSemanticsEntry(StrictSchemaModel):
    entity_label: str
    description: str = ""
    location: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dimensions: DimensionMap = Field(
        default_factory=dict,
        serialization_alias="attributes",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_attributes_alias(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "attributes" in payload:
            if "dimensions" in payload:
                raise ValueError("use dimensions or attributes, not both")
            payload["dimensions"] = payload.pop("attributes")
        return payload

    @field_validator("dimensions", mode="before")
    @classmethod
    def normalize_dimensions(cls, value: object) -> object:
        return normalize_dimension_map(value)


class ItemSchema(StrictSchemaModel):
    entity_type: str
    item_id_pattern: str
    spatial_dimension: str
    target: list[MetricField] = Field(default_factory=list)
    past_feat_dynamic_real: list[MetricField] = Field(default_factory=list)
    static_features: list[MetricField] = Field(default_factory=list)
    item_semantics: dict[str, ItemSemanticsEntry] = Field(default_factory=dict)


class ItemGroup(StrictSchemaModel):
    group_id: str
    label: str = ""
    example_ids: list[str] = Field(default_factory=list)
    shared_terms: list[str] = Field(default_factory=list)


class QueryHints(StrictSchemaModel):
    dataset_terms: list[str] = Field(default_factory=list)
    event_bias: list[str] = Field(default_factory=list)


class OfflineMetadata(StrictSchemaModel):
    schema_version: Literal["2.0"]
    dataset_semantics: DatasetSemantics
    item_schema: ItemSchema
    item_groups: list[ItemGroup] = Field(default_factory=list)
    query_hints: QueryHints = Field(default_factory=QueryHints)


def parse_offline_metadata(raw: object, path: str | Path) -> OfflineMetadata:
    try:
        return OfflineMetadata.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid OfflineMetadata at {path}: {exc}") from exc

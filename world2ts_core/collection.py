"""Strict online collection into the canonical offline input pair."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, get_args
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyarrow as pa
import pyarrow.ipc as ipc
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from world2ts_core.bootstrap import ensure_sources_registered
from world2ts_core.freq_routing import build_adapter, constructor_options_for_source
from world2ts_core.offline_discover.readiness import prepare_data_readiness
from world2ts_core.offline_discover.schema import (
    DatasetSemantics,
    ItemSchema,
    ItemSemanticsEntry,
    MetricField,
    OfflineMetadata,
    QueryHints,
    TimeRange,
    normalize_dimension_value,
)
from world2ts_core.registry import get_series_source_class


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CollectionFrequency = Literal["15min", "1h", "1d", "1w"]
COLLECTION_FREQUENCIES = frozenset(get_args(CollectionFrequency))
_RESERVED_DIMENSIONS = {
    "domain",
    "frequency",
    "metric_type",
    "provider_name",
    "source_name",
    "source_series_id",
    "target_description",
    "target_name",
    "target_unit",
}


class _StrictPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CollectionTargetPlan(_StrictPlanModel):
    source_field: NonEmptyStr
    name: NonEmptyStr
    description: NonEmptyStr
    unit: str

    @field_validator("source_field")
    @classmethod
    def normalize_source_field(cls, value: str) -> str:
        return value.casefold()


class CollectionQueryHintsPlan(_StrictPlanModel):
    dataset_terms: list[NonEmptyStr]
    event_bias: list[NonEmptyStr]

    @field_validator("dataset_terms", "event_bias")
    @classmethod
    def validate_unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("query hint values must be unique")
        return value


class CollectionMetadataPlan(_StrictPlanModel):
    subcategory: NonEmptyStr
    geo: NonEmptyStr
    description_short: NonEmptyStr
    description_long: NonEmptyStr
    entity_type: NonEmptyStr
    spatial_dimension: NonEmptyStr
    target: CollectionTargetPlan
    dimensions: list[NonEmptyStr]
    query_hints: CollectionQueryHintsPlan

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("metadata dimensions must be unique")
        reserved = sorted(set(value) & _RESERVED_DIMENSIONS)
        if reserved:
            raise ValueError(
                "metadata dimensions collide with canonical dimensions: "
                f"{reserved}"
            )
        return value


class CollectionPlan(_StrictPlanModel):
    schema_version: Literal["1.0"]
    dataset_id: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        ),
    ]
    source: NonEmptyStr
    domain: NonEmptyStr
    symbols: list[NonEmptyStr] = Field(min_length=1)
    start: datetime
    end: datetime
    frequency: CollectionFrequency
    timezone: NonEmptyStr
    adapter_options: dict[NonEmptyStr, Any]
    metadata: CollectionMetadataPlan

    @field_validator("start", "end", mode="before")
    @classmethod
    def parse_date_or_datetime(cls, value: object) -> object:
        if isinstance(value, str) and len(value) == 10:
            try:
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                return value
        return value

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("symbols must be unique")
        return value

    @model_validator(mode="after")
    def validate_window_and_timezone(self) -> "CollectionPlan":
        if self.start.tzinfo is not None or self.end.tzinfo is not None:
            raise ValueError(
                "start and end must not include a UTC offset; interpret them with timezone"
            )
        if self.end <= self.start:
            raise ValueError("end must be after start")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {self.timezone!r}") from exc
        try:
            json.dumps(self.adapter_options, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("adapter_options must contain finite JSON values") from exc
        return self


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_collection_plan(path: Path) -> CollectionPlan:
    """Load one strict Agent-authored collection plan."""

    plan_path = Path(path).expanduser()
    try:
        raw = json.loads(
            plan_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except OSError as exc:
        raise ValueError(f"Unable to read collection plan {plan_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in collection plan {plan_path}: {exc}") from exc

    try:
        return CollectionPlan.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid collection plan at {plan_path}: {exc}") from exc


def _utc_timestamp(value: object, *, timezone_name: str, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{label} must be a datetime")
    try:
        localized = (
            value.replace(tzinfo=ZoneInfo(timezone_name))
            if value.tzinfo is None
            else value
        )
        return localized.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid timestamp: {value!r}") from exc


def _build_exact_adapter(plan: CollectionPlan):
    ensure_sources_registered()
    source_class = get_series_source_class(plan.source)
    if source_class is None:
        raise ValueError(f"Unknown series source: {plan.source!r}")

    capability = source_class.capability
    if plan.frequency not in capability.frequencies:
        raise ValueError(
            f"Source {plan.source!r} does not declare frequency {plan.frequency!r}; "
            f"supported frequencies: {', '.join(capability.frequencies)}"
        )
    if plan.source in {"coingecko", "owid", "polymarket"} and plan.timezone != "UTC":
        raise ValueError(
            f"Source {plan.source!r} emits UTC timestamps and requires plan timezone 'UTC'"
        )

    allowed_option_names = set(capability.adapter_options)
    unexpected_options = sorted(set(plan.adapter_options) - allowed_option_names)
    if unexpected_options:
        raise ValueError(
            f"Source {plan.source!r} does not accept Agent adapter_options: "
            + ", ".join(unexpected_options)
        )
    for option_name, allowed_values in capability.adapter_options.items():
        if option_name not in plan.adapter_options:
            if option_name == capability.target_option:
                raise ValueError(
                    f"Source {plan.source!r} requires explicit adapter_options.{option_name}"
                )
            continue
        if plan.adapter_options[option_name] not in allowed_values:
            raise ValueError(
                f"Source {plan.source!r} adapter_options.{option_name} must be one of "
                f"{allowed_values}"
            )

    if capability.target_option:
        expected_source_field = str(
            plan.adapter_options[capability.target_option]
        ).casefold()
    elif plan.source == "eia":
        expected_fields = {
            source_class.primary_target_field(symbol)
            for symbol in plan.symbols
        }
        if len(expected_fields) != 1:
            raise ValueError(
                "One collection dataset cannot mix EIA symbols with different "
                f"primary target fields: {sorted(expected_fields)}"
            )
        expected_source_field = next(iter(expected_fields))
    elif len(capability.primary_target_fields) == 1:
        expected_source_field = capability.primary_target_fields[0]
    else:
        raise ValueError(
            f"Source {plan.source!r} has an ambiguous primary target contract"
        )
    if plan.metadata.target.source_field != expected_source_field.casefold():
        raise ValueError(
            f"metadata.target.source_field must be {expected_source_field!r} for "
            f"source {plan.source!r}"
        )

    unsupported_dimensions = sorted(
        set(plan.metadata.dimensions) - set(capability.metadata_dimensions)
    )
    if unsupported_dimensions:
        raise ValueError(
            f"Source {plan.source!r} does not advertise metadata dimensions: "
            + ", ".join(unsupported_dimensions)
        )

    routed_options = constructor_options_for_source(plan.source, plan.frequency)
    conflicts = {
        key: (routed_options[key], plan.adapter_options[key])
        for key in set(routed_options) & set(plan.adapter_options)
        if routed_options[key] != plan.adapter_options[key]
    }
    if conflicts:
        raise ValueError(
            "adapter_options conflict with the requested frequency: "
            f"{conflicts}"
        )

    options = dict(plan.adapter_options)
    options.update(routed_options)
    try:
        return build_adapter(plan.source, options)
    except TypeError as exc:
        raise ValueError(
            f"Invalid adapter_options for source {plan.source!r}: {exc}"
        ) from exc


def _validated_records(plan: CollectionPlan, records: object) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("Series adapter must return a list of SeriesRecord values")

    actual_ids: list[str] = []
    by_id: dict[str, Any] = {}
    for record in records:
        series_id = getattr(record, "series_id", None)
        if not isinstance(series_id, str) or not series_id:
            raise ValueError("Series adapter returned a record without a non-empty series_id")
        if series_id in by_id:
            raise ValueError(f"Series adapter returned duplicate series_id: {series_id!r}")
        actual_ids.append(series_id)
        by_id[series_id] = record

    requested = set(plan.symbols)
    actual = set(actual_ids)
    if requested != actual or len(records) != len(plan.symbols):
        missing = sorted(requested - actual)
        unexpected = sorted(actual - requested)
        raise ValueError(
            "Series adapter must return exactly one record per requested symbol: "
            f"missing={missing} unexpected={unexpected}"
        )

    timezone_name = plan.timezone
    requested_start = _utc_timestamp(
        plan.start,
        timezone_name=timezone_name,
        label="plan.start",
    )
    requested_end = _utc_timestamp(
        plan.end,
        timezone_name=timezone_name,
        label="plan.end",
    )
    validated: list[dict[str, Any]] = []

    for symbol in plan.symbols:
        record = by_id[symbol]
        if record.domain != plan.domain:
            raise ValueError(
                f"Record {symbol!r} domain {record.domain!r} does not match plan {plan.domain!r}"
            )
        if record.freq != plan.frequency:
            raise ValueError(
                f"Record {symbol!r} frequency {record.freq!r} does not match plan "
                f"{plan.frequency!r}"
            )
        if not isinstance(record.entity, str) or not record.entity.strip():
            raise ValueError(f"Record {symbol!r} has no entity label")
        if not isinstance(record.metadata, dict):
            raise ValueError(f"Record {symbol!r} metadata must be an object")

        provider = record.metadata.get("source")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f"Record {symbol!r} metadata.source must be a non-empty string")

        timestamps = [
            _utc_timestamp(
                timestamp,
                timezone_name=timezone_name,
                label=f"record {symbol!r} timestamp[{index}]",
            )
            for index, timestamp in enumerate(record.timestamps)
        ]
        if not timestamps:
            raise ValueError(f"Record {symbol!r} has no timestamps")
        if len(timestamps) != len(record.values):
            raise ValueError(
                f"Record {symbol!r} timestamp count {len(timestamps)} does not match "
                f"target length {len(record.values)}"
            )
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError(
                f"Record {symbol!r} timestamps must be unique and strictly increasing"
            )
        if timestamps[0] < requested_start or timestamps[-1] >= requested_end:
            raise ValueError(
                f"Record {symbol!r} timestamps fall outside the requested range "
                f"[{requested_start.isoformat()}, {requested_end.isoformat()})"
            )

        record_start = _utc_timestamp(
            record.start,
            timezone_name=timezone_name,
            label=f"record {symbol!r} start",
        )
        record_end = _utc_timestamp(
            record.end,
            timezone_name=timezone_name,
            label=f"record {symbol!r} end",
        )
        if record_start != timestamps[0] or record_end != timestamps[-1]:
            raise ValueError(
                f"Record {symbol!r} start/end must equal its first/last timestamp"
            )

        values: list[float] = []
        for index, raw_value in enumerate(record.values):
            if isinstance(raw_value, bool):
                raise ValueError(f"Record {symbol!r} target[{index}] must be numeric, not boolean")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Record {symbol!r} target[{index}] is not numeric"
                ) from exc
            if math.isinf(value):
                raise ValueError(f"Record {symbol!r} target[{index}] must not be infinite")
            values.append(value)
        if not values or all(math.isnan(value) for value in values):
            raise ValueError(f"Record {symbol!r} target has no observed values")

        dimensions: dict[str, Any] = {
            "source_name": plan.source,
            "provider_name": provider.strip(),
            "source_series_id": symbol,
            "domain": plan.domain,
            "frequency": plan.frequency,
            "metric_type": plan.metadata.target.name,
            "target_name": plan.metadata.target.name,
            "target_description": plan.metadata.target.description,
        }
        if plan.metadata.target.unit:
            dimensions["target_unit"] = plan.metadata.target.unit

        for key in plan.metadata.dimensions:
            if key not in record.metadata:
                raise ValueError(
                    f"Record {symbol!r} is missing requested metadata dimension {key!r}"
                )
            raw_dimension = record.metadata[key]
            if raw_dimension is None or raw_dimension == "" or raw_dimension == []:
                raise ValueError(
                    f"Record {symbol!r} has an empty requested metadata dimension {key!r}"
                )
            try:
                dimensions[key] = normalize_dimension_value(raw_dimension)
            except ValueError as exc:
                raise ValueError(
                    f"Record {symbol!r} has invalid metadata dimension {key!r}: {exc}"
                ) from exc

        validated.append(
            {
                "item_id": symbol,
                "entity_label": record.entity.strip(),
                "location": record.geo.strip() if isinstance(record.geo, str) else "",
                "provider": provider.strip(),
                "timestamps": timestamps,
                "values": values,
                "dimensions": dimensions,
            }
        )

    return validated


def _offline_metadata(plan: CollectionPlan, rows: list[dict[str, Any]]) -> OfflineMetadata:
    actual_start = min(row["timestamps"][0] for row in rows)
    actual_end = max(row["timestamps"][-1] for row in rows)
    providers = list(dict.fromkeys(row["provider"] for row in rows))
    target = plan.metadata.target

    return OfflineMetadata(
        schema_version="2.0",
        dataset_semantics=DatasetSemantics(
            dataset_name=plan.dataset_id,
            category=plan.domain,
            subcategory=plan.metadata.subcategory,
            geo=plan.metadata.geo,
            providers=providers,
            time_granularity=plan.frequency,
            time_range=TimeRange(start=actual_start, end=actual_end),
            num_series=len(rows),
            description_short=plan.metadata.description_short,
            description_long=plan.metadata.description_long,
        ),
        item_schema=ItemSchema(
            entity_type=plan.metadata.entity_type,
            item_id_pattern="{series_id}",
            spatial_dimension=plan.metadata.spatial_dimension,
            target=[
                MetricField(
                    name=target.name,
                    description=target.description,
                    unit=target.unit,
                )
            ],
            past_feat_dynamic_real=[],
            static_features=[],
            item_semantics={
                row["item_id"]: ItemSemanticsEntry(
                    entity_label=row["entity_label"],
                    description=target.description,
                    location=row["location"],
                    confidence=0.0,
                    dimensions=row["dimensions"],
                )
                for row in rows
            },
        ),
        item_groups=[],
        query_hints=QueryHints(
            dataset_terms=plan.metadata.query_hints.dataset_terms,
            event_bias=plan.metadata.query_hints.event_bias,
        ),
    )


def _write_arrow(path: Path, rows: list[dict[str, Any]], frequency: str) -> None:
    schema = pa.schema(
        [
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("start", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("freq", pa.string(), nullable=False),
            pa.field("target", pa.list_(pa.float64()), nullable=False),
            pa.field(
                "timestamps",
                pa.list_(pa.timestamp("us", tz="UTC")),
                nullable=False,
            ),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {
                "item_id": row["item_id"],
                "start": row["timestamps"][0],
                "freq": frequency,
                "target": row["values"],
                "timestamps": row["timestamps"],
            }
            for row in rows
        ],
        schema=schema,
    )
    with path.open("wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)


def _write_metadata(path: Path, metadata: OfflineMetadata) -> None:
    payload = metadata.model_dump(
        mode="json",
        by_alias=True,
        exclude={"dataset_semantics": {"data_quality"}},
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_collection(plan: CollectionPlan, output_dir: Path) -> dict[str, Any]:
    """Fetch and atomically materialize ``data.arrow`` plus strict metadata."""

    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output directory must not already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    adapter = _build_exact_adapter(plan)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.collect-",
            dir=str(destination.parent),
        )
    )
    try:
        fetch_start = plan.start
        fetch_end = plan.end
        if plan.source == "noaa_isd":
            fetch_start = _utc_timestamp(
                plan.start,
                timezone_name=plan.timezone,
                label="plan.start",
            )
            fetch_end = _utc_timestamp(
                plan.end,
                timezone_name=plan.timezone,
                label="plan.end",
            )
        records = adapter.fetch(
            plan.symbols,
            fetch_start,
            fetch_end,
            plan.domain,
        )
        rows = _validated_records(plan, records)
        metadata = _offline_metadata(plan, rows)

        arrow_path = temporary / "data.arrow"
        metadata_path = temporary / "offline_metadata.json"
        _write_arrow(arrow_path, rows, plan.frequency)
        _write_metadata(metadata_path, metadata)

        readiness_dir = temporary / ".readiness"
        report = prepare_data_readiness(
            dataset_id=plan.dataset_id,
            arrow_files=[arrow_path],
            metadata_file=metadata_path,
            output_dir=readiness_dir,
        )
        if report.status != "ready":
            raise ValueError("Materialized collection did not pass offline readiness")
        shutil.rmtree(readiness_dir)

        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    timestamps = [timestamp for row in rows for timestamp in row["timestamps"]]
    return {
        "dataset_id": plan.dataset_id,
        "source": plan.source,
        "domain": plan.domain,
        "frequency": plan.frequency,
        "item_count": len(rows),
        "target_count": 1,
        "time_range": {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat(),
        },
        "arrow_file": str(destination / "data.arrow"),
        "offline_metadata_file": str(destination / "offline_metadata.json"),
    }

"""Arrow loading helpers for offline-discover."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

from .models import LoadedExample


@contextmanager
def _open_arrow_reader(path: Path) -> Iterator[ipc.RecordBatchReader]:
    try:
        with pa.memory_map(str(path), "r") as source:
            with ipc.open_file(source) as reader:
                yield reader
    except (pa.ArrowInvalid, OSError):
        with pa.memory_map(str(path), "r") as source:
            with ipc.open_stream(source) as reader:
                yield reader


def _read_arrow_batches(path: Path) -> Iterator[tuple[pa.RecordBatch, pa.Schema]]:
    with _open_arrow_reader(path) as reader:
        schema = reader.schema
        if hasattr(reader, "num_record_batches"):
            for batch_index in range(reader.num_record_batches):
                yield reader.get_batch(batch_index), schema
            return

        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                return
            if batch is None:
                return
            yield batch, schema


def _coerce_required_string(value: object, field_name: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {field_name} in shard {path}: expected non-empty string")
    return value


def _coerce_datetime(value: object, field_name: str, path: Path, item_id: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(
            f"Invalid {field_name} for item_id {item_id} in shard {path}: expected datetime"
        )
    return value


def _coerce_list(value: object, field_name: str, path: Path, item_id: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"Invalid {field_name} for item_id {item_id} in shard {path}: expected list"
        )
    return value


def _coerce_target_channels(value: object, path: Path, item_id: str) -> list[tuple[str, list[float]]]:
    target = _coerce_list(value, "target", path, item_id)
    if not target:
        return [(item_id, [])]
    if all(not isinstance(item, list) for item in target):
        return [(item_id, [float(item) for item in target])]
    if not all(isinstance(item, list) for item in target):
        raise ValueError(
            f"Invalid target for item_id {item_id} in shard {path}: expected scalar list or aligned channel lists"
        )

    channels = target
    lengths = {len(channel) for channel in channels}
    if len(lengths) != 1:
        raise ValueError(
            f"Invalid target for item_id {item_id} in shard {path}: target channel lengths differ"
        )
    return [
        (f"{item_id}__target_{channel_index}", [float(item) for item in channel])
        for channel_index, channel in enumerate(channels)
    ]


def _normalized_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _coerce_timestamps(
    value: object,
    *,
    field_present: bool,
    path: Path,
    item_id: str,
) -> list[datetime] | None:
    if not field_present:
        return None
    if value is None:
        raise ValueError(
            f"Invalid timestamps for item_id {item_id} in shard {path}: field must not be null"
        )

    raw_timestamps = _coerce_list(value, "timestamps", path, item_id)
    timestamps = [
        _coerce_datetime(timestamp, "timestamps", path, item_id)
        for timestamp in raw_timestamps
    ]
    normalized = [_normalized_datetime(timestamp) for timestamp in timestamps]
    if any(current <= previous for previous, current in zip(normalized, normalized[1:])):
        raise ValueError(
            f"Invalid timestamps for item_id {item_id} in shard {path}: "
            "values must be unique and strictly increasing"
        )
    return normalized


def _validate_timestamp_field(schema: pa.Schema, path: Path) -> None:
    field_type = schema.field("timestamps").type
    if not pa.types.is_list(field_type):
        raise ValueError(
            f"Invalid timestamps in shard {path}: expected list<timestamp[..., UTC]>"
        )
    value_type = field_type.value_type
    if not pa.types.is_timestamp(value_type) or value_type.tz != "UTC":
        raise ValueError(
            f"Invalid timestamps in shard {path}: expected list<timestamp[..., UTC]>"
        )


def load_arrow_examples(
    paths: list[Path], *, preserve_physical_order: bool = False
) -> list[LoadedExample]:
    if not paths:
        raise ValueError("load_arrow_examples requires at least one path")

    examples: list[LoadedExample] = []
    seen_item_ids: set[str] = set()
    seen_example_ids: set[str] = set()
    reference_schema: pa.Schema | None = None

    for path in paths:
        for batch, schema in _read_arrow_batches(path):
            if reference_schema is None:
                reference_schema = schema
            elif not schema.equals(reference_schema, check_metadata=True):
                raise ValueError("incompatible schema across shards")

            columns = [batch.column(i) for i in range(batch.num_columns)]
            field_names = batch.schema.names
            timestamps_present = "timestamps" in field_names
            if timestamps_present:
                _validate_timestamp_field(batch.schema, path)
            for row_index in range(batch.num_rows):
                row = {
                    field_names[column_index]: columns[column_index][row_index].as_py()
                    for column_index in range(batch.num_columns)
                }
                item_id = row.get("item_id")
                if item_id is None:
                    raise ValueError(f"Missing item_id in shard {path}")
                item_id = _coerce_required_string(item_id, "item_id", path)
                if item_id in seen_item_ids:
                    raise ValueError(f"Duplicate item_id: {item_id}")
                seen_item_ids.add(item_id)

                start = _coerce_datetime(row.get("start"), "start", path, item_id)
                if timestamps_present and start.tzinfo is None:
                    raise ValueError(
                        f"Invalid start for item_id {item_id} in shard {path}: "
                        "explicit timestamps require a timezone-aware UTC start"
                    )
                timestamps = _coerce_timestamps(
                    row.get("timestamps"),
                    field_present=timestamps_present,
                    path=path,
                    item_id=item_id,
                )
                if timestamps is not None:
                    start = _normalized_datetime(start)
                for example_id, target in _coerce_target_channels(row.get("target"), path, item_id):
                    if example_id in seen_example_ids:
                        raise ValueError(f"Duplicate example_id: {example_id}")
                    if timestamps is not None:
                        if len(timestamps) != len(target):
                            raise ValueError(
                                f"Invalid timestamps for item_id {item_id} in shard {path}: "
                                f"timestamp count {len(timestamps)} does not match target length {len(target)}"
                            )
                        if timestamps and _normalized_datetime(start) != _normalized_datetime(
                            timestamps[0]
                        ):
                            raise ValueError(
                                f"Invalid start for item_id {item_id} in shard {path}: "
                                "start must equal the first explicit timestamp"
                            )
                    seen_example_ids.add(example_id)
                    examples.append(
                        LoadedExample(
                            example_id=example_id,
                            start=start,
                            freq=_coerce_required_string(row.get("freq"), "freq", path),
                            target=target,
                            timestamps=timestamps,
                        )
                    )

    if preserve_physical_order:
        return examples
    return sorted(examples, key=lambda example: example.example_id)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .schema import OfflineMetadata, parse_offline_metadata


_TARGET_CHANNEL_MARKER = "__target_"


@dataclass(frozen=True)
class OfflineMetadataRuntime:
    metadata: OfflineMetadata
    item_by_id: dict[str, dict]


def _split_target_channel_example_id(example_id: str) -> tuple[str, int] | None:
    base_id, marker, channel_text = example_id.rpartition(_TARGET_CHANNEL_MARKER)
    if marker != _TARGET_CHANNEL_MARKER or not base_id or not channel_text.isdigit():
        return None
    return base_id, int(channel_text)


def _item_projection(item_id: str, item, *, target=None, target_index: int | None = None) -> dict:
    attrs = dict(item.dimensions)
    entity_label = item.entity_label
    description_parts = [item.description]
    if target is not None and target_index is not None:
        attrs.update(
            {
                "base_item_id": item_id,
                "target_channel_index": target_index,
                "target_name": target.name,
                "target_description": target.description,
                "metric_type": target.name,
                "metric_family": target.name,
                "metric_unit": target.unit,
            }
        )
        entity_label = f"{entity_label} {target.name}".strip()
        description_parts.extend(
            part
            for part in (
                f"Target channel: {target.name}.",
                target.description,
            )
            if part
        )

    return {
        "entity_label": entity_label,
        "description": " ".join(part for part in description_parts if part).strip(),
        "location": item.location,
        "confidence": item.confidence,
        "provider": attrs.get("provider_name", ""),
        "service": attrs.get("provider_service", ""),
        "variable": attrs.get("metric_type", ""),
        "metric_family": attrs.get("metric_family", ""),
        "attributes": attrs,
    }


def load_runtime_metadata(path: Path, *, example_ids: Sequence[str]) -> OfflineMetadataRuntime:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = parse_offline_metadata(raw, path)

    expected_ids = sorted(str(example_id) for example_id in example_ids)
    actual_ids = sorted(model.item_schema.item_semantics)
    target_by_expanded_id: dict[str, tuple[str, int]] = {}
    use_expanded_target_ids = False

    if expected_ids == actual_ids:
        if model.dataset_semantics.num_series != len(expected_ids):
            raise ValueError(
                "dataset_semantics.num_series does not match Arrow examples: "
                f"declared={model.dataset_semantics.num_series} actual={len(expected_ids)}"
            )
    else:
        parsed_expected_ids = {
            example_id: _split_target_channel_example_id(example_id) for example_id in expected_ids
        }
        if all(parsed is not None for parsed in parsed_expected_ids.values()):
            target_count = len(model.item_schema.target)
            generated_ids: list[str] = []
            for base_id in actual_ids:
                for target_index in range(target_count):
                    generated_id = f"{base_id}{_TARGET_CHANNEL_MARKER}{target_index}"
                    generated_ids.append(generated_id)
                    target_by_expanded_id[generated_id] = (base_id, target_index)

            if model.dataset_semantics.num_series != len(actual_ids):
                raise ValueError(
                    "dataset_semantics.num_series does not match OfflineMetadata item_semantics: "
                    f"declared={model.dataset_semantics.num_series} actual={len(actual_ids)}"
                )
            if expected_ids == sorted(generated_ids):
                use_expanded_target_ids = True

        if not use_expanded_target_ids:
            if model.dataset_semantics.num_series != len(expected_ids):
                raise ValueError(
                    "dataset_semantics.num_series does not match Arrow examples: "
                    f"declared={model.dataset_semantics.num_series} actual={len(expected_ids)}"
                )
            raise ValueError(
                "Arrow example IDs do not match OfflineMetadata item_semantics: "
                f"expected={expected_ids} actual={actual_ids}"
            )

    item_by_id: dict[str, dict] = {}
    runtime_ids = expected_ids if use_expanded_target_ids else actual_ids

    if use_expanded_target_ids:
        for example_id in runtime_ids:
            item_id, target_index = target_by_expanded_id[example_id]
            item_by_id[example_id] = _item_projection(
                item_id,
                model.item_schema.item_semantics[item_id],
                target=model.item_schema.target[target_index],
                target_index=target_index,
            )
    else:
        for item_id, item in model.item_schema.item_semantics.items():
            item_by_id[item_id] = _item_projection(item_id, item)

    valid_group_ids = set(actual_ids) | set(runtime_ids)
    for group in model.item_groups:
        unknown_example_ids = sorted(set(group.example_ids) - valid_group_ids)
        if unknown_example_ids:
            raise ValueError(
                "Item group references unknown example IDs: "
                f"group_id={group.group_id} unknown={unknown_example_ids}"
            )

    return OfflineMetadataRuntime(
        metadata=model,
        item_by_id=item_by_id,
    )

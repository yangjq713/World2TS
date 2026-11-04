from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from world2ts_core.offline_discover.runtime import load_runtime_metadata


def _scratch_root(name: str) -> Path:
    root = Path(__file__).resolve().parents[1] / ".test_scratch" / f"{name}-{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rideshare_payload(example_ids: list[str]) -> dict:
    return {
        "schema_version": "2.0",
        "dataset_semantics": {
            "dataset_name": "RIDESHARE_WITH_MISSING",
            "category": "transport",
            "subcategory": "rideshare pricing",
            "geo": "Boston, Massachusetts, USA",
            "providers": ["Lyft", "Uber"],
            "time_granularity": "H",
            "time_range": {
                "start": "2018-11-26T06:00:00",
                "end": "2018-12-18T18:00:00",
            },
            "num_series": len(example_ids),
            "description_short": "Hourly rideshare series.",
            "description_long": "Hourly rideshare and weather-aligned metric series.",
        },
        "item_schema": {
            "entity_type": "rideshare_metric_series",
            "item_id_pattern": "T{n}",
            "spatial_dimension": "source_location",
            "target": [{"name": "value", "description": "metric value", "unit": ""}],
            "past_feat_dynamic_real": [],
            "static_features": [],
            "item_semantics": {
                example_id: {
                    "entity_label": f"{example_id} Lyft price_mean",
                    "description": "Hourly quoted ride price.",
                    "location": "Boston, Massachusetts, USA",
                    "confidence": 1.0,
                    "attributes": {
                        "source_location": "Back Bay",
                        "provider_name": "Lyft",
                        "provider_service": "Lux",
                        "metric_type": "price_mean",
                        "metric_family": "price",
                    },
                }
                for example_id in example_ids
            },
        },
        "item_groups": [
            {
                "group_id": "back_bay__lyft__lux",
                "label": "Back Bay | Lyft | Lux",
                "example_ids": list(example_ids),
                "shared_terms": ["Back Bay", "Lyft", "Lux", "rideshare"],
            }
        ],
        "query_hints": {
            "dataset_terms": ["Boston rideshare", "Uber Lyft Boston"],
            "event_bias": ["weather", "surge pricing", "public transit disruption"],
        },
    }


def test_runtime_load_projects_strict_item_semantics():
    root = _scratch_root("runtime-v2")
    metadata_path = _write(root / "rideshare.offline_metadata.json", _rideshare_payload(["T0", "T1"]))

    runtime = load_runtime_metadata(metadata_path, example_ids=["T0", "T1"])

    assert runtime.metadata.schema_version == "2.0"
    assert runtime.item_by_id["T0"]["provider"] == "Lyft"


def test_runtime_load_rejects_missing_arrow_item_ids():
    root = _scratch_root("runtime-v2-mismatch")
    payload = _rideshare_payload(["T0", "T1"])
    payload["dataset_semantics"]["num_series"] = 1
    metadata_path = _write(root / "rideshare.offline_metadata.json", payload)

    with pytest.raises(ValueError, match="Arrow example IDs do not match OfflineMetadata item_semantics"):
        load_runtime_metadata(metadata_path, example_ids=["T0"])


def test_runtime_load_rejects_num_series_mismatch():
    root = _scratch_root("runtime-v2-num-series")
    payload = _rideshare_payload(["T0", "T1"])
    payload["dataset_semantics"]["num_series"] = 3
    metadata_path = _write(root / "rideshare.offline_metadata.json", payload)

    with pytest.raises(ValueError, match="dataset_semantics.num_series does not match Arrow examples"):
        load_runtime_metadata(metadata_path, example_ids=["T0", "T1"])


def test_runtime_load_rejects_unknown_group_example_ids():
    root = _scratch_root("runtime-v2-unknown-group-example")
    payload = _rideshare_payload(["T0", "T1"])
    payload["item_groups"][0]["example_ids"] = ["T0", "ghost"]
    metadata_path = _write(root / "rideshare.offline_metadata.json", payload)

    with pytest.raises(ValueError, match="Item group references unknown example IDs"):
        load_runtime_metadata(metadata_path, example_ids=["T0", "T1"])


def test_runtime_expands_multivariate_target_item_semantics():
    root = _scratch_root("runtime-v2-expanded-targets")
    payload = _rideshare_payload(["station_1"])
    payload["dataset_semantics"]["num_series"] = 1
    payload["item_schema"]["target"] = [
        {"name": "in_flow", "description": "Inbound flow.", "unit": "passengers"},
        {"name": "out_flow", "description": "Outbound flow.", "unit": "passengers"},
    ]
    payload["item_schema"]["item_semantics"] = {
        "station_1": {
            "entity_label": "Station 1",
            "description": "Station passenger flow.",
            "location": "Beijing, China",
            "confidence": 0.8,
            "attributes": {
                "station_name": "Station 1",
                "line_name": "Line 1",
            },
        }
    }
    payload["item_groups"] = [
        {
            "group_id": "station_1",
            "label": "Station 1",
            "example_ids": ["station_1"],
            "shared_terms": ["Station 1", "Beijing subway"],
        }
    ]
    metadata_path = _write(root / "subway.offline_metadata.json", payload)

    runtime = load_runtime_metadata(
        metadata_path,
        example_ids=["station_1__target_0", "station_1__target_1"],
    )

    assert sorted(runtime.item_by_id) == ["station_1__target_0", "station_1__target_1"]
    assert runtime.item_by_id["station_1__target_0"]["entity_label"] == "Station 1 in_flow"
    assert runtime.item_by_id["station_1__target_0"]["metric_family"] == "in_flow"
    assert runtime.item_by_id["station_1__target_1"]["variable"] == "out_flow"



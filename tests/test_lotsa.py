"""Tests for LOTSA metadata core logic."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pyarrow as pa
import pytest


def test_lotsa_dir_default(monkeypatch, tmp_path):
    """_lotsa_dir uses store_root config."""
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path / "mystore"))
    from world2ts_core import lotsa
    assert lotsa._lotsa_dir() == tmp_path / "mystore" / "LOTSA"


def test_meta_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path / "s"))
    from world2ts_core import lotsa
    assert lotsa._meta_dir() == tmp_path / "s" / "LOTSA" / "meta_data_info"


def test_validate_phase_ok():
    from world2ts_core import lotsa
    assert lotsa.validate_phase(1) == "phase_01_transport_mobility"
    assert lotsa.validate_phase(6) == "phase_06_compute_misc"


def test_validate_phase_bad():
    from world2ts_core import lotsa
    with pytest.raises(ValueError, match="Invalid phase"):
        lotsa.validate_phase(0)
    with pytest.raises(ValueError, match="Invalid phase"):
        lotsa.validate_phase(7)
    with pytest.raises(ValueError, match="Invalid phase"):
        lotsa.validate_phase(-1)


# --- Sample HF API response fixtures ---
# Matches the real HF datasets-server format: features have top-level "name"/"type"
HF_FIRST_ROWS_RESPONSE = {
    "features": [
        {"feature_idx": 0, "name": "item_id", "type": {"_type": "Value", "dtype": "string"}},
        {"feature_idx": 1, "name": "start", "type": {"_type": "Value", "dtype": "timestamp[s]"}},
        {"feature_idx": 2, "name": "freq", "type": {"_type": "Value", "dtype": "string"}},
        {"feature_idx": 3, "name": "target", "type": {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "float32"}}},
        {"feature_idx": 4, "name": "past_feat_dynamic_real", "type": {
            "_type": "Sequence",
            "length": 3,
            "feature": {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "float32"}},
        }},
    ],
    "rows": [
        {"row_idx": 0, "row": {
            "item_id": "T1",
            "start": "2002-01-01T00:00:00",
            "freq": "30T",
            "target": [1.0, 2.0, 3.0],
            "past_feat_dynamic_real": [[1.0], [2.0], [3.0]],
        }, "truncated_cells": []}
    ],
}

HF_INFO_RESPONSE = {
    "dataset_info": {
        "australian_electricity_demand": {
            "splits": {"train": {"num_examples": 5, "num_bytes": 5237038}}
        },
        "test_ds": {
            "splits": {"train": {"num_examples": 5, "num_bytes": 1000}}
        },
        "bad_ds": {
            "splits": {"train": {"num_examples": 1, "num_bytes": 100}}
        },
    }
}


def _make_curl_result(response_dict):
    """Create a mock subprocess result that returns given JSON."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = json.dumps(response_dict)
    return m


def _curl_fail_result():
    m = MagicMock()
    m.returncode = 7
    m.stdout = ""
    return m


def _mock_curl_two_apis(*args, **kwargs):
    """Mock that returns first-rows for first call, info for second."""
    # Detect which API by URL
    cmd = args[0] if args else kwargs.get("args", [])
    url = cmd[-1] if cmd else ""
    if "first-rows" in url:
        return _make_curl_result(HF_FIRST_ROWS_RESPONSE)
    elif "info" in url:
        return _make_curl_result(HF_INFO_RESPONSE)
    return _curl_fail_result()


def test_hf_fetch_metadata_parse():
    from world2ts_core import lotsa
    with patch("subprocess.run", side_effect=_mock_curl_two_apis):
        result = lotsa.hf_fetch_metadata("australian_electricity_demand")

    assert result["hf_num_examples"] == 5
    assert result["hf_preview_freq"] == "30T"
    assert result["hf_preview_start"] == "2002-01-01T00:00:00"
    assert result["hf_preview_item_id"] == "T1"
    assert result["hf_target_dim"] == 1
    assert result["hf_past_feat_dynamic_real_dim"] == 3
    assert result["hf_has_freq_field"] is True
    assert result["hf_preview_target_length"] == 3
    assert "target" in result["hf_preview_keys"]
    assert result["hf_size_bytes"] == 5237038


def test_hf_fetch_metadata_retry():
    """Two failures on first-rows, then success on both APIs."""
    from world2ts_core import lotsa

    call_count = [0]
    def _mock_retry(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        url = cmd[-1] if cmd else ""
        if "first-rows" in url:
            call_count[0] += 1
            if call_count[0] <= 2:
                return _curl_fail_result()
            return _make_curl_result(HF_FIRST_ROWS_RESPONSE)
        return _make_curl_result(HF_INFO_RESPONSE)

    with patch("subprocess.run", side_effect=_mock_retry):
        result = lotsa.hf_fetch_metadata("test_ds", timeout=5)
    assert result["hf_num_examples"] == 5


def test_hf_fetch_metadata_exhaust():
    """All retries fail → SourceError."""
    from world2ts_core import lotsa
    from world2ts_core.errors import SourceError

    def _always_fail(*args, **kwargs):
        return _curl_fail_result()

    with patch("subprocess.run", side_effect=_always_fail):
        with pytest.raises(SourceError, match="failed after 3 attempts"):
            lotsa.hf_fetch_metadata("bad_ds", timeout=5)


def test_load_phase_datasets(monkeypatch, tmp_path):
    from world2ts_core import lotsa
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    lotsa_dir = tmp_path / "LOTSA"
    lotsa_dir.mkdir()
    (lotsa_dir / "phase_06_compute_misc.jsonl").write_text(
        '{"dataset_name": "bull", "dataset_id": "lotsa_bull"}\n'
        '{"dataset_name": "elf", "dataset_id": "lotsa_elf"}\n'
    )
    result = lotsa.load_phase_datasets(6)
    assert len(result) == 2
    assert result[0]["dataset_name"] == "bull"
    assert result[1]["dataset_name"] == "elf"


def test_load_phase_datasets_missing(monkeypatch, tmp_path):
    from world2ts_core import lotsa
    from world2ts_core.errors import StoreError
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    (tmp_path / "LOTSA").mkdir()
    with pytest.raises(StoreError, match="not found"):
        lotsa.load_phase_datasets(4)


def test_load_enriched_datasets_empty(monkeypatch, tmp_path):
    from world2ts_core import lotsa
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    (tmp_path / "LOTSA").mkdir()
    result = lotsa.load_enriched_datasets(3)
    assert result == []


def test_save_enriched_roundtrip(monkeypatch, tmp_path):
    from world2ts_core import lotsa
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    (tmp_path / "LOTSA").mkdir()
    entries = [
        {"dataset_name": "a", "hf_num_examples": 10},
        {"dataset_name": "b", "hf_num_examples": 20},
    ]
    path = lotsa.save_enriched(6, entries)
    assert path.exists()
    loaded = lotsa.load_enriched_datasets(6)
    assert len(loaded) == 2
    assert loaded[0]["hf_num_examples"] == 10


def test_merge_hf_preserves_existing():
    from world2ts_core import lotsa
    existing = {
        "dataset_name": "bull",
        "metadata_status": "phase2_hf_enriched",
        "classification_confidence": "high",
        "physical_edges": 248,
        "hf_num_examples": 40,
    }
    hf_meta = {
        "hf_num_examples": 41,
        "hf_preview_freq": "H",
    }
    merged = lotsa.merge_hf_into_entry(existing, hf_meta)
    assert merged["physical_edges"] == 248  # preserved
    assert merged["hf_num_examples"] == 41  # updated
    assert merged["hf_preview_freq"] == "H"  # added
    assert merged["metadata_status"] == "phase2_hf_enriched"  # not re-promoted


def test_merge_hf_status_promotion():
    from world2ts_core import lotsa
    existing = {
        "dataset_name": "x",
        "metadata_status": "phase1_coarse_inventory",
        "classification_confidence": "low_medium",
    }
    hf_meta = {"hf_num_examples": 5}
    merged = lotsa.merge_hf_into_entry(existing, hf_meta)
    assert merged["metadata_status"] == "phase2_hf_enriched"
    assert merged["classification_confidence"] == "high"


def test_find_dataset_phase(monkeypatch, tmp_path):
    from world2ts_core import lotsa
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    lotsa_dir = tmp_path / "LOTSA"
    lotsa_dir.mkdir()
    (lotsa_dir / "phase_04_health_public.jsonl").write_text(
        '{"dataset_name": "hospital"}\n'
    )
    # Create empty files for other phases so they don't raise StoreError
    for p, label in lotsa.PHASE_MAP.items():
        f = lotsa_dir / f"{label}.jsonl"
        if not f.exists():
            f.write_text("")
    assert lotsa.find_dataset_phase("hospital") == 4


def test_find_dataset_phase_not_found(monkeypatch, tmp_path):
    from world2ts_core import lotsa
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    lotsa_dir = tmp_path / "LOTSA"
    lotsa_dir.mkdir()
    for label in lotsa.PHASE_MAP.values():
        (lotsa_dir / f"{label}.jsonl").write_text("")
    with pytest.raises(ValueError, match="not found"):
        lotsa.find_dataset_phase("nonexistent")


def _write_arrow(path: Path, item_ids: list[str]) -> Path:
    table = pa.Table.from_pylist(
        [
            {
                "item_id": item_id,
                "start": datetime(2020, 1, 1),
                "freq": "D",
                "target": [1.0, 2.0, 3.0],
            }
            for item_id in item_ids
        ]
    )
    with path.open("wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return path


def _offline_metadata(item_ids: list[str]) -> dict:
    return {
        "schema_version": "2.0",
        "dataset_semantics": {
            "dataset_name": "test_ds",
            "category": "health",
            "subcategory": "test observations",
            "geo": "United States",
            "providers": ["Example"],
            "time_granularity": "D",
            "time_range": {
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-03T00:00:00",
            },
            "num_series": len(item_ids),
            "description_short": "Daily test observations.",
            "description_long": "Daily test observations for metadata validation.",
        },
        "item_schema": {
            "entity_type": "test_series",
            "item_id_pattern": "{item_id}",
            "spatial_dimension": "",
            "target": [{"name": "value", "description": "Observed value.", "unit": ""}],
            "past_feat_dynamic_real": [],
            "static_features": [],
            "item_semantics": {
                item_id: {
                    "entity_label": item_id,
                    "description": f"Daily observations for {item_id}.",
                    "location": "United States",
                    "confidence": 1.0,
                    "attributes": {"cohort": "test"},
                }
                for item_id in item_ids
            },
        },
        "item_groups": [],
        "query_hints": {"dataset_terms": [], "event_bias": []},
    }


def test_build_metadata_briefing(monkeypatch, tmp_path):
    from world2ts_core import lotsa

    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    lotsa_dir = tmp_path / "LOTSA"
    lotsa_dir.mkdir()
    for phase, label in lotsa.PHASE_MAP.items():
        rows = (
            '{"dataset_name": "test_ds", "dataset_id": "lotsa_test_ds"}\n'
            if phase == 6
            else ""
        )
        (lotsa_dir / f"{label}.jsonl").write_text(rows)
    (lotsa_dir / "phase_06_compute_misc_enriched.jsonl").write_text(
        '{"dataset_name": "test_ds", "dataset_id": "lotsa_test_ds", '
        '"hf_num_examples": 2, "hf_preview_freq": "D"}\n'
    )
    arrow = _write_arrow(tmp_path / "data.arrow", ["A", "B"])

    briefing = lotsa.build_metadata_briefing("test_ds", [arrow])

    assert briefing["dataset_id"] == "test_ds"
    assert briefing["arrow_facts"]["base_series_ids"] == ["A", "B"]
    assert briefing["arrow_facts"]["runtime_series_count"] == 2
    assert briefing["arrow_facts"]["time_range"]["end"] == "2020-01-03T00:00:00"
    requirements = " ".join(briefing["agent_output"]["requirements"])
    assert "search queries" in requirements
    assert "query_hints empty" in requirements


def test_base_series_inventory_only_collapses_multichannel_ids():
    from world2ts_core import lotsa

    assert lotsa._base_series_inventory(["A__target_0", "A__target_1"]) == (["A"], 2)
    assert lotsa._base_series_inventory(["A__target_0"]) == (["A__target_0"], 1)


def test_validate_offline_metadata_uses_offline_readiness(tmp_path):
    from world2ts_core import lotsa

    arrow = _write_arrow(tmp_path / "data.arrow", ["A", "B"])
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(_offline_metadata(["A", "B"])))

    result = lotsa.validate_offline_metadata(metadata, [arrow])

    assert result["status"] == "ready"
    assert result["runtime_series_count"] == 2
    assert len(result["series_id_order_sha256"]) == 64


def test_render_metadata_card_has_no_agent_placeholders(tmp_path):
    from world2ts_core import lotsa

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(_offline_metadata(["A"])))
    metadata = lotsa.load_offline_metadata(metadata_path)

    card = lotsa.render_metadata_card(metadata, metadata_path.name)

    assert "# TEST_DS" in card
    assert "The strict JSON is authoritative" in card
    assert "<!-- TODO" not in card
    assert "event_bias" not in card


def test_get_metadata_status(monkeypatch, tmp_path):
    from world2ts_core import lotsa
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    lotsa_dir = tmp_path / "LOTSA"
    lotsa_dir.mkdir()
    meta_dir = lotsa_dir / "meta_data_info"
    meta_dir.mkdir()

    # Phase 6 coarse
    (lotsa_dir / "phase_06_compute_misc.jsonl").write_text(
        '{"dataset_name": "bull", "dataset_id": "lotsa_bull"}\n'
        '{"dataset_name": "elf", "dataset_id": "lotsa_elf"}\n'
    )
    # Phase 6 enriched (only bull)
    (lotsa_dir / "phase_06_compute_misc_enriched.jsonl").write_text(
        '{"dataset_name": "bull", "dataset_id": "lotsa_bull", "hf_num_examples": 41}\n'
    )
    # Strict metadata and rendered card exist for bull only
    (meta_dir / "BULL.metadata_briefing.json").write_text("{}")
    (meta_dir / "BULL.offline_metadata.json").write_text(
        json.dumps(_offline_metadata(["A"]))
    )
    (meta_dir / "BULL.md").write_text("# BULL\ncontent")

    # Create empty files for other phases
    for p, label in lotsa.PHASE_MAP.items():
        if p != 6:
            (lotsa_dir / f"{label}.jsonl").write_text("")

    status = lotsa.get_metadata_status(phase=6)
    assert len(status) == 2

    bull_status = next(s for s in status if s["dataset_name"] == "bull")
    assert bull_status["hf_enriched"] is True
    assert bull_status["briefing_exists"] is True
    assert bull_status["offline_metadata_exists"] is True
    assert bull_status["schema_valid"] is True
    assert bull_status["card_exists"] is True

    elf_status = next(s for s in status if s["dataset_name"] == "elf")
    assert elf_status["hf_enriched"] is False
    assert elf_status["briefing_exists"] is False
    assert elf_status["offline_metadata_exists"] is False
    assert elf_status["schema_valid"] is False
    assert elf_status["card_exists"] is False


def test_get_metadata_status_includes_enriched_without_coarse(monkeypatch, tmp_path):
    from world2ts_core import lotsa

    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    lotsa_dir = tmp_path / "LOTSA"
    lotsa_dir.mkdir()
    (lotsa_dir / "phase_03_environment_climate_enriched.jsonl").write_text(
        '{"dataset_name": "beijing_air_quality", '
        '"dataset_id": "lotsa_beijing_air_quality"}\n'
    )

    status = lotsa.get_metadata_status(phase=3)

    assert [item["dataset_name"] for item in status] == ["beijing_air_quality"]
    assert status[0]["hf_enriched"] is True


def test_get_metadata_status_includes_arrow_only_without_phase(monkeypatch, tmp_path):
    from world2ts_core import lotsa

    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    arrow_dir = tmp_path / "LOTSA" / "data" / "elecdemand"
    arrow_dir.mkdir(parents=True)
    _write_arrow(arrow_dir / "data.arrow", ["T1"])

    status = lotsa.get_metadata_status()

    assert [item["dataset_name"] for item in status] == ["elecdemand"]
    assert status[0]["phase"] == "arrow_only"
    assert status[0]["dataset_id"] == "elecdemand"

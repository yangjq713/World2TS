"""CLI integration tests for metadata commands."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pyarrow as pa
import pytest

from world2ts_core.cli import app
from tests.conftest import make_cli_runner

runner = make_cli_runner(mix_stderr=False)


@pytest.fixture
def lotsa_store(tmp_path, monkeypatch):
    """Set up a tmp LOTSA directory with phase 6 data."""
    monkeypatch.setenv("WORLD2TS_STORE_ROOT", str(tmp_path))
    lotsa_dir = tmp_path / "LOTSA"
    lotsa_dir.mkdir()
    meta_dir = lotsa_dir / "meta_data_info"
    meta_dir.mkdir()
    (lotsa_dir / "phase_06_compute_misc.jsonl").write_text(
        '{"dataset_name": "bull", "dataset_id": "lotsa_bull"}\n'
        '{"dataset_name": "elf", "dataset_id": "lotsa_elf"}\n'
    )
    from world2ts_core.lotsa import PHASE_MAP
    for p, label in PHASE_MAP.items():
        f = lotsa_dir / f"{label}.jsonl"
        if not f.exists():
            f.write_text("")
    return lotsa_dir


HF_FIRST_ROWS = {
    "features": [
        {"feature_idx": 0, "name": "item_id", "type": {"_type": "Value", "dtype": "string"}},
        {"feature_idx": 1, "name": "start", "type": {"_type": "Value", "dtype": "timestamp[s]"}},
        {"feature_idx": 2, "name": "freq", "type": {"_type": "Value", "dtype": "string"}},
        {"feature_idx": 3, "name": "target", "type": {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "float32"}}},
    ],
    "rows": [{"row_idx": 0, "row": {"item_id": "T1", "start": "2016-01-01", "freq": "H", "target": [1.0, 2.0]}, "truncated_cells": []}],
}

HF_INFO = {
    "dataset_info": {
        "bull": {"splits": {"train": {"num_examples": 41, "num_bytes": 11511245}}},
        "elf": {"splits": {"train": {"num_examples": 1, "num_bytes": 87192}}},
    }
}


def _mock_curl_ok(*args, **kwargs):
    cmd = args[0] if args else kwargs.get("args", [])
    url = cmd[-1] if cmd else ""
    m = MagicMock()
    m.returncode = 0
    if "info" in url:
        m.stdout = json.dumps(HF_INFO)
    else:
        m.stdout = json.dumps(HF_FIRST_ROWS)
    return m


def _write_arrow(path: Path) -> Path:
    table = pa.Table.from_pylist(
        [
            {
                "item_id": "T1",
                "start": datetime(2020, 1, 1),
                "freq": "D",
                "target": [1.0, 2.0, 3.0],
            }
        ]
    )
    with path.open("wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return path


def _write_metadata(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "dataset_semantics": {
                    "dataset_name": "bull",
                    "category": "health",
                    "subcategory": "test observations",
                    "geo": "United States",
                    "providers": ["Example"],
                    "time_granularity": "D",
                    "time_range": {
                        "start": "2020-01-01T00:00:00",
                        "end": "2020-01-03T00:00:00",
                    },
                    "num_series": 1,
                    "description_short": "Daily test observations.",
                    "description_long": "Daily test observations for metadata validation.",
                },
                "item_schema": {
                    "entity_type": "test_series",
                    "item_id_pattern": "{item_id}",
                    "spatial_dimension": "",
                    "target": [
                        {"name": "value", "description": "Observed value.", "unit": ""}
                    ],
                    "past_feat_dynamic_real": [],
                    "static_features": [],
                    "item_semantics": {
                        "T1": {
                            "entity_label": "Test series",
                            "description": "Daily test observations.",
                            "location": "United States",
                            "confidence": 1.0,
                            "attributes": {"cohort": "test"},
                        }
                    },
                },
                "item_groups": [],
                "query_hints": {"dataset_terms": [], "event_bias": []},
            }
        )
    )
    return path


def test_metadata_help():
    result = runner.invoke(app, ["metadata", "--help"])
    assert result.exit_code == 0
    assert "hf-fetch" in result.stdout
    assert "enrich" in result.stdout
    assert "prepare" in result.stdout
    assert "validate" in result.stdout
    assert "render-card" in result.stdout
    assert "card-init" not in result.stdout
    assert "status" in result.stdout


def test_hf_fetch_single(lotsa_store):
    with patch("subprocess.run", side_effect=_mock_curl_ok):
        result = runner.invoke(app, ["--json", "metadata", "hf-fetch", "--dataset", "bull"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["hf_num_examples"] == 41


def test_hf_fetch_phase(lotsa_store):
    with patch("subprocess.run", side_effect=_mock_curl_ok):
        result = runner.invoke(app, ["--json", "metadata", "hf-fetch", "--phase", "6"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) == 2


def test_hf_fetch_exclusive(lotsa_store):
    result = runner.invoke(app, ["--json", "metadata", "hf-fetch", "--dataset", "x", "--phase", "6"])
    assert result.exit_code == 1


def test_hf_fetch_neither(lotsa_store):
    result = runner.invoke(app, ["--json", "metadata", "hf-fetch"])
    assert result.exit_code == 1


def test_enrich_creates_file(lotsa_store):
    with patch("subprocess.run", side_effect=_mock_curl_ok):
        result = runner.invoke(app, ["--json", "metadata", "enrich", "--phase", "6"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["succeeded"] == 2
    assert data["failed"] == 0
    enriched_path = lotsa_store / "phase_06_compute_misc_enriched.jsonl"
    assert enriched_path.exists()


def test_enrich_incremental(lotsa_store):
    enriched_path = lotsa_store / "phase_06_compute_misc_enriched.jsonl"
    enriched_path.write_text(
        '{"dataset_name": "bull", "dataset_id": "lotsa_bull", '
        '"metadata_status": "phase2_hf_enriched", "physical_edges": 99, '
        '"hf_num_examples": 40}\n'
    )
    with patch("subprocess.run", side_effect=_mock_curl_ok):
        result = runner.invoke(app, ["--json", "metadata", "enrich", "--phase", "6"])
    assert result.exit_code == 0
    from world2ts_core.lotsa import load_enriched_datasets
    enriched = load_enriched_datasets(6)
    bull = next(e for e in enriched if e["dataset_name"] == "bull")
    assert bull["physical_edges"] == 99
    assert bull["hf_num_examples"] == 41


def test_prepare_allows_arrow_only_dataset(lotsa_store, tmp_path):
    arrow = _write_arrow(tmp_path / "data.arrow")
    output = tmp_path / "UNREGISTERED.metadata_briefing.json"
    result = runner.invoke(
        app,
        [
            "--json",
            "metadata",
            "prepare",
            "--dataset",
            "unregistered",
            "--arrow-file",
            str(arrow),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    briefing = json.loads(output.read_text())
    assert briefing["huggingface_structure"] == {}
    assert briefing["registry_context"]["metadata_status"] == "arrow_only"


def test_prepare_creates_briefing(lotsa_store, tmp_path):
    enriched_path = lotsa_store / "phase_06_compute_misc_enriched.jsonl"
    enriched_path.write_text(
        '{"dataset_name": "bull", "dataset_id": "lotsa_bull", "hf_num_examples": 1, '
        '"hf_preview_freq": "D", "hf_preview_start": "2020-01-01", '
        '"hf_target_dim": 1, "hf_past_feat_dynamic_real_dim": 0, '
        '"hf_size_bytes": 11511245}\n'
    )
    arrow = _write_arrow(tmp_path / "data.arrow")
    result = runner.invoke(
        app,
        [
            "--json",
            "metadata",
            "prepare",
            "--dataset",
            "bull",
            "--arrow-file",
            str(arrow),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    briefing_path = Path(data["briefing_path"])
    assert briefing_path.name == "BULL.metadata_briefing.json"
    briefing = json.loads(briefing_path.read_text())
    assert briefing["arrow_facts"]["runtime_series_ids"] == ["T1"]
    assert briefing["agent_output"]["path"].endswith("BULL.offline_metadata.json")


def test_validate_and_render_card(lotsa_store, tmp_path):
    arrow = _write_arrow(tmp_path / "data.arrow")
    metadata = _write_metadata(tmp_path / "BULL.offline_metadata.json")
    result = runner.invoke(
        app,
        [
            "--json",
            "metadata",
            "validate",
            "--offline-metadata-file",
            str(metadata),
            "--arrow-file",
            str(arrow),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["status"] == "ready"

    card = tmp_path / "BULL.md"
    result = runner.invoke(
        app,
        [
            "--json",
            "metadata",
            "render-card",
            "--offline-metadata-file",
            str(metadata),
            "--output",
            str(card),
        ],
    )
    assert result.exit_code == 0
    content = card.read_text()
    assert "The strict JSON is authoritative" in content
    assert "<!-- TODO" not in content


def test_status_output(lotsa_store):
    result = runner.invoke(app, ["--json", "metadata", "status", "--phase", "6"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) == 2
    names = {d["dataset_name"] for d in data}
    assert names == {"bull", "elf"}
    bull = next(d for d in data if d["dataset_name"] == "bull")
    assert bull["dataset_id"] == "lotsa_bull"


def test_status_phase_filter(lotsa_store):
    result = runner.invoke(app, ["--json", "metadata", "status", "--phase", "4"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data) == 0


def test_list_includes_arrow_only_dataset(lotsa_store):
    arrow_dir = lotsa_store / "data" / "elecdemand"
    arrow_dir.mkdir(parents=True)
    _write_arrow(arrow_dir / "data.arrow")

    result = runner.invoke(app, ["--json", "metadata", "list"])

    assert result.exit_code == 0
    assert "elecdemand" in json.loads(result.stdout)

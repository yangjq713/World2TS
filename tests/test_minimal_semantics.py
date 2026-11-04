from __future__ import annotations

import json
from pathlib import Path

import pytest

from world2ts_core.minimal_semantics import export_agent_authored_minimal_semantics


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "dataset-refined-minimal"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "unified_event_series_samples.jsonl",
        [
            {"series_id": "T1", "event_links": [{"event_id": "event_1"}]},
            {"series_id": "T2", "event_links": []},
        ],
    )
    _write_jsonl(
        run_dir / "unified_events.jsonl",
        [{"event_id": "event_1", "timestamp": "2020-01-02T00:00:00", "summary": "Event one."}],
    )
    dataset_file = tmp_path / "agent_dataset_semantics.json"
    _write_json(
        dataset_file,
        {
            "dataset_id": "dataset",
            "description": "This hourly dataset records a verified target for two example entities.",
            "domain": "energy",
            "frequency_phrases": ["hourly"],
        },
    )
    series_file = tmp_path / "agent_series_semantics.jsonl"
    _write_jsonl(
        series_file,
        [
            {
                "series_id": "T1",
                "description": "This hourly series records the verified target for example entity T1.",
                "time_range": {"start": "2020-01-01T00:00:00", "end": "2020-01-03T00:00:00"},
                "frequency_phrase": "hourly",
            },
            {
                "series_id": "T2",
                "description": "This hourly series records the verified target for example entity T2.",
                "time_range": {"start": "2020-01-01T00:00:00", "end": "2020-01-03T00:00:00"},
                "frequency_phrase": "hourly",
            },
        ],
    )
    return run_dir, dataset_file, series_file


def test_export_agent_authored_minimal_semantics(tmp_path: Path) -> None:
    run_dir, dataset_file, series_file = _fixture(tmp_path)

    result = export_agent_authored_minimal_semantics(
        run_dir=run_dir,
        agent_dataset_semantics_file=dataset_file,
        agent_series_semantics_file=series_file,
    )

    assert result["num_series"] == 2
    assert result["num_event_links"] == 1
    assert not (run_dir / "unified_event_series_samples.jsonl").exists()
    dataset = json.loads((run_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert set(dataset) == {"dataset_id", "description", "domain"}
    rows = [
        json.loads(line)
        for line in (run_dir / "unified_event_series.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["series_id"] for row in rows] == ["T1", "T2"]
    assert all(set(row) == {"series_id", "description", "time_range", "event_links"} for row in rows)
    assert rows[0]["event_links"] == [{"event_id": "event_1"}]
    assert rows[1]["event_links"] == []


def test_export_rejects_missing_frequency_without_changing_run(tmp_path: Path) -> None:
    run_dir, dataset_file, series_file = _fixture(tmp_path)
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    dataset["description"] = "This dataset records a verified target for two example entities."
    _write_json(dataset_file, dataset)

    with pytest.raises(ValueError, match="frequency phrase"):
        export_agent_authored_minimal_semantics(
            run_dir=run_dir,
            agent_dataset_semantics_file=dataset_file,
            agent_series_semantics_file=series_file,
        )

    assert (run_dir / "unified_event_series_samples.jsonl").exists()
    assert not (run_dir / "dataset_info.json").exists()
    assert not (run_dir / "unified_event_series.jsonl").exists()


def test_export_rejects_incomplete_agent_series_set(tmp_path: Path) -> None:
    run_dir, dataset_file, series_file = _fixture(tmp_path)
    first_line = series_file.read_text(encoding="utf-8").splitlines()[0]
    series_file.write_text(first_line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="IDs do not match"):
        export_agent_authored_minimal_semantics(
            run_dir=run_dir,
            agent_dataset_semantics_file=dataset_file,
            agent_series_semantics_file=series_file,
        )

    assert (run_dir / "unified_event_series_samples.jsonl").exists()

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from world2ts_core.offline_discover.readiness import (
    frequency_phrase_matches,
    prepare_data_readiness,
    validate_semantic_readiness,
)


def _write_arrow(path: Path, rows: list[dict]) -> Path:
    table = pa.Table.from_pylist(rows)
    with path.open("wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return path


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _metadata(item_ids: list[str]) -> dict:
    return {
        "schema_version": "2.0",
        "dataset_semantics": {
            "dataset_name": "readiness_fixture",
            "category": "energy",
            "subcategory": "electricity demand",
            "geo": "Example region",
            "providers": ["Example operator"],
            "time_granularity": "H",
            "time_range": {
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-01T02:00:00",
            },
            "num_series": len(item_ids),
            "description_short": "Hourly electricity demand series.",
            "description_long": "Hourly electricity demand series for readiness testing.",
        },
        "item_schema": {
            "entity_type": "demand_series",
            "item_id_pattern": "{name}",
            "spatial_dimension": "region",
            "target": [{"name": "demand", "description": "Demand.", "unit": "MW"}],
            "past_feat_dynamic_real": [],
            "static_features": [],
            "item_semantics": {
                item_id: {
                    "entity_label": f"Region {item_id}",
                    "description": f"Hourly demand for region {item_id}.",
                    "location": "Example region",
                    "confidence": 1.0,
                    "dimensions": {"region": item_id, "provider": "Example operator"},
                }
                for item_id in item_ids
            },
        },
        "item_groups": [],
        "query_hints": {"dataset_terms": [], "event_bias": []},
    }


def _rows(item_ids: list[str], *, freq: str = "H", target: list[float] | None = None) -> list[dict]:
    values = [1.0, 2.0, 3.0] if target is None else target
    return [
        {
            "item_id": item_id,
            "start": datetime(2020, 1, 1),
            "freq": freq,
            "target": values,
        }
        for item_id in item_ids
    ]


def _prepare(tmp_path: Path, item_ids: list[str]):
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(item_ids))
    metadata = _write_json(tmp_path / "metadata.json", _metadata(item_ids))
    output = tmp_path / "run"
    report = prepare_data_readiness(
        dataset_id="readiness_fixture",
        arrow_files=[arrow],
        metadata_file=metadata,
        output_dir=output,
    )
    return report, output


def test_data_readiness_preserves_physical_order_and_profiles_dimensions(tmp_path: Path) -> None:
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        _rows(["B", "A"], target=[1.0, float("nan"), 3.0]),
    )
    metadata = _write_json(tmp_path / "metadata.json", _metadata(["A", "B"]))

    report = prepare_data_readiness(
        dataset_id="readiness_fixture",
        arrow_files=[arrow],
        metadata_file=metadata,
        output_dir=tmp_path / "run",
    )

    assert report.status == "ready"
    assert report.series_ids == ["B", "A"]
    assert report.series_profiles[0].end == datetime(2020, 1, 1, 2)
    assert report.series_profiles[0].has_explicit_timestamps is False
    assert report.series_profiles[0].missing_rate == pytest.approx(1 / 3)
    assert {profile.dimension for profile in report.dimensions} == {"provider", "region"}
    assert (tmp_path / "run" / "series_semantic_dimensions.jsonl").is_file()
    assert (tmp_path / "run" / "data_readiness_report.json").is_file()
    assert (tmp_path / "run" / "agent_semantic_briefing.json").is_file()


def test_data_readiness_uses_explicit_irregular_timestamps(tmp_path: Path) -> None:
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [
            {
                "item_id": "A",
                "start": datetime(2020, 1, 1, tzinfo=timezone.utc),
                "freq": "D",
                "target": [1.0, 2.0],
                "timestamps": [
                    datetime(2020, 1, 1, tzinfo=timezone.utc),
                    datetime(2020, 1, 8, tzinfo=timezone.utc),
                ],
            }
        ],
    )
    metadata = _metadata(["A"])
    metadata["dataset_semantics"]["time_granularity"] = "D"
    metadata["dataset_semantics"]["time_range"]["end"] = "2020-01-08T00:00:00"
    metadata_path = _write_json(tmp_path / "metadata.json", metadata)

    report = prepare_data_readiness(
        dataset_id="readiness_fixture",
        arrow_files=[arrow],
        metadata_file=metadata_path,
        output_dir=tmp_path / "run",
    )

    assert report.status == "ready"
    assert report.series_profiles[0].end == datetime(2020, 1, 8)
    assert report.series_profiles[0].has_explicit_timestamps is True


def test_data_readiness_rejects_misaligned_explicit_timestamps(tmp_path: Path) -> None:
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [
            {
                "item_id": "A",
                "start": datetime(2020, 1, 1, tzinfo=timezone.utc),
                "freq": "H",
                "target": [1.0, 2.0, 3.0],
                "timestamps": [
                    datetime(2020, 1, 1, tzinfo=timezone.utc),
                    datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
                ],
            }
        ],
    )
    metadata_path = _write_json(tmp_path / "metadata.json", _metadata(["A"]))

    with pytest.raises(
        ValueError,
        match="timestamp count 2 does not match target length 3",
    ):
        prepare_data_readiness(
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata_path,
            output_dir=tmp_path / "run",
        )


def test_data_readiness_rejects_naive_explicit_timestamps(tmp_path: Path) -> None:
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [
            {
                "item_id": "A",
                "start": datetime(2020, 1, 1),
                "freq": "D",
                "target": [1.0, 2.0],
                "timestamps": [datetime(2020, 1, 1), datetime(2020, 1, 8)],
            }
        ],
    )
    metadata_path = _write_json(tmp_path / "metadata.json", _metadata(["A"]))

    with pytest.raises(ValueError, match=r"list<timestamp\[\.\.\., UTC\]>"):
        prepare_data_readiness(
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata_path,
            output_dir=tmp_path / "run",
        )


def test_data_readiness_rejects_metadata_arrow_id_mismatch(tmp_path: Path) -> None:
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(["A"]))
    metadata = _write_json(tmp_path / "metadata.json", _metadata(["B"]))

    with pytest.raises(ValueError, match="Arrow example IDs do not match"):
        prepare_data_readiness(
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata,
            output_dir=tmp_path / "run",
        )


def test_data_readiness_rejects_duplicate_series_ids(tmp_path: Path) -> None:
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(["A", "A"]))
    metadata = _write_json(tmp_path / "metadata.json", _metadata(["A"]))

    with pytest.raises(ValueError, match="Duplicate item_id"):
        prepare_data_readiness(
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata,
            output_dir=tmp_path / "run",
        )


def test_data_readiness_rejects_unsupported_frequency(tmp_path: Path) -> None:
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(["A"], freq="fortnightly"))
    metadata = _write_json(tmp_path / "metadata.json", _metadata(["A"]))

    with pytest.raises(ValueError, match="Unsupported fixed frequency"):
        prepare_data_readiness(
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata,
            output_dir=tmp_path / "run",
        )


def test_data_readiness_reports_empty_target_before_failing(tmp_path: Path) -> None:
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(["A"], target=[]))
    metadata = _metadata(["A"])
    metadata["dataset_semantics"]["time_range"]["end"] = "2020-01-01T00:00:00"
    metadata_path = _write_json(tmp_path / "metadata.json", metadata)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="empty target"):
        prepare_data_readiness(
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata_path,
            output_dir=output,
        )

    report = json.loads((output / "data_readiness_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert not (output / "agent_semantic_briefing.json").exists()


def test_data_readiness_rejects_unrecoverable_series_semantics(tmp_path: Path) -> None:
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(["A"]))
    metadata = _metadata(["A"])
    metadata["item_schema"]["item_semantics"]["A"] = {
        "entity_label": "",
        "description": "",
        "location": "",
        "confidence": 0.0,
        "dimensions": {},
    }
    metadata_path = _write_json(tmp_path / "metadata.json", metadata)

    with pytest.raises(ValueError, match="unrecoverable series semantics"):
        prepare_data_readiness(
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata_path,
            output_dir=tmp_path / "run",
        )


def test_data_readiness_validates_target_channel_count(tmp_path: Path) -> None:
    arrow = _write_arrow(tmp_path / "data.arrow", _rows(["A"]))
    metadata = _metadata(["A"])
    metadata["item_schema"]["target"].append(
        {"name": "temperature", "description": "Temperature.", "unit": "C"}
    )
    metadata_path = _write_json(tmp_path / "metadata.json", metadata)

    with pytest.raises(ValueError, match="target_channels"):
        prepare_data_readiness(
            dataset_id="readiness_fixture",
            arrow_files=[arrow],
            metadata_file=metadata_path,
            output_dir=tmp_path / "run",
        )


def test_data_readiness_normalizes_expanded_target_dimensions(tmp_path: Path) -> None:
    arrow = _write_arrow(
        tmp_path / "data.arrow",
        [
            {
                "item_id": "A",
                "start": datetime(2020, 1, 1),
                "freq": "H",
                "target": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            }
        ],
    )
    metadata = _metadata(["A"])
    metadata["item_schema"]["target"] = [
        {"name": "demand", "description": "Demand.", "unit": "MW"},
        {"name": "temperature", "description": "", "unit": ""},
    ]
    metadata_path = _write_json(tmp_path / "metadata.json", metadata)
    output = tmp_path / "run"

    report = prepare_data_readiness(
        dataset_id="readiness_fixture",
        arrow_files=[arrow],
        metadata_file=metadata_path,
        output_dir=output,
    )

    assert report.series_ids == ["A__target_0", "A__target_1"]
    rows = [
        json.loads(line)
        for line in (output / "series_semantic_dimensions.jsonl").read_text().splitlines()
    ]
    assert rows[0]["dimensions"]["target_name"] == "demand"
    assert rows[0]["dimensions"]["target_unit"] == "MW"
    assert rows[1]["dimensions"]["target_name"] == "temperature"
    assert "target_unit" not in rows[1]["dimensions"]


def test_semantic_readiness_checks_agent_frequency_and_exact_time_range(tmp_path: Path) -> None:
    report, output = _prepare(tmp_path, ["A"])
    dataset_file = _write_json(
        tmp_path / "agent_dataset_semantics.json",
        {
            "dataset_id": "readiness_fixture",
            "description": "This hourly dataset records verified electricity demand for an example region.",
            "domain": "energy",
            "frequency_phrases": ["hourly"],
        },
    )
    series_file = tmp_path / "agent_series_semantics.jsonl"
    series_file.write_text(
        json.dumps(
            {
                "series_id": "A",
                "description": "This hourly series records verified electricity demand for region A.",
                "time_range": {
                    "start": "2020-01-01T00:00:00",
                    "end": "2020-01-01T02:00:00",
                },
                "frequency_phrase": "hourly",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    semantic_report = validate_semantic_readiness(
        data_readiness=report,
        agent_dataset_semantics_file=dataset_file,
        agent_series_semantics_file=series_file,
        output_dir=output,
    )

    assert semantic_report.status == "ready"
    assert semantic_report.validated_series_ids == ["A"]

    valid = json.loads(series_file.read_text(encoding="utf-8"))
    offset_equivalent = json.loads(json.dumps(valid))
    offset_equivalent["time_range"] = {
        "start": "2020-01-01T08:00:00+08:00",
        "end": "2020-01-01T10:00:00+08:00",
    }
    series_file.write_text(
        json.dumps(offset_equivalent) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="time_range"):
        validate_semantic_readiness(
            data_readiness=report,
            agent_dataset_semantics_file=dataset_file,
            agent_series_semantics_file=series_file,
            output_dir=output,
        )

    invalid = json.loads(json.dumps(valid))
    invalid["time_range"]["end"] = "2020-01-01T01:00:00"
    series_file.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="time_range"):
        validate_semantic_readiness(
            data_readiness=report,
            agent_dataset_semantics_file=dataset_file,
            agent_series_semantics_file=series_file,
            output_dir=output,
        )
    blocked = json.loads((output / "semantic_readiness_report.json").read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"


@pytest.mark.parametrize(
    ("freq", "phrase"),
    [("30min", "half-hourly"), ("5min", "five-minute")],
)
def test_frequency_phrases_accept_common_subhourly_wording(
    freq: str,
    phrase: str,
) -> None:
    assert frequency_phrase_matches(freq, phrase)

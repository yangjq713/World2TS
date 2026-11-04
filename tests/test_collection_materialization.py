from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

import world2ts_core.collection as collection
from world2ts_core.cli import app
from world2ts_core.bootstrap import ensure_sources_registered
from world2ts_core.freq_routing import build_adapter, constructor_options_for_source
from world2ts_core.series.model import SeriesRecord
from world2ts_core.offline_discover.io import load_arrow_examples
from world2ts_core.offline_discover.projection import _event_overlaps_profile
from world2ts_core.offline_discover.readiness import prepare_data_readiness
from world2ts_core.offline_discover.schema import parse_offline_metadata
from world2ts_core.registry import series_source_classes_by_name
from tests.conftest import make_cli_runner


def _plan_payload(*, symbols: list[str] | None = None) -> dict:
    return {
        "schema_version": "1.0",
        "dataset_id": "semiconductor_equities_daily",
        "source": "yahoo",
        "domain": "semiconductors",
        "symbols": symbols or ["NVDA"],
        "start": "2024-01-01",
        "end": "2024-01-15",
        "frequency": "1d",
        "timezone": "America/New_York",
        "adapter_options": {},
        "metadata": {
            "subcategory": "semiconductor equity prices",
            "geo": "United States",
            "description_short": "Daily semiconductor equity closing prices.",
            "description_long": (
                "Daily closing prices for selected semiconductor companies with "
                "original exchange-session timestamps; trading-session gaps are "
                "preserved and only the primary close target is materialized."
            ),
            "entity_type": "public_company",
            "spatial_dimension": "company_location",
            "target": {
                "source_field": "close",
                "name": "close",
                "description": "Unadjusted closing market price.",
                "unit": "USD",
            },
            "dimensions": ["ticker", "exchange"],
            "query_hints": {
                "dataset_terms": ["semiconductor equities"],
                "event_bias": ["earnings", "export controls"],
            },
        },
    }


def _record(
    symbol: str = "NVDA",
    *,
    timestamps: list[datetime] | None = None,
    metadata: dict | None = None,
) -> SeriesRecord:
    observation_times = timestamps or [
        datetime(2024, 1, 5, 16),
        datetime(2024, 1, 8, 16),
    ]
    values = [490.0 + index for index in range(len(observation_times))]
    return SeriesRecord(
        series_id=symbol,
        domain="semiconductors",
        entity="NVIDIA Corporation",
        geo="Santa Clara, CA, United States",
        freq="1d",
        start=observation_times[0],
        end=observation_times[-1],
        values=values,
        timestamps=observation_times,
        columns={
            "close": [999.0] * len(values),
            "volume": [1_000_000.0] * len(values),
        },
        metadata=metadata
        or {
            "source": "yahoo_finance",
            "ticker": symbol,
            "exchange": "NMS",
        },
    )


class _Adapter:
    def __init__(self, records: list[SeriesRecord]):
        self.records = records

    def fetch(self, symbols, start, end, domain):
        return self.records


def _write_plan(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_collection_plan_is_strict() -> None:
    payload = _plan_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        collection.CollectionPlan.model_validate(payload)

    payload = _plan_payload(symbols=["NVDA", "NVDA"])
    with pytest.raises(ValueError, match="symbols must be unique"):
        collection.CollectionPlan.model_validate(payload)

    payload = _plan_payload()
    payload["frequency"] = "1mo"
    with pytest.raises(ValueError, match="Input should be"):
        collection.CollectionPlan.model_validate(payload)

    payload = _plan_payload()
    payload["metadata"]["data_quality"] = {"time_axis": "Trading sessions only."}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        collection.CollectionPlan.model_validate(payload)

    for field_path in (
        ("adapter_options",),
        ("metadata", "dimensions"),
        ("metadata", "query_hints"),
        ("metadata", "target", "unit"),
        ("metadata", "target", "source_field"),
    ):
        payload = _plan_payload()
        parent = payload
        for key in field_path[:-1]:
            parent = parent[key]
        del parent[field_path[-1]]
        with pytest.raises(ValueError, match="Field required"):
            collection.CollectionPlan.model_validate(payload)


def test_collection_capabilities_have_exact_constructor_mappings() -> None:
    ensure_sources_registered()

    for source_name, source_class in series_source_classes_by_name().items():
        collection_frequencies = (
            set(source_class.capability.frequencies)
            & collection.COLLECTION_FREQUENCIES
        )
        for frequency in collection_frequencies:
            options = constructor_options_for_source(source_name, frequency)
            adapter = build_adapter(source_name, options)
            assert isinstance(adapter, source_class)


def test_target_source_field_must_match_adapter_contract(monkeypatch) -> None:
    payload = _plan_payload()
    payload["metadata"]["target"]["source_field"] = "volume"
    plan = collection.CollectionPlan.model_validate(payload)

    with pytest.raises(ValueError, match="source_field must be 'close'"):
        collection._build_exact_adapter(plan)


@pytest.mark.parametrize("source", ["coingecko", "owid", "polymarket"])
def test_utc_civil_sources_require_utc_plan_timezone(source: str) -> None:
    payload = _plan_payload()
    payload["source"] = source
    plan = collection.CollectionPlan.model_validate(payload)

    with pytest.raises(ValueError, match="requires plan timezone 'UTC'"):
        collection._build_exact_adapter(plan)


def test_load_collection_plan_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text('{"schema_version":"1.0","schema_version":"1.0"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        collection.load_collection_plan(path)


def test_exact_adapter_uses_agent_source_without_domain_routing(monkeypatch) -> None:
    plan = collection.CollectionPlan.model_validate(_plan_payload())
    captured: dict = {}
    sentinel = object()

    def fake_build_adapter(source_name: str, options: dict):
        captured.update(source=source_name, options=options)
        return sentinel

    monkeypatch.setattr(collection, "build_adapter", fake_build_adapter)

    assert collection._build_exact_adapter(plan) is sentinel
    assert captured == {"source": "yahoo", "options": {"interval": "1d"}}


def test_materialize_collection_is_directly_offline_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = collection.CollectionPlan.model_validate(_plan_payload())
    monkeypatch.setattr(
        collection,
        "_build_exact_adapter",
        lambda _: _Adapter([_record()]),
    )
    output_dir = tmp_path / "collected"

    summary = collection.materialize_collection(plan, output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "data.arrow",
        "offline_metadata.json",
    }
    assert summary["item_count"] == 1
    assert summary["target_count"] == 1

    with pa.memory_map(str(output_dir / "data.arrow"), "r") as source:
        table = ipc.open_file(source).read_all()
    assert table.schema.names == ["item_id", "start", "freq", "target", "timestamps"]
    assert table.column("target")[0].as_py() == [490.0, 491.0]
    assert table.column("timestamps")[0].as_py() == [
        datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
        datetime(2024, 1, 8, 21, tzinfo=timezone.utc),
    ]
    loaded = load_arrow_examples([output_dir / "data.arrow"])
    assert loaded[0].timestamps == [
        datetime(2024, 1, 5, 21),
        datetime(2024, 1, 8, 21),
    ]

    raw_metadata = json.loads((output_dir / "offline_metadata.json").read_text())
    assert set(raw_metadata) == {
        "schema_version",
        "dataset_semantics",
        "item_schema",
        "item_groups",
        "query_hints",
    }
    assert set(raw_metadata["dataset_semantics"]) == {
        "dataset_name",
        "category",
        "subcategory",
        "geo",
        "providers",
        "time_granularity",
        "time_range",
        "num_series",
        "description_short",
        "description_long",
    }
    assert set(raw_metadata["item_schema"]) == {
        "entity_type",
        "item_id_pattern",
        "spatial_dimension",
        "target",
        "past_feat_dynamic_real",
        "static_features",
        "item_semantics",
    }
    raw_item = raw_metadata["item_schema"]["item_semantics"]["NVDA"]
    assert set(raw_item) == {
        "entity_label",
        "description",
        "location",
        "confidence",
        "attributes",
    }
    assert "dimensions" not in raw_item
    assert raw_item["attributes"]["ticker"] == "NVDA"
    metadata = parse_offline_metadata(raw_metadata, output_dir / "offline_metadata.json")
    assert metadata.dataset_semantics.dataset_name == plan.dataset_id
    assert metadata.dataset_semantics.category == "semiconductors"
    assert metadata.dataset_semantics.time_range.end == datetime(
        2024, 1, 8, 21, tzinfo=timezone.utc
    )
    assert len(metadata.item_schema.target) == 1
    assert (
        metadata.item_schema.item_semantics["NVDA"].dimensions["ticker"]
        == "NVDA"
    )

    report = prepare_data_readiness(
        dataset_id=plan.dataset_id,
        arrow_files=[output_dir / "data.arrow"],
        metadata_file=output_dir / "offline_metadata.json",
        output_dir=tmp_path / "offline_run",
    )
    assert report.status == "ready"
    assert report.series_profiles[0].end == datetime(2024, 1, 8, 21)
    assert _event_overlaps_profile(
        datetime(2024, 1, 6, tzinfo=timezone.utc),
        None,
        report.series_profiles[0],
    )


@pytest.mark.parametrize(
    "records,error",
    [
        ([], "missing=\\['NVDA'\\]"),
        (
            [
                _record(
                    timestamps=[
                        datetime(2024, 1, 8, 16),
                        datetime(2024, 1, 5, 16),
                    ]
                )
            ],
            "strictly increasing",
        ),
        (
            [
                _record(
                    timestamps=[
                        datetime(2024, 1, 14, 16),
                        datetime(2024, 1, 15),
                    ]
                )
            ],
            "outside the requested range",
        ),
        (
            [
                _record(
                    metadata={
                        "source": "yahoo_finance",
                        "ticker": "NVDA",
                    }
                )
            ],
            "missing requested metadata dimension 'exchange'",
        ),
    ],
)
def test_materialize_collection_failure_is_atomic(
    monkeypatch,
    tmp_path: Path,
    records: list[SeriesRecord],
    error: str,
) -> None:
    plan = collection.CollectionPlan.model_validate(_plan_payload())
    monkeypatch.setattr(collection, "_build_exact_adapter", lambda _: _Adapter(records))
    output_dir = tmp_path / "collected"

    with pytest.raises(ValueError, match=error):
        collection.materialize_collection(plan, output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".collected.collect-*"))


def test_noaa_isd_receives_plan_window_as_utc_instants(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = _plan_payload(symbols=["72505394728"])
    payload.update(
        source="noaa_isd",
        domain="weather",
        frequency="1h",
        adapter_options={"field": "TMP"},
    )
    payload["metadata"].update(
        subcategory="hourly station weather",
        entity_type="weather_station",
        spatial_dimension="station_location",
        target={
            "source_field": "tmp",
            "name": "temperature",
            "description": "Hourly air temperature.",
            "unit": "degrees Celsius",
        },
        dimensions=[],
    )
    plan = collection.CollectionPlan.model_validate(payload)
    captured: dict[str, datetime] = {}

    class NoaaAdapter:
        def fetch(self, symbols, start, end, domain):
            captured.update(start=start, end=end)
            timestamps = [
                datetime(2024, 1, 5, 20, tzinfo=timezone.utc),
                datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
            ]
            return [
                SeriesRecord(
                    series_id=symbols[0],
                    domain=domain,
                    entity="Station 72505394728",
                    geo="New York, United States",
                    freq="1h",
                    start=timestamps[0],
                    end=timestamps[-1],
                    values=[2.0, 1.5],
                    timestamps=timestamps,
                    metadata={"source": "noaa_isd"},
                )
            ]

    monkeypatch.setattr(collection, "_build_exact_adapter", lambda _: NoaaAdapter())

    collection.materialize_collection(plan, tmp_path / "collected")

    assert captured == {
        "start": datetime(2024, 1, 1, 5, tzinfo=timezone.utc),
        "end": datetime(2024, 1, 15, 5, tzinfo=timezone.utc),
    }


def test_collect_cli_outputs_one_json_summary(monkeypatch, tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json", _plan_payload())
    output_dir = tmp_path / "collected"
    monkeypatch.setattr(
        collection,
        "_build_exact_adapter",
        lambda _: _Adapter([_record()]),
    )
    runner = make_cli_runner(mix_stderr=False)

    result = runner.invoke(
        app,
        [
            "--json",
            "collect",
            "--plan",
            str(plan_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stderr
    summary = json.loads(result.output)
    assert summary["dataset_id"] == "semiconductor_equities_daily"
    assert summary["arrow_file"] == str(output_dir / "data.arrow")
    assert summary["offline_metadata_file"] == str(output_dir / "offline_metadata.json")


def test_collect_cli_failure_leaves_no_output(monkeypatch, tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json", _plan_payload())
    output_dir = tmp_path / "collected"
    monkeypatch.setattr(collection, "_build_exact_adapter", lambda _: _Adapter([]))
    runner = make_cli_runner(mix_stderr=False)

    result = runner.invoke(
        app,
        [
            "--json",
            "collect",
            "--plan",
            str(plan_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 1
    assert result.output == ""
    assert "missing=['NVDA']" in result.stderr
    assert not output_dir.exists()

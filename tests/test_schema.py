from __future__ import annotations

from pathlib import Path

import pytest

from world2ts_core.offline_discover.schema import parse_offline_metadata


def test_parse_offline_metadata_accepts_minimal_valid_payload():
    model = parse_offline_metadata(
        {
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
                "num_series": 1,
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
                    "T0": {
                        "entity_label": "T0 Lyft price_mean",
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
                },
            },
            "item_groups": [
                {
                    "group_id": "back_bay__lyft__lux",
                    "label": "Back Bay | Lyft | Lux",
                    "example_ids": ["T0"],
                    "shared_terms": ["Back Bay", "Lyft", "Lux", "rideshare"],
                }
            ],
            "query_hints": {
                "dataset_terms": ["Boston rideshare"],
                "event_bias": ["weather"],
            },
        },
        Path("rideshare.offline_metadata.json"),
    )

    assert model.schema_version == "2.0"
    assert model.dataset_semantics.category == "transport"
    item = model.item_schema.item_semantics["T0"]
    assert item.dimensions["provider_name"] == "Lyft"
    assert "attributes" not in item.model_dump()


def test_parse_offline_metadata_normalizes_dimension_values():
    payload = _minimal_payload()
    payload["item_schema"]["item_semantics"]["T0"]["dimensions"] = {
        " provider_name ": " Lyft ",
        "zones": ["Back Bay", "Back Bay", "Fenway"],
        "range": [1, 2.5],
        "enabled": True,
    }

    model = parse_offline_metadata(payload, Path("metadata.json"))

    assert model.item_schema.item_semantics["T0"].dimensions == {
        "provider_name": "Lyft",
        "zones": ["Back Bay", "Fenway"],
        "range": [1.0, 2.5],
        "enabled": True,
    }


@pytest.mark.parametrize(
    "bad_value",
    [None, {}, [], ["Boston", 1], float("nan")],
)
def test_parse_offline_metadata_rejects_unsupported_dimension_values(bad_value):
    payload = _minimal_payload()
    payload["item_schema"]["item_semantics"]["T0"]["dimensions"] = {"bad": bad_value}

    with pytest.raises(ValueError, match="dimension"):
        parse_offline_metadata(payload, Path("metadata.json"))


def test_parse_offline_metadata_forbids_unknown_fields():
    payload = _minimal_payload()
    payload["item_schema"]["item_semantics"]["T0"]["dimensons"] = {"city": "Boston"}

    with pytest.raises(ValueError, match="dimensons"):
        parse_offline_metadata(payload, Path("metadata.json"))


def test_parse_offline_metadata_preserves_explicit_data_quality_notes():
    payload = _minimal_payload()
    payload["dataset_semantics"]["_data_quality"] = {
        "timestamp_phase_check": "The source phase was verified against a calendar signal."
    }

    model = parse_offline_metadata(payload, Path("metadata.json"))

    assert model.dataset_semantics.data_quality == payload["dataset_semantics"]["_data_quality"]


def test_parse_offline_metadata_rejects_wrong_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        parse_offline_metadata(
            {
                "schema_version": "1.0",
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
                    "num_series": 1,
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
                        "T0": {
                            "entity_label": "T0 Lyft price_mean",
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
                    },
                },
                "item_groups": [],
                "query_hints": {"dataset_terms": [], "event_bias": []},
            },
            Path("rideshare.offline_metadata.json"),
        )


def _minimal_payload() -> dict:
    return {
        "schema_version": "2.0",
        "dataset_semantics": {
            "dataset_name": "example",
            "category": "transport",
            "subcategory": "mobility",
            "geo": "Boston, USA",
            "providers": ["provider"],
            "time_granularity": "H",
            "time_range": {
                "start": "2024-01-01T00:00:00",
                "end": "2024-01-02T00:00:00",
            },
            "num_series": 1,
            "description_short": "Example series.",
            "description_long": "An example hourly mobility series.",
        },
        "item_schema": {
            "entity_type": "mobility_series",
            "item_id_pattern": "T{n}",
            "spatial_dimension": "city",
            "target": [{"name": "value", "description": "metric", "unit": ""}],
            "past_feat_dynamic_real": [],
            "static_features": [],
            "item_semantics": {
                "T0": {
                    "entity_label": "T0",
                    "description": "Example item.",
                    "location": "Boston",
                    "confidence": 1.0,
                }
            },
        },
        "item_groups": [],
        "query_hints": {"dataset_terms": [], "event_bias": []},
    }

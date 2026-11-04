from __future__ import annotations

import json

import pytest

from world2ts_core.offline_discover import publication
from world2ts_core.offline_discover.publication import publish_minimal
from world2ts_core.offline_discover.projection import project_events
from world2ts_core.offline_discover.quality_audit import apply_quality_review
from tests.test_offline_discover_quality_audit import (
    _decision,
    _event,
    _prepare_report,
    _valid_run,
    _write_jsonl,
)


def _approved_run(tmp_path):
    _valid_run(tmp_path)
    packet, _ = _prepare_report(tmp_path)
    apply_quality_review(
        run_dir=tmp_path,
        decision_file=_decision(tmp_path, packet, "approve"),
    )


def test_publication_is_blocked_before_explicit_quality_approval(tmp_path) -> None:
    _valid_run(tmp_path)

    with pytest.raises(ValueError, match="user_quality_review_required"):
        publish_minimal(tmp_path)

    assert not (tmp_path / "dataset_info.json").exists()
    assert not (tmp_path / "unified_event_series.jsonl").exists()
    assert not (tmp_path / "unified_events.jsonl").exists()


def test_publication_writes_exact_minimal_contract_and_published_state(tmp_path) -> None:
    _valid_run(tmp_path)
    held = _event().model_copy(
        update={
            "event_id": "event_held",
            "summary": (
                "On January 11, 2024, a separate Boston disruption remained "
                "ambiguous in affected scope."
            ),
            "merge_or_split_notes": "The source describes a second episode.",
        }
    )
    _write_jsonl(
        tmp_path / "canonical_events.jsonl",
        [_event(), held],
    )
    project_events(tmp_path)
    (tmp_path / "unified_event_series_samples.jsonl").write_text(
        "legacy\n",
        encoding="utf-8",
    )
    packet, _ = _prepare_report(tmp_path)
    apply_quality_review(
        run_dir=tmp_path,
        decision_file=_decision(tmp_path, packet, "approve"),
    )

    result = publish_minimal(tmp_path)

    dataset = json.loads(
        (tmp_path / "dataset_info.json").read_text(encoding="utf-8")
    )
    series = [
        json.loads(line)
        for line in (tmp_path / "unified_event_series.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    events = [
        json.loads(line)
        for line in (tmp_path / "unified_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))

    assert result["state"] == "published"
    assert set(dataset) == {"dataset_id", "description", "domain"}
    assert [row["series_id"] for row in series] == ["boston", "chicago"]
    assert all(
        set(row) == {"series_id", "description", "time_range", "event_links"}
        for row in series
    )
    assert series[0]["event_links"] == [{"event_id": "event_outage"}]
    assert series[1]["event_links"] == []
    assert events == [
        {
            "event_id": "event_outage",
            "timestamp": "2024-01-10T00:00:00",
            "summary": (
                "On January 10, 2024, an electricity outage disrupted "
                "operations across Boston."
            ),
        }
    ]
    assert state["state"] == "published"
    assert set(state["artifact_sha256"]) == {
        "dataset_info.json",
        "unified_event_series.jsonl",
        "unified_events.jsonl",
    }
    history = [
        json.loads(line)
        for line in (tmp_path / "run_history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert history[-1]["next_state"] == "published"
    assert not (tmp_path / "unified_event_series_samples.jsonl").exists()


def test_publication_rolls_back_all_outputs_if_replace_fails(
    tmp_path,
    monkeypatch,
) -> None:
    _approved_run(tmp_path)
    old = {
        "dataset_info.json": "old dataset\n",
        "unified_event_series.jsonl": "old series\n",
        "unified_events.jsonl": "old events\n",
    }
    for filename, content in old.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")

    real_replace = publication.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        destination = str(destination)
        if (
            not failed
            and destination.endswith("unified_event_series.jsonl")
            and ".publication-" not in destination
        ):
            failed = True
            raise OSError("simulated replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(publication.os, "replace", fail_once)
    with pytest.raises(OSError, match="simulated replace failure"):
        publish_minimal(tmp_path)

    assert {
        filename: (tmp_path / filename).read_text(encoding="utf-8")
        for filename in old
    } == old
    assert not (tmp_path / "run_state.json").exists()

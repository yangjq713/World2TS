from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from world2ts_core.errors import SourceError
from world2ts_core.sources.anysearch import (
    AnySearchSource,
    has_anysearch_api_key_configured,
    resolve_anysearch_api_keys,
)

# Window used across tests.
START = datetime(2015, 3, 10)
END = datetime(2015, 3, 24)

# AnySearch returns no publication date; the in-window date is recovered from the
# /YYYY/MM/DD/ URL path by the shared extractor.
_INWINDOW = {
    "title": "St. Patrick's Day Parade kicks off in New York City",
    "url": "https://news.example.com/2015/03/17/nyc-st-patricks-parade",
    "description": "",
    "content": "The parade marched up Fifth Avenue.",
    "score": 87.0,
}
_OUTWINDOW = {
    "title": "Some January story",
    "url": "https://news.example.com/2015/01/02/old-story",
    "description": "",
    "content": "Unrelated January coverage.",
    "score": 50.0,
}
_NO_DATE = {
    "title": "Evergreen parade info page",
    "url": "https://www.nycstpatricksparade.org/",
    "description": "",
    "content": "About the parade.",
    "score": 60.0,
}


def _envelope(results: list[dict]) -> str:
    return json.dumps({
        "code": 0,
        "message": "success",
        "data": {"results": results, "metadata": {"total_results": len(results)}},
    })


def _search(source, results_or_raw):
    """Run a search with _curl_post mocked and URL enrichment disabled."""
    raw = results_or_raw if isinstance(results_or_raw, str) else _envelope(results_or_raw)
    with patch("world2ts_core.sources.anysearch._curl_post", return_value=raw), \
         patch("world2ts_core.sources.anysearch.fetch_url_publication_timestamp", return_value=(None, "")):
        return source.search(query="NYC St. Patrick parade March 2015", start=START, end=END)


class TestAnySearchSource:
    def test_parses_success_envelope(self):
        events = _search(AnySearchSource(api_key="k"), [_INWINDOW])
        assert len(events) == 1
        ev = events[0]
        assert ev.source_name == "anysearch"
        assert ev.timestamp.date() == datetime(2015, 3, 17).date()
        assert ev.evidence_quality == "web_snippet"

    def test_score_normalized_to_unit_interval(self):
        events = _search(AnySearchSource(api_key="k"), [_INWINDOW])
        assert events[0].retrieval_score == pytest.approx(0.87, abs=1e-6)

    def test_window_post_filter_drops_out_of_window(self):
        events = _search(AnySearchSource(api_key="k"), [_INWINDOW, _OUTWINDOW])
        assert [e.timestamp.date() for e in events] == [datetime(2015, 3, 17).date()]

    def test_undateable_result_dropped(self):
        events = _search(AnySearchSource(api_key="k"), [_NO_DATE])
        assert events == []

    def test_undated_result_recovered_via_url_metadata(self):
        # AnySearch returns no date and the URL has no date path, but the article's
        # HTML publication metadata yields an in-window date -> recovered, not dropped.
        recovered = datetime(2015, 3, 17)
        with patch("world2ts_core.sources.anysearch._curl_post", return_value=_envelope([_NO_DATE])), \
             patch("world2ts_core.sources.anysearch.fetch_url_publication_timestamp",
                   return_value=(recovered, "2015-03-17T10:00:00")):
            events = AnySearchSource(api_key="k").search(query="q", start=START, end=END)
        assert len(events) == 1
        assert events[0].timestamp == recovered
        assert events[0].timestamp_precision == "exact"
        assert events[0].structured_payload.get("date_source") == "url_metadata"

    def test_error_envelope_raises(self):
        envelope = json.dumps({"code": 50304, "message": "temporarily unavailable", "data": {}})
        with pytest.raises(SourceError, match="AnySearch error"):
            _search(AnySearchSource(api_key="k"), envelope)

    def test_missing_data_key_raises(self):
        with pytest.raises(SourceError):
            _search(AnySearchSource(api_key="k"), json.dumps({"code": 0, "message": "success"}))

    def test_invalid_json_raises(self):
        with pytest.raises(SourceError, match="valid JSON"):
            _search(AnySearchSource(api_key="k"), "<html>not json</html>")

    def test_total_failure_raises_not_empty(self):
        with patch("world2ts_core.sources.anysearch._curl_post", side_effect=SourceError("HTTP 432 quota_exhausted")):
            with pytest.raises(SourceError):
                AnySearchSource(api_key="k").search(query="q", start=START, end=END)

    def test_transient_503_retries_then_succeeds(self):
        calls = {"n": 0}

        def fake_post(body, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                raise SourceError("HTTP 503 from https://api.anysearch.com/v1/search: 50304")
            return _envelope([_INWINDOW])

        with patch("world2ts_core.sources.anysearch._curl_post", side_effect=fake_post), \
             patch("world2ts_core.sources.anysearch.fetch_url_publication_timestamp", return_value=(None, "")), \
             patch("world2ts_core.sources.anysearch.time.sleep"):
            events = AnySearchSource(api_key="k").search(query="q", start=START, end=END)
        assert calls["n"] == 2
        assert len(events) == 1

    def test_failover_to_next_key_on_invalid(self):
        calls = {"n": 0}

        def fake_post(body, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                raise SourceError("HTTP 401 unauthorized")
            return _envelope([_INWINDOW])

        with patch("world2ts_core.sources.anysearch._curl_post", side_effect=fake_post), \
             patch("world2ts_core.sources.anysearch.fetch_url_publication_timestamp", return_value=(None, "")), \
             patch("world2ts_core.sources.anysearch.time.sleep"):
            events = AnySearchSource(api_keys=["k1", "k2"]).search(query="q", start=START, end=END)
        assert calls["n"] == 2
        assert len(events) == 1

    def test_auth_header_sent_when_key_present(self):
        captured = {}

        def fake_post(body, headers):
            captured["headers"] = headers
            return _envelope([_INWINDOW])

        with patch("world2ts_core.sources.anysearch._curl_post", side_effect=fake_post), \
             patch("world2ts_core.sources.anysearch.fetch_url_publication_timestamp", return_value=(None, "")):
            AnySearchSource(api_key="secret-key").search(query="q", start=START, end=END)
        assert captured["headers"] == ["Authorization: Bearer secret-key"]

    def test_anonymous_mode_sends_no_auth_header(self, tmp_path, monkeypatch):
        # chdir away from the repo so the real anysearch.toml is not picked up
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
        monkeypatch.delenv("ANYSEARCH_API_KEYS", raising=False)
        captured = {}

        def fake_post(body, headers):
            captured["headers"] = headers
            return _envelope([_INWINDOW])

        with patch("world2ts_core.sources.anysearch._curl_post", side_effect=fake_post), \
             patch("world2ts_core.sources.anysearch.fetch_url_publication_timestamp", return_value=(None, "")):
            src = AnySearchSource()
            assert src._key_pool.anonymous is True
            src.search(query="q", start=START, end=END)
        assert captured["headers"] == []


class TestAnySearchKeyResolution:
    def test_resolve_from_env(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ANYSEARCH_API_KEY", "env-key-123")
        assert resolve_anysearch_api_keys() == ["env-key-123"]
        assert has_anysearch_api_key_configured() is True

    def test_anonymous_when_nothing_configured(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
        monkeypatch.delenv("ANYSEARCH_API_KEYS", raising=False)
        assert resolve_anysearch_api_keys() == []
        assert has_anysearch_api_key_configured() is False

    def test_reads_root_anysearch_toml(self, tmp_path, monkeypatch):
        (tmp_path / "anysearch.toml").write_text(
            'api_keys = ["file-key-1", "file-key-2"]\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
        monkeypatch.delenv("ANYSEARCH_API_KEYS", raising=False)
        assert resolve_anysearch_api_keys() == ["file-key-1", "file-key-2"]

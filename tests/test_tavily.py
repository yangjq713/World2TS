"""Tests for Tavily web-search source adapter."""

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from world2ts_core.errors import SourceError
from world2ts_core.sources.tavily import TavilySource
from world2ts_core.events import DiscoveredEvent

_MOCK_RESPONSE = """{
    "results": [
        {
            "title": "OPEC Cuts Oil Output by 2 Million Barrels Per Day",
            "url": "https://reuters.com/opec-cut-2024",
            "content": "OPEC announced a surprise cut of 2 million barrels per day, sending oil prices up 5%.",
            "score": 0.95,
            "published_date": "2024-10-05"
        },
        {
            "title": "Oil Markets React to OPEC Decision",
            "url": "https://bloomberg.com/opec-reaction-2024",
            "content": "Markets reacted sharply as OPEC announced production cuts.",
            "score": 0.87,
            "published_date": "2024-10-05"
        }
    ]
}"""

_MOCK_EMPTY = '{"results": []}'

_MOCK_RESPONSE_WITH_URL_DATE = """{
    "results": [
        {
            "title": "Beijing takes action to ease its thick traffic congestion",
            "url": "https://www.chinadaily.com.cn/business/motoring/2016-01/04/content_22920572.htm",
            "content": "City authorities are working on solutions to ease traffic pressure.",
            "score": 0.91
        }
    ]
}"""

_MOCK_RESPONSE_WITHOUT_DATE = """{
    "results": [
        {
            "title": "Beijing Subway",
            "url": "https://en.wikipedia.org/wiki/Beijing_Subway",
            "content": "The Beijing Subway is the rapid transit system of Beijing Municipality.",
            "score": 0.95
        }
    ]
}"""

_MOCK_RESPONSE_WITH_UNLABELED_TEXT_DATE = """{
    "results": [
        {
            "title": "2015 Annual Airport Traffic Report",
            "url": "https://example.com/airport-report.pdf",
            "content": "The report includes passenger traffic tables for 2016-04-01 and other dates.",
            "score": 0.91
        }
    ]
}"""

_MOCK_RESPONSE_WITH_LABELED_RAW_DATE = """{
    "results": [
        {
            "title": "Beijing takes action to ease traffic congestion",
            "url": "https://example.com/beijing-traffic",
            "content": "City authorities are working on solutions to ease traffic pressure.",
            "raw_content": "Updated: 2016-01-04 10:17. City authorities are working on solutions.",
            "score": 0.91
        }
    ]
}"""

_MOCK_RESPONSE_WITH_EXACT_PUBLISHED_AT = """{
    "results": [
        {
            "title": "Beijing traffic update",
            "url": "https://example.com/beijing-traffic-exact",
            "content": "Traffic update with exact publication time.",
            "score": 0.91,
            "published_at": "2016-01-04T10:17:00Z"
        }
    ]
}"""

_MOCK_RESPONSE_WITH_DATE_AND_EXACT_PUBLISHED_AT = """{
    "results": [
        {
            "title": "Beijing traffic update",
            "url": "https://example.com/beijing-traffic-exact",
            "content": "Traffic update with exact publication time.",
            "score": 0.91,
            "published_date": "2016-01-04",
            "published_at": "2016-01-04T10:17:00Z"
        }
    ]
}"""


class TestTavilySource:

    def test_search_returns_discovered_events(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE):
            src = TavilySource(api_key="test-key")
            events = src.search(
                "OPEC oil cut",
                start=datetime(2024, 10, 1),
                end=datetime(2024, 10, 10),
            )
        assert len(events) == 2
        assert all(isinstance(e, DiscoveredEvent) for e in events)

    def test_event_fields(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE):
            src = TavilySource(api_key="test-key")
            events = src.search("OPEC", start=datetime(2024, 10, 1), end=datetime(2024, 10, 10))
        e = events[0]
        assert e.title == "OPEC Cuts Oil Output by 2 Million Barrels Per Day"
        assert e.url == "https://reuters.com/opec-cut-2024"
        assert e.source_name == "tavily"
        assert e.source_kind == "news_search"
        assert e.evidence_quality == "web_snippet"
        assert e.timestamp == datetime(2024, 10, 5)

    def test_description_from_content(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE):
            src = TavilySource(api_key="test-key")
            events = src.search("OPEC", start=datetime(2024, 10, 1), end=datetime(2024, 10, 10))
        assert "2 million barrels" in events[0].description

    def test_description_preserves_full_tavily_content(self):
        long_content = "Opening sentence. " + ("full Tavily content " * 80)
        raw = {
            "results": [
                {
                    "title": "Long Tavily article",
                    "url": "https://example.com/long-tavily-article",
                    "content": long_content,
                    "score": 0.91,
                    "published_at": "2016-01-04T10:17:00Z",
                }
            ]
        }
        with patch("world2ts_core.sources.tavily._curl_post", return_value=__import__("json").dumps(raw)):
            src = TavilySource(api_key="test-key")
            events = src.search("Long article", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))

        assert events[0].description == long_content
        assert len(events[0].description) > 500

    def test_retrieval_score_from_tavily_score(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE):
            src = TavilySource(api_key="test-key")
            events = src.search("OPEC", start=datetime(2024, 10, 1), end=datetime(2024, 10, 10))
        assert events[0].retrieval_score == 0.95

    def test_empty_results(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_EMPTY):
            src = TavilySource(api_key="test-key")
            events = src.search("xyz", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))
        assert events == []

    def test_search_passes_date_range_to_tavily(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_EMPTY) as post:
            src = TavilySource(api_key="test-key")
            src.search("OPEC", start=datetime(2024, 10, 1), end=datetime(2024, 10, 10))

        body = post.call_args.args[0]
        assert body["start_date"] == "2024-10-01"
        assert body["end_date"] == "2024-10-11"
        assert body["include_raw_content"] is True

    def test_event_date_can_be_extracted_from_url(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITH_URL_DATE):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing traffic", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))

        assert len(events) == 1
        assert events[0].timestamp == datetime(2016, 1, 4)

    def test_url_metadata_publication_time_upgrades_date_only_timestamp(self):
        html = """
        <html><head>
          <meta property="article:published_time" content="2016-01-04T10:17:00Z" />
        </head><body>City authorities are working on traffic.</body></html>
        """
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITH_URL_DATE), \
             patch("world2ts_core.sources.tavily._curl_get", return_value=html, create=True):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing traffic", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))

        assert len(events) == 1
        assert events[0].timestamp == datetime(2016, 1, 4, 10, 17)
        assert events[0].timestamp_precision == "exact"
        assert events[0].structured_payload["date_source"] == "url_metadata"
        assert events[0].structured_payload["timestamp_kind"] == "publication_time"

    def test_url_metadata_fetch_uses_short_timeout(self):
        with patch("world2ts_core.sources.tavily._curl_get_base", return_value="<html></html>") as get:
            from world2ts_core.sources.tavily import _curl_get

            _curl_get("https://example.com/article")

        get.assert_called_once_with(
            "https://example.com/article",
            timeout=5,
            retries=1,
            headers=["User-Agent: Mozilla/5.0"],
        )

    def test_exact_tavily_timestamp_field_keeps_time_component(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITH_EXACT_PUBLISHED_AT):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing traffic", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))

        assert len(events) == 1
        assert events[0].timestamp == datetime(2016, 1, 4, 10, 17)
        assert events[0].timestamp_precision == "exact"

    def test_exact_tavily_timestamp_wins_over_date_only_field(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITH_DATE_AND_EXACT_PUBLISHED_AT), \
             patch("world2ts_core.sources.tavily._curl_get") as get:
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing traffic", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))

        assert len(events) == 1
        assert events[0].timestamp == datetime(2016, 1, 4, 10, 17)
        assert events[0].timestamp_precision == "exact"
        assert events[0].structured_payload["date_source"] == "published_at"
        get.assert_not_called()

    def test_events_outside_requested_window_are_skipped(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITH_URL_DATE):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing traffic", start=datetime(2016, 1, 5), end=datetime(2016, 1, 10))

        assert events == []

    def test_results_without_event_date_are_skipped(self):
        # No date in the result, and the URL-metadata fallback also yields nothing,
        # so the candidate is skipped (stays offline via the mocked fetch).
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITHOUT_DATE), \
             patch("world2ts_core.sources.tavily.fetch_url_publication_timestamp", return_value=(None, "")):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing subway", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))

        assert events == []

    def test_undated_result_recovered_via_url_metadata(self):
        # No date in the result, but the article's HTML publication metadata yields
        # an in-window date -> the candidate is recovered rather than dropped.
        recovered = datetime(2016, 1, 3)
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITHOUT_DATE), \
             patch("world2ts_core.sources.tavily.fetch_url_publication_timestamp",
                   return_value=(recovered, "2016-01-03T08:00:00")):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing subway", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))
        assert len(events) == 1
        assert events[0].timestamp == recovered
        assert events[0].timestamp_precision == "exact"
        assert events[0].structured_payload.get("date_source") == "url_metadata"

    def test_undated_result_recovered_but_out_of_window_is_dropped(self):
        # URL metadata resolves to a date OUTSIDE the window (e.g. a retrospective) -> dropped.
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITHOUT_DATE), \
             patch("world2ts_core.sources.tavily.fetch_url_publication_timestamp",
                   return_value=(datetime(2024, 8, 1), "2024-08-01T00:00:00")):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing subway", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))
        assert events == []

    def test_unlabeled_text_dates_are_not_used_as_event_dates(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITH_UNLABELED_TEXT_DATE):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing subway", start=datetime(2016, 4, 1), end=datetime(2016, 4, 1))

        assert events == []

    def test_labeled_raw_content_date_can_be_used(self):
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITH_LABELED_RAW_DATE):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing traffic", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))

        assert len(events) == 1
        assert events[0].timestamp == datetime(2016, 1, 4)
        assert events[0].structured_payload["date_source"] == "text"

    def test_naked_datetime_in_published_at_falls_back_to_day_precision(self):
        """A naked datetime (no tz) must not be treated as UTC. Drop to day precision."""
        raw = """{
            "results": [
                {
                    "title": "WBUR sports recap",
                    "url": "https://www.wbur.org/news/2018/12/10/patriots-loss",
                    "content": "WBUR Boston public radio recap.",
                    "score": 0.9,
                    "published_at": "2018-12-10 08:16:38"
                }
            ]
        }"""
        with patch("world2ts_core.sources.tavily._curl_post", return_value=raw):
            src = TavilySource(api_key="test-key")
            events = src.search("Patriots", start=datetime(2018, 12, 1), end=datetime(2018, 12, 15))

        assert len(events) == 1
        assert events[0].timestamp_precision == "day"
        assert events[0].timestamp.date() == events[0].timestamp.date()
        assert events[0].timestamp.hour == 0
        assert events[0].timestamp.minute == 0

    def test_tz_aware_datetime_is_converted_to_utc_and_kept_exact(self):
        """Z-suffixed and offset-bearing timestamps stay exact, normalized to UTC."""
        raw = """{
            "results": [
                {
                    "title": "Boston Globe article",
                    "url": "https://www.bostonglobe.com/news/2018/12/02/test",
                    "content": "BG story.",
                    "score": 0.9,
                    "published_at": "2018-12-02T18:13:50-05:00"
                }
            ]
        }"""
        with patch("world2ts_core.sources.tavily._curl_post", return_value=raw):
            src = TavilySource(api_key="test-key")
            events = src.search("BG", start=datetime(2018, 12, 1), end=datetime(2018, 12, 15))

        assert len(events) == 1
        assert events[0].timestamp_precision == "exact"
        # -05:00 → UTC: 18:13:50 + 5h = 23:13:50
        assert events[0].timestamp == datetime(2018, 12, 2, 23, 13, 50)

    def test_naked_datetime_in_html_meta_falls_back_to_day_precision(self):
        """Inline URL meta upgrade with naked datetime must drop to day precision."""
        html_naked = """
        <html><head>
          <meta property="article:published_time" content="2016-01-04 10:17:00" />
        </head><body>City authorities working on traffic.</body></html>
        """
        with patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_RESPONSE_WITH_URL_DATE), \
             patch("world2ts_core.sources.tavily._curl_get", return_value=html_naked, create=True):
            src = TavilySource(api_key="test-key")
            events = src.search("Beijing traffic", start=datetime(2016, 1, 1), end=datetime(2016, 1, 5))

        assert len(events) == 1
        # URL meta fetch returns None for naked → falls through to URL-path day precision
        assert events[0].timestamp_precision == "day"
        assert events[0].timestamp == datetime(2016, 1, 4)
        # Should NOT be marked as url_metadata since naked was rejected
        assert events[0].structured_payload["date_source"] != "url_metadata"

    def test_missing_api_key_raises(self):
        with patch("world2ts_core.sources.tavily.config_get", return_value=""), \
             patch("world2ts_core.sources.tavily.os.environ.get", return_value=""), \
             patch("world2ts_core.sources.tavily.load_tavily_file_config", return_value={}):
            with pytest.raises(ValueError, match="TAVILY_API_KEY"):
                TavilySource(api_key="")

    def test_quota_aware_pool_chooses_key_with_most_remaining_credits(self):
        usages = {
            "low-key": {"key": {"usage": 95, "limit": 100}},
            "high-key": {"key": {"usage": 10, "limit": 100}},
        }

        with patch("world2ts_core.sources.tavily._curl_usage", side_effect=lambda key: json.dumps(usages[key]), create=True), \
             patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_EMPTY) as post:
            src = TavilySource(api_keys=["low-key", "high-key"])
            src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

        assert post.call_args.kwargs["api_key"] == "high-key"

    def test_quota_aware_pool_uses_account_plan_limit_when_key_limit_is_null(self):
        usages = {
            "low-key": {
                "key": {"usage": 10, "limit": None},
                "account": {"plan_usage": 950, "plan_limit": 1000},
            },
            "high-key": {
                "key": {"usage": 900, "limit": None},
                "account": {"plan_usage": 100, "plan_limit": 1000},
            },
        }

        with patch("world2ts_core.sources.tavily._curl_usage", side_effect=lambda key: json.dumps(usages[key]), create=True), \
             patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_EMPTY) as post:
            src = TavilySource(api_keys=["low-key", "high-key"])
            src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

        assert post.call_args.kwargs["api_key"] == "high-key"

    def test_key_pool_reads_keys_from_configured_environment_names(self):
        env = {
            "WORLD2TS_TAVILY_ONE": "low-key",
            "WORLD2TS_TAVILY_TWO": "high-key",
        }

        def fake_config_get(key, default=""):
            if key == "tavily_api_key_envs":
                return ["WORLD2TS_TAVILY_ONE", "WORLD2TS_TAVILY_TWO"]
            return default

        def fake_env_get(key, default=""):
            return env.get(key, default)

        usages = {
            "low-key": {"key": {"usage": 80, "limit": 100}},
            "high-key": {"key": {"usage": 5, "limit": 100}},
        }

        with patch("world2ts_core.sources.tavily.config_get", side_effect=fake_config_get), \
             patch("world2ts_core.sources.tavily.os.environ.get", side_effect=fake_env_get), \
             patch("world2ts_core.sources.tavily._curl_usage", side_effect=lambda key: json.dumps(usages[key]), create=True), \
             patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_EMPTY) as post:
            src = TavilySource()
            src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

        assert post.call_args.kwargs["api_key"] == "high-key"

    def test_key_pool_reads_root_tavily_toml(self, tmp_path, monkeypatch):
        tavily_config = tmp_path / "tavily.toml"
        tavily_config.write_text(
            "\n".join(
                [
                    'api_keys = ["low-key", "high-key"]',
                    "usage_cache_seconds = 60",
                    "min_remaining_credits = 10",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        usages = {
            "low-key": {"key": {"usage": 95, "limit": 100}},
            "high-key": {"key": {"usage": 90, "limit": 100}},
        }

        monkeypatch.chdir(tmp_path)
        with patch("world2ts_core.sources.tavily.config_get", side_effect=lambda key, default="": default), \
             patch("world2ts_core.sources.tavily.os.environ.get", return_value=""), \
             patch("world2ts_core.sources.tavily._curl_usage", side_effect=lambda key: json.dumps(usages[key]), create=True), \
             patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_EMPTY) as post:
            src = TavilySource()
            src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

        assert post.call_args.kwargs["api_key"] == "high-key"

    def test_root_tavily_toml_path_can_be_overridden(self, tmp_path):
        tavily_config = tmp_path / "custom-tavily.toml"
        tavily_config.write_text('api_keys = ["file-key"]\n', encoding="utf-8")

        with patch("world2ts_core.sources.tavily.config_get", side_effect=lambda key, default="": str(tavily_config) if key == "tavily_config_file" else default), \
             patch("world2ts_core.sources.tavily.os.environ.get", return_value=""), \
             patch("world2ts_core.sources.tavily._curl_post", return_value=_MOCK_EMPTY) as post:
            src = TavilySource()
            src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

        assert post.call_args.kwargs["api_key"] == "file-key"

    def test_key_pool_retries_next_key_when_selected_key_fails(self):
        usages = {
            "bad-key": {"key": {"usage": 0, "limit": 100}},
            "good-key": {"key": {"usage": 50, "limit": 100}},
        }
        calls = []

        def fake_post(body, api_key=""):
            calls.append(api_key)
            if api_key == "bad-key":
                raise SourceError("429 rate limit")
            return _MOCK_EMPTY

        with patch("world2ts_core.sources.tavily._curl_usage", side_effect=lambda key: json.dumps(usages[key]), create=True), \
             patch("world2ts_core.sources.tavily._curl_post", side_effect=fake_post):
            src = TavilySource(api_keys=["bad-key", "good-key"])
            events = src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

        assert events == []
        assert calls == ["bad-key", "good-key"]

    def test_curl_failure_raises(self):
        # Total search failure (all keys errored) must fail loudly, not be
        # mistaken for a zero-result window.
        with patch("world2ts_core.sources.tavily._curl_post", side_effect=SourceError("timeout")):
            src = TavilySource(api_key="test-key")
            with pytest.raises(SourceError):
                src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

    def test_quota_error_envelope_raises(self):
        # HTTP-200-style error envelope (no 'results' key) — e.g. Tavily quota
        # exhaustion — must raise, not be parsed as an empty result set.
        envelope = '{"detail":{"error":"This request exceeds your plan\'s set usage limit"}}'
        with patch("world2ts_core.sources.tavily._curl_post", return_value=envelope):
            src = TavilySource(api_key="test-key")
            with pytest.raises(SourceError, match="results"):
                src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

    def test_successful_zero_result_returns_empty(self):
        # A genuine zero-result response (valid 'results': []) stays [].
        with patch("world2ts_core.sources.tavily._curl_post", return_value='{"results": []}'):
            src = TavilySource(api_key="test-key")
            events = src.search("query", start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))
        assert events == []

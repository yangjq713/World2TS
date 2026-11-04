import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, get_ident
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch
import yaml
from torch import nn

from utils.llm_prompts import build_direct_forecast_prompt


DEFAULTS = {
    "openai": {
        "model": "gpt-5.6-sol",
        "url": "https://api.openai.com/v1/responses",
    },
    "deepseek": {
        "model": "deepseek-v4-flash",
        "url": "https://api.deepseek.com/chat/completions",
    },
}

CACHE_VERSION = 1


class LLMTransportError(RuntimeError):
    """Raised after a request could not be delivered despite transport retries."""


class LLMIncompleteError(RuntimeError):
    """Raised when a streamed response ends with an incomplete terminal status."""


def _iter_sse_events(response):
    event_name = None
    data_lines = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event_name, "\n".join(data_lines)


def _read_responses_stream(response):
    deltas = {}
    completed_text = {}
    terminal_response = None
    stream_error = None

    for event_name, data in _iter_sse_events(response):
        if data == "[DONE]":
            continue
        event = json.loads(data)
        event_type = event.get("type") or event_name
        if event_type == "response.output_text.delta":
            key = (event.get("output_index", 0), event.get("content_index", 0))
            deltas.setdefault(key, []).append(event.get("delta", ""))
        elif event_type == "response.output_text.done":
            key = (event.get("output_index", 0), event.get("content_index", 0))
            completed_text[key] = event.get("text", "")
        elif event_type in {
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            terminal_response = event.get("response")
        elif event_type == "error":
            stream_error = event.get("error") or event

    if stream_error is not None:
        raise RuntimeError(f"Responses stream returned an error event: {stream_error}")
    if not isinstance(terminal_response, dict):
        raise RuntimeError("Responses stream ended without a terminal response event")

    keys = sorted(set(deltas) | set(completed_text))
    output_text = "".join(
        completed_text.get(key, "".join(deltas.get(key, []))) for key in keys
    )
    terminal_response["output_text"] = output_text
    return terminal_response


def _batch_text(values, index):
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return values[index]


def _forecast_schema(pred_len):
    return {
        "type": "object",
        "properties": {
            "forecast": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": pred_len,
                "maxItems": pred_len,
            }
        },
        "required": ["forecast"],
        "additionalProperties": False,
    }


def _load_json_output(text):
    if not isinstance(text, str):
        raise TypeError(f"LLM output must be text, got {type(text).__name__}")
    boundary_format_characters = "\ufeff\u200b\u200c\u200d\u2060"
    normalized = text.strip().strip(boundary_format_characters).strip()
    if not normalized:
        raise ValueError("LLM returned empty output text")
    return json.loads(normalized)


def _validated_forecast(values, pred_len):
    if not isinstance(values, list) or len(values) != pred_len:
        length = len(values) if isinstance(values, list) else type(values).__name__
        raise ValueError(f"LLM returned {length} values, expected {pred_len}")
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("LLM returned a non-finite forecast value")
    return values


class Model(nn.Module):
    """Direct API-based LLM forecaster for the two evaluation conditions."""

    requires_training = False
    progress_name = "DirectLLM"
    cache_directory_name = "directllm_cache"

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.provider = getattr(configs, "llm_provider", "openai").lower()
        if self.provider not in DEFAULTS:
            raise ValueError("llm_provider must be 'openai' or 'deepseek'")
        defaults = DEFAULTS[self.provider]
        self.model_name = getattr(configs, "llm_model", None) or defaults["model"]
        keys_path = getattr(configs, "llm_keys_path", None)
        keys_path = keys_path or Path(__file__).resolve().parents[1] / "llm_keys.yaml"
        local_config = yaml.safe_load(Path(keys_path).read_text(encoding="utf-8"))
        configured_urls = local_config.get("api_urls") or {}
        self.api_url = (
            getattr(configs, "llm_api_url", None)
            or configured_urls.get(self.provider)
            or defaults["url"]
        )
        keys = local_config["api_keys"]
        self.api_key = keys[self.provider]
        if not self.api_key:
            raise ValueError(
                f"No API key configured for {self.provider} in {keys_path}"
            )
        self.reasoning_effort = getattr(configs, "llm_reasoning_effort", "high")
        self.max_output_tokens = getattr(configs, "llm_max_output_tokens", 4096)
        self.retry_max_output_tokens = getattr(
            configs,
            "llm_retry_max_output_tokens",
            98304,
        )
        self.timeout = getattr(configs, "llm_timeout", 360)
        self.concurrency = getattr(configs, "llm_concurrency", 16)
        self.transport_retries = getattr(configs, "llm_transport_retries", 2)
        self.retry_backoff_seconds = getattr(
            configs,
            "llm_retry_backoff_seconds",
            5.0,
        )
        self.progress_interval = getattr(configs, "llm_progress_interval", 8)
        self.cache_enabled = bool(getattr(configs, "llm_cache", False))
        cache_namespace = getattr(configs, "llm_cache_namespace", None)
        cache_root = getattr(configs, "llm_cache_dir", None)
        if self.cache_enabled:
            if not cache_namespace:
                raise ValueError(
                    "llm_cache_namespace is required when caching is enabled"
                )
            if cache_root:
                self.cache_dir = Path(cache_root) / cache_namespace
            else:
                self.cache_dir = (
                    Path(getattr(configs, "results", "./results"))
                    / cache_namespace
                    / self.cache_directory_name
                )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        if self.max_output_tokens <= 0:
            raise ValueError("llm_max_output_tokens must be positive")
        if self.retry_max_output_tokens < self.max_output_tokens:
            raise ValueError(
                "llm_retry_max_output_tokens must be at least llm_max_output_tokens"
            )
        if self.concurrency <= 0:
            raise ValueError("llm_concurrency must be positive")
        if self.transport_retries < 0:
            raise ValueError("llm_transport_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("llm_retry_backoff_seconds must be non-negative")
        if self.progress_interval <= 0:
            raise ValueError("llm_progress_interval must be positive")
        self._progress_lock = Lock()
        self._progress_total = None
        self._progress_completed = 0
        self._progress_started = None
        self._usage_lock = Lock()
        self._api_usage = {
            "requests": 0,
            "responses": 0,
            "responses_with_usage": 0,
            "failed_requests": 0,
            "retries": 0,
            "transport_retries": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "cache_miss_prompt_tokens": 0,
            "forecast_cache_hits": 0,
            "forecast_cache_misses": 0,
            "forecast_cache_writes": 0,
        }

    def _cache_key(self, system_prompt, user_prompt):
        identity = {
            "version": CACHE_VERSION,
            "provider": self.provider,
            "model": self.model_name,
            "api_url": self.api_url,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "retry_max_output_tokens": self.retry_max_output_tokens,
            "pred_len": self.pred_len,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read_cached_forecast(self, cache_key):
        if self.cache_dir is None:
            return None
        cache_path = self.cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            self._increment_api_usage("forecast_cache_misses")
            return None
        try:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            if record.get("version") != CACHE_VERSION:
                raise ValueError("cache version mismatch")
            if record.get("cache_key") != cache_key:
                raise ValueError("cache key mismatch")
            values = _validated_forecast(record.get("forecast"), self.pred_len)
        except Exception as error:
            raise RuntimeError(
                f"Invalid DirectLLM forecast cache entry {cache_path}: {error}"
            ) from error
        self._increment_api_usage("forecast_cache_hits")
        return values

    def _write_cached_forecast(self, cache_key, values):
        if self.cache_dir is None:
            return
        cache_path = self.cache_dir / f"{cache_key}.json"
        temporary_path = cache_path.with_name(
            f".{cache_path.stem[:16]}.{get_ident()}.tmp"
        )
        record = {
            "version": CACHE_VERSION,
            "cache_key": cache_key,
            "forecast": values,
        }
        try:
            temporary_path.write_text(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._increment_api_usage("forecast_cache_writes")

    def start_progress(self, total):
        with self._progress_lock:
            self._progress_total = total
            self._progress_completed = 0
            self._progress_started = time.monotonic()
        print(f"{self.progress_name} progress: 0/{total}", flush=True)

    def _record_progress(self):
        with self._progress_lock:
            self._progress_completed += 1
            completed = self._progress_completed
            total = self._progress_total
            started = self._progress_started
            should_report = (
                completed == 1
                or completed % self.progress_interval == 0
                or completed == total
            )
        if should_report:
            elapsed = time.monotonic() - started if started is not None else 0.0
            print(
                f"{self.progress_name} progress: {completed}/{total} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    def _increment_api_usage(self, field):
        with self._usage_lock:
            self._api_usage[field] += 1

    def _record_api_response(self, response):
        usage = response.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        completion_tokens = (
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        prompt_details = (
            usage.get("prompt_tokens_details")
            or usage.get("input_tokens_details")
            or {}
        )
        completion_details = (
            usage.get("completion_tokens_details")
            or usage.get("output_tokens_details")
            or {}
        )
        cached_prompt_tokens = usage.get("prompt_cache_hit_tokens")
        if cached_prompt_tokens is None:
            cached_prompt_tokens = prompt_details.get("cached_tokens", 0)
        cache_miss_prompt_tokens = usage.get("prompt_cache_miss_tokens")
        if cache_miss_prompt_tokens is None:
            cache_miss_prompt_tokens = max(
                0,
                prompt_tokens - (cached_prompt_tokens or 0),
            )
        reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0

        with self._usage_lock:
            self._api_usage["responses"] += 1
            if usage:
                self._api_usage["responses_with_usage"] += 1
            self._api_usage["prompt_tokens"] += int(prompt_tokens)
            self._api_usage["completion_tokens"] += int(completion_tokens)
            self._api_usage["reasoning_tokens"] += int(reasoning_tokens)
            self._api_usage["total_tokens"] += int(total_tokens)
            self._api_usage["cached_prompt_tokens"] += int(cached_prompt_tokens or 0)
            self._api_usage["cache_miss_prompt_tokens"] += int(
                cache_miss_prompt_tokens or 0
            )

    def get_api_usage_metrics(self):
        with self._usage_lock:
            usage = dict(self._api_usage)
        usage["non_reasoning_completion_tokens"] = max(
            0,
            usage["completion_tokens"] - usage["reasoning_tokens"],
        )
        return usage

    def _post(self, payload):
        request_body = json.dumps(payload).encode("utf-8")
        transient_statuses = {408, 409, 429, 500, 502, 503, 504}
        attempts = self.transport_retries + 1
        for attempt in range(1, attempts + 1):
            self._increment_api_usage("requests")
            request = Request(
                self.api_url,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": (
                        "text/event-stream"
                        if payload.get("stream")
                        else "application/json"
                    ),
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if payload.get("stream") and "text/event-stream" in content_type:
                        parsed_response = _read_responses_stream(response)
                    else:
                        parsed_response = json.loads(response.read().decode("utf-8"))
                    self._record_api_response(parsed_response)
                    if payload.get("stream"):
                        status = parsed_response.get("status")
                        if status == "incomplete":
                            details = parsed_response.get("incomplete_details")
                            raise LLMIncompleteError(
                                f"Responses stream was incomplete: {details}"
                            )
                        if status != "completed":
                            error = parsed_response.get("error")
                            raise RuntimeError(
                                f"Responses stream ended with status={status!r}: {error}"
                            )
                    return parsed_response
            except LLMIncompleteError:
                self._increment_api_usage("failed_requests")
                raise
            except HTTPError as error:
                self._increment_api_usage("failed_requests")
                detail = error.read().decode("utf-8", errors="replace")
                message = f"LLM API returned HTTP {error.code}: {detail}"
                can_retry = error.code in transient_statuses and attempt < attempts
                cause = error
            except Exception as error:
                self._increment_api_usage("failed_requests")
                message = f"LLM API transport failed: {type(error).__name__}: {error}"
                can_retry = attempt < attempts
                cause = error

            if not can_retry:
                raise LLMTransportError(message) from cause

            delay = min(self.retry_backoff_seconds * (2 ** (attempt - 1)), 60.0)
            self._increment_api_usage("transport_retries")
            print(
                f"{message}; transport retry {attempt}/{self.transport_retries} "
                f"in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

        raise AssertionError("unreachable")

    def _openai(self, system_prompt, user_prompt, max_output_tokens):
        response = self._post(
            {
                "model": self.model_name,
                "instructions": system_prompt,
                "input": user_prompt,
                "reasoning": {"effort": self.reasoning_effort},
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "time_series_forecast",
                        "strict": True,
                        "schema": _forecast_schema(self.pred_len),
                    }
                },
                "max_output_tokens": max_output_tokens,
                "store": False,
                "stream": True,
            }
        )
        if response.get("output_text"):
            return response["output_text"]
        return next(
            content["text"]
            for item in response["output"]
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )

    def _deepseek(self, system_prompt, user_prompt, max_output_tokens):
        reasoning_effort = (
            "max" if self.reasoning_effort.casefold() in {"xhigh", "max"} else "high"
        )
        response = self._post(
            {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "thinking": {"type": "enabled"},
                "reasoning_effort": reasoning_effort,
                "max_tokens": max_output_tokens,
                "stream": False,
            }
        )
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("DeepSeek returned no choices")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise ValueError(
                f"DeepSeek response reached max_tokens={max_output_tokens}"
            )
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"DeepSeek returned empty content (finish_reason={finish_reason!r})"
            )
        return content

    def _forecast(
        self,
        history,
        metadata,
        events,
        history_time_range,
        forecast_time_range,
    ):
        system_prompt, user_prompt, _ = build_direct_forecast_prompt(
            history,
            self.pred_len,
            metadata=metadata,
            events=events,
            history_time_range=history_time_range,
            forecast_time_range=forecast_time_range,
        )
        cache_key = self._cache_key(system_prompt, user_prompt)
        cached_forecast = self._read_cached_forecast(cache_key)
        if cached_forecast is not None:
            return cached_forecast

        request = self._openai if self.provider == "openai" else self._deepseek
        limits = [self.max_output_tokens]
        if self.retry_max_output_tokens > self.max_output_tokens:
            limits.append(self.retry_max_output_tokens)
        for attempt, max_output_tokens in enumerate(limits, start=1):
            try:
                payload = _load_json_output(
                    request(system_prompt, user_prompt, max_output_tokens)
                )
                values = _validated_forecast(payload["forecast"], self.pred_len)
                break
            except Exception as error:
                if isinstance(error, LLMTransportError):
                    raise RuntimeError(
                        f"DirectLLM forecast failed after transport retries: {error}"
                    ) from error
                if attempt == len(limits):
                    raise RuntimeError(
                        f"DirectLLM forecast failed after {attempt} attempts: {error}"
                    ) from error
                self._increment_api_usage("retries")
                print(
                    "DirectLLM request failed with "
                    f"max_tokens={max_output_tokens}; retrying with "
                    f"max_tokens={limits[attempt]} ({type(error).__name__}: {error})",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            raise AssertionError("unreachable")

        self._write_cached_forecast(cache_key, values)
        return values

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        metadata=None,
        events=None,
        history_time_range=None,
        forecast_time_range=None,
    ):
        if x_enc.shape[-1] != 1:
            raise ValueError("DirectLLM currently supports one variable per evaluation")
        forecasts = [None] * x_enc.shape[0]
        if self.concurrency == 1:
            for index in range(x_enc.shape[0]):
                forecasts[index] = self._forecast(
                    x_enc[index, :, 0].detach().cpu().tolist(),
                    _batch_text(metadata, index),
                    _batch_text(events, index),
                    _batch_text(history_time_range, index),
                    _batch_text(forecast_time_range, index),
                )
                self._record_progress()
            return torch.tensor(
                forecasts,
                dtype=x_enc.dtype,
                device=x_enc.device,
            ).unsqueeze(-1)

        with ThreadPoolExecutor(
            max_workers=min(self.concurrency, x_enc.shape[0])
        ) as executor:
            futures = {
                executor.submit(
                    self._forecast,
                    x_enc[index, :, 0].detach().cpu().tolist(),
                    _batch_text(metadata, index),
                    _batch_text(events, index),
                    _batch_text(history_time_range, index),
                    _batch_text(forecast_time_range, index),
                ): index
                for index in range(x_enc.shape[0])
            }
            for future in as_completed(futures):
                index = futures[future]
                forecasts[index] = future.result()
                self._record_progress()
        return torch.tensor(
            forecasts, dtype=x_enc.dtype, device=x_enc.device
        ).unsqueeze(-1)

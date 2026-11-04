import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from threading import Lock, get_ident
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from scipy.signal import find_peaks, peak_widths

from .DirectLLM import (
    LLMTransportError,
    Model as DirectLLMModel,
    _load_json_output,
    _validated_forecast,
)
from .PatchTST import Model as PatchTSTExpert


PROTOCOL_VERSION = 2
FEATURE_TOOLS = (
    "extract_basic_statistics",
    "extract_within_channel_dynamics",
    "extract_forecast_residuals",
    "extract_data_quality",
    "extract_event_summary",
)
PREDICTORS = ("arima", "dlinear", "patchtst", "itransformer", "chronos2")
PREDICTOR_DESCRIPTIONS = {
    "arima": "linear dynamics with stable seasonality",
    "dlinear": "decomposable trend and seasonal dynamics",
    "patchtst": "regular dependence and multi-scale seasonality",
    "itransformer": "pronounced cross-channel dependence",
    "chronos2": "irregular, nonlinear, or shifting regimes",
}

COORDINATOR_SYSTEM = """You are the CastFSR-Zero forecasting coordinator.
Follow this target-blind Fast--Slow--Reflect workflow in order:
Stage 1a calls exactly one diagnostic feature tool. Stage 1b uses that result to
route the instance to exactly one numerical forecaster. Stage 2 treats the
forecaster output as a prior and makes only localized, evidence-supported
adjustments. Stage 3 audits temporal regularity, contextual support, domain
constraints, timestamp correctness, and numerical feasibility, repairing only
unsupported or invalid segments. All event text describes pre-forecast history;
it is never a future-known covariate. Return only the JSON object requested by
the current stage."""


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("DLinear moving-average kernel must be a positive odd value")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, values):
        padding = (self.kernel_size - 1) // 2
        front = values[:, :1, :].repeat(1, padding, 1)
        end = values[:, -1:, :].repeat(1, padding, 1)
        padded = torch.cat([front, values, end], dim=1)
        return self.pool(padded.permute(0, 2, 1)).permute(0, 2, 1)


class DLinearExpert(nn.Module):
    """Univariate DLinear expert matching the released decomposition architecture."""

    def __init__(self, seq_len, pred_len, moving_average=25):
        super().__init__()
        self.pred_len = pred_len
        self.moving_average = MovingAverage(moving_average)
        self.seasonal_projection = nn.Linear(seq_len, pred_len)
        self.trend_projection = nn.Linear(seq_len, pred_len)
        initial_weight = torch.full((pred_len, seq_len), 1.0 / seq_len)
        with torch.no_grad():
            self.seasonal_projection.weight.copy_(initial_weight)
            self.trend_projection.weight.copy_(initial_weight)

    def forward(self, values):
        trend = self.moving_average(values)
        seasonal = values - trend
        seasonal = self.seasonal_projection(seasonal.permute(0, 2, 1))
        trend = self.trend_projection(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1)


def _lag_correlation(values, lag):
    if lag <= 0 or len(values) <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _finite_value(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_finite_value(item) for item in value]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _finite_metrics(metrics):
    return {key: _finite_value(value) for key, value in metrics.items()}


def _spectral_features(values):
    power = np.abs(np.fft.fft(values)[: values.size // 2]) ** 2
    frequencies = np.fft.fftfreq(values.size)[: values.size // 2]
    if power.size == 0 or float(np.sum(power)) <= 0:
        return 0.0, 0.0
    peak_frequency = abs(float(frequencies[int(np.argmax(power))]))
    normalized = power / np.sum(power)
    normalized = normalized[normalized > 0]
    entropy = -float(np.sum(normalized * np.log2(normalized)))
    return peak_frequency, entropy


def _changepoint_features(values):
    if values.size < 10:
        return 0.0, 0.0
    try:
        import ruptures as rpt

        breakpoints = rpt.Pelt(model="rbf").fit(values).predict(pen=5.0)
        count = max(0, len(breakpoints) - 1)
        segments = np.split(values, breakpoints[:-1])
        means = [float(np.mean(segment)) for segment in segments if segment.size]
        score = max(np.abs(np.diff(means))) if len(means) > 1 else 0.0
        return float(min(count, 5)), float(score)
    except Exception:
        return 0.0, 0.0


def extract_feature(tool_name, history, seasonal_period=1):
    """Execute one Fast-stage feature tool using the released CastFSR metrics."""
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("CastFSR feature tools require finite one-dimensional history")
    differences = np.diff(values)

    if tool_name == "extract_basic_statistics":
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        peak_frequency, spectral_entropy = _spectral_features(values)
        q975, q025, q75, q25 = np.percentile(values, [97.5, 2.5, 75, 25])
        interquartile_range = q75 - q25
        metrics = {
            "median": median,
            "mad": mad,
            "acf1": _lag_correlation(values, 1),
            "acf_seasonal": _lag_correlation(values, seasonal_period),
            "peak_freq": peak_frequency,
            "spec_entropy": spectral_entropy,
            "cusum_max": np.max(np.abs(np.cumsum(values - median))),
            "qkurt": (
                (q975 - q025) / interquartile_range
                if interquartile_range != 0
                else 0.0
            ),
            "mean_abs_corr": 0.0,
            "pca_var_ratio1": 1.0,
        }
    elif tool_name == "extract_within_channel_dynamics":
        changepoint_count, changepoint_score = _changepoint_features(values)
        second_differences = np.diff(values, n=2)
        signs = np.sign(differences)
        longest = current = 1 if signs.size else 0
        previous = signs[0] if signs.size else 0
        for sign in signs[1:]:
            if sign == 0 or sign == previous:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
            if sign != 0:
                previous = sign
        longest = max(longest, current)
        normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
        peaks, _ = find_peaks(normalized, prominence=1.0)
        widths = peak_widths(normalized, peaks)[0] if peaks.size else np.array([])
        spacing = np.diff(peaks)
        metrics = {
            "changepoint_count": changepoint_count,
            "changepoint_score": changepoint_score,
            "slope_max": np.max(np.abs(differences)) if differences.size else 0.0,
            "slope_second_diff_max": (
                np.max(np.abs(second_differences))
                if second_differences.size
                else 0.0
            ),
            "monotone_duration": longest / max(values.size, 1),
            "peak_count": float(peaks.size),
            "peak_max_width": np.max(widths) if widths.size else 0.0,
            "peak_spacing_cv": (
                np.std(spacing) / (np.mean(spacing) + 1e-8)
                if spacing.size
                else 0.0
            ),
        }
    elif tool_name == "extract_forecast_residuals":
        if values.size < 3 or np.std(values[:-1]) == 0:
            coefficient = 0.0
            intercept = float(np.mean(values))
            residuals = values - intercept
        else:
            design = np.column_stack([values[:-1], np.ones(values.size - 1)])
            coefficient, intercept = np.linalg.lstsq(
                design, values[1:], rcond=None
            )[0]
            residuals = values[1:] - (coefficient * values[:-1] + intercept)
        residual_scale = np.std(residuals) + 1e-8
        sorted_residuals = np.sort(np.abs(residuals))
        tail_start = int(0.9 * residuals.size)
        metrics = {
            "residual_mean": np.mean(residuals),
            "residual_max": np.max(np.abs(residuals)),
            "residual_exceed_ratio": np.mean(
                np.abs(residuals) > 2 * residual_scale
            ),
            "residual_acf1": _lag_correlation(residuals, 1),
            "residual_concentration": np.sum(sorted_residuals[tail_start:])
            / (np.sum(sorted_residuals) + 1e-8),
        }
    elif tool_name == "extract_data_quality":
        histogram, _ = np.histogram(values, bins=25)
        span = float(np.ptp(values))
        margin = max(span * 0.01, 1e-3)
        saturation = (values <= np.min(values) + margin) | (
            values >= np.max(values) - margin
        )
        metrics = {
            "quality_quantization_score": np.max(histogram) / values.size,
            "quality_saturation_ratio": np.mean(saturation),
            "quality_constant_channel_ratio": float(np.std(values) < 1e-3),
            "quality_dropout_ratio": 0.0,
        }
    elif tool_name == "extract_event_summary":
        if values.size < 4:
            segments = [values]
        else:
            threshold = np.std(differences) * 1.5 + 1e-6
            change_points = [0]
            for index in range(1, differences.size):
                if abs(differences[index] - differences[index - 1]) > threshold:
                    change_points.append(index)
            change_points.append(values.size - 1)
            segments = [
                values[start : end + 1]
                for start, end in zip(change_points[:-1], change_points[1:])
                if end >= start
            ][:8]
            if len(segments) < 3 and values.size >= 3:
                split_size = max(1, values.size // 3)
                segments = [
                    values[:split_size],
                    values[split_size : 2 * split_size],
                    values[2 * split_size :],
                ]
        slope_threshold = np.std(values) * 0.05 + 1e-6
        event_counts = Counter(rise=0, fall=0, flat=0, oscillation=0)
        for segment in segments:
            if segment.size < 2:
                continue
            slope = segment[-1] - segment[0]
            if slope > slope_threshold:
                event_counts["rise"] += 1
            elif slope < -slope_threshold:
                event_counts["fall"] += 1
            elif np.std(segment) > slope_threshold * 2:
                event_counts["oscillation"] += 1
            else:
                event_counts["flat"] += 1
        counts = [
            event_counts[name] for name in ("rise", "fall", "flat", "oscillation")
        ]
        metrics = {
            "event_segment_count": len(segments),
            "event_counts": counts,
            "event_dominant_pattern": int(np.argmax(counts)) if counts else 0,
        }
    else:
        raise ValueError(
            f"Unknown CastFSR feature tool {tool_name!r}; expected one of {FEATURE_TOOLS}"
        )
    return _finite_metrics(metrics)


def _timestamps_from_range(serialized_range, count, field_name):
    if not serialized_range:
        raise ValueError(f"CastFSR requires {field_name}")
    try:
        time_range = json.loads(serialized_range)
        start = pd.Timestamp(time_range["start"])
        end = pd.Timestamp(time_range["end"])
    except Exception as error:
        raise ValueError(f"Invalid CastFSR {field_name}: {error}") from error
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError(f"Invalid CastFSR {field_name}: start={start}, end={end}")
    return [timestamp.isoformat() for timestamp in pd.date_range(start, end, periods=count)]


def _calendar_context(timestamps):
    parsed = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    return {
        "timestamps": timestamps,
        "day_of_week": parsed.dayofweek.astype(int).tolist(),
        "day_of_month": parsed.day.astype(int).tolist(),
        "month": parsed.month.astype(int).tolist(),
        "is_weekend": (parsed.dayofweek >= 5).astype(int).tolist(),
        "time_role": "future_known_calendar_only",
    }


def _robust_scale(values, floor=1e-6):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return max(mad, float((q75 - q25) / 1.349), abs(median) * 0.01, floor)


def assess_forecast(history, forecast, seasonal_period=1):
    """Apply the released reflection families with generic adaptive thresholds.

    CastFSR's public hard bounds cover its ten benchmark datasets only. World2TS
    series therefore use scale-relative temporal checks and never invent domain
    bounds for an unseen target.
    """
    history = np.asarray(history, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    if history.ndim != 1 or history.size < 2 or not np.isfinite(history).all():
        raise ValueError("Reflection requires at least two finite history values")
    if forecast.ndim != 1 or forecast.size == 0 or not np.isfinite(forecast).all():
        raise ValueError("Reflection requires a finite one-dimensional forecast")

    history_differences = np.diff(history)
    level_scale = _robust_scale(history)
    difference_scale = _robust_scale(history_differences)
    recent_period = max(1, min(int(seasonal_period), history.size))
    recent_level = float(np.median(history[-max(8, recent_period) :]))
    history_range = max(
        float(np.quantile(history, 0.95) - np.quantile(history, 0.05)),
        level_scale,
    )
    boundary_z = abs(float(forecast[0] - history[-1])) / max(
        difference_scale, level_scale * 0.15, 1e-6
    )
    level_shift_z = abs(float(np.median(forecast) - recent_level)) / level_scale
    range_ratio = float(np.ptp(forecast)) / history_range
    ramp_z = (
        float(np.max(np.abs(np.diff(forecast)))) / difference_scale
        if forecast.size >= 2
        else 0.0
    )
    diagnostics = {
        "level_scale": level_scale,
        "difference_scale": difference_scale,
        "boundary_z": boundary_z,
        "level_shift_z": level_shift_z,
        "range_ratio": range_ratio,
        "ramp_z": ramp_z,
    }
    violations = []
    if boundary_z > 12.0:
        violations.append(
            {
                "code": "boundary_jump",
                "detail": f"first point jumps {boundary_z:.2f} adaptive scales",
            }
        )
    if level_shift_z > 8.0:
        violations.append(
            {
                "code": "extreme_level_shift",
                "detail": f"forecast median shifts {level_shift_z:.2f} robust scales",
            }
        )
    if range_ratio > 8.0:
        violations.append(
            {
                "code": "range_explosion",
                "detail": f"forecast range is {range_ratio:.2f} times history range",
            }
        )
    if ramp_z > 20.0:
        violations.append(
            {
                "code": "extreme_ramp",
                "detail": f"largest step is {ramp_z:.2f} difference scales",
            }
        )

    if (
        seasonal_period > 1
        and history.size >= 2 * seasonal_period
        and forecast.size >= seasonal_period
    ):
        previous = history[-2 * seasonal_period : -seasonal_period]
        latest = history[-seasonal_period:]
        repeatability = _lag_correlation(np.concatenate([previous, latest]), seasonal_period)
        anchor = np.resize(latest, forecast.size)
        seasonal_error_z = float(np.median(np.abs(forecast - anchor))) / level_scale
        diagnostics.update(
            {
                "seasonal_repeatability": repeatability,
                "seasonal_error_z": seasonal_error_z,
            }
        )
        if repeatability >= 0.85 and seasonal_error_z > 8.0:
            violations.append(
                {
                    "code": "seasonal_phase_break",
                    "detail": (
                        f"repeatability={repeatability:.3f} but seasonal error is "
                        f"{seasonal_error_z:.2f} robust scales"
                    ),
                }
            )

    families = {item["code"].split("_", 1)[0] for item in violations}
    return {
        "diagnostics": _finite_metrics(diagnostics),
        "violations": violations,
        "triggered": len(families) >= 2,
        "bounds_check": "not_applicable_no_dataset_specific_bounds",
    }


def _checkpoint_component(value):
    return "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in str(value)
    )


class Model(DirectLLMModel):
    """CastFSR-Zero coordinator with trainable local and remote numeric experts."""

    progress_name = "CastFSRZero"
    cache_directory_name = "castfsr_cache"

    def __init__(self, configs):
        super().__init__(configs)
        if self.provider != "deepseek":
            raise ValueError(
                "CastFSRZero currently requires --llm_provider deepseek to match "
                "the released zero-shot coordinator"
            )
        raw_predictors = getattr(configs, "castfsr_predictors", ",".join(PREDICTORS))
        if isinstance(raw_predictors, str):
            predictors = [item.strip().casefold() for item in raw_predictors.split(",")]
        else:
            predictors = [str(item).strip().casefold() for item in raw_predictors]
        self.predictors = tuple(item for item in predictors if item)
        if not self.predictors:
            raise ValueError("castfsr_predictors must contain at least one predictor")
        unsupported = sorted(set(self.predictors) - set(PREDICTORS))
        if unsupported:
            raise ValueError(
                f"Unsupported CastFSR predictors {unsupported}; expected {PREDICTORS}"
            )

        self.dataset_name = (
            getattr(configs, "castfsr_dataset_name", None) or configs.series_id
        )
        self.series_id = configs.series_id
        self.seq_len = configs.seq_len
        self.training_run = bool(getattr(configs, "is_training", 0))
        self.use_local_experts = bool(
            getattr(configs, "castfsr_use_local_experts", True)
        )
        local_supported = {"dlinear", "patchtst"}
        self.local_expert_names = tuple(
            predictor
            for predictor in self.predictors
            if self.use_local_experts and predictor in local_supported
        )
        self.remote_predictors = tuple(
            predictor
            for predictor in self.predictors
            if predictor not in self.local_expert_names
        )
        checkpoint_root = getattr(configs, "castfsr_expert_checkpoint_dir", None)
        if not checkpoint_root:
            checkpoint_root = Path(getattr(configs, "checkpoints", "./checkpoints")) / (
                "castfsr_experts"
            )
        self.expert_checkpoint_dir = (
            Path(checkpoint_root)
            / _checkpoint_component(
                getattr(configs, "castfsr_dataset_name", None) or configs.series_id
            )
            / _checkpoint_component(configs.series_id)
            / f"sl{configs.seq_len}_pl{configs.pred_len}"
        )
        self.expert_train_epochs = getattr(configs, "train_epochs", 10)
        self.expert_patience = getattr(configs, "patience", 3)
        self.expert_learning_rate = getattr(configs, "learning_rate", 1e-4)
        self.expert_lr_decay = getattr(configs, "lr_decay", 0.5)
        self.dlinear_moving_average = getattr(
            configs, "castfsr_dlinear_moving_average", 25
        )
        self.patchtst_config = {
            key: getattr(configs, key)
            for key in (
                "patch_len",
                "patch_stride",
                "d_model",
                "n_heads",
                "e_layers",
                "d_ff",
                "factor",
                "dropout",
                "activation",
            )
        }
        self.local_experts = nn.ModuleDict()
        if "dlinear" in self.local_expert_names:
            self.local_experts["dlinear"] = DLinearExpert(
                configs.seq_len,
                configs.pred_len,
                moving_average=self.dlinear_moving_average,
            )
        if "patchtst" in self.local_expert_names:
            self.local_experts["patchtst"] = PatchTSTExpert(configs)
        self.requires_training = self.training_run and bool(self.local_expert_names)
        self._local_experts_ready = not self.local_expert_names
        self._expert_inference_lock = Lock()
        if self.local_expert_names and not self.training_run:
            self._load_local_expert_checkpoints()

        self.predictor_url = getattr(
            configs, "castfsr_predictor_url", "http://127.0.0.1:8996"
        ).rstrip("/")
        self.predictor_timeout = getattr(configs, "castfsr_predictor_timeout", 120)
        self.predictor_preflight = getattr(
            configs, "castfsr_predictor_preflight", True
        )
        self.seasonal_period = getattr(configs, "seasonal_period", 1)
        # Keep CastFSR's DeepSeek completion budgets aligned with the native
        # DirectLLM path.  The CastFSR-specific names remain supported as
        # explicit overrides for experiments that need a smaller budget.
        native_max_output_tokens = getattr(configs, "llm_max_output_tokens", 65536)
        native_retry_max_output_tokens = getattr(
            configs, "llm_retry_max_output_tokens", 98304
        )
        configured_stage_tokens = getattr(
            configs, "castfsr_stage_max_output_tokens", None
        )
        configured_castfsr_tokens = getattr(
            configs, "castfsr_max_output_tokens", None
        )
        configured_castfsr_retry_tokens = getattr(
            configs, "castfsr_retry_max_output_tokens", None
        )
        self.stage_max_output_tokens = (
            native_max_output_tokens
            if configured_stage_tokens is None
            else configured_stage_tokens
        )
        self.castfsr_max_output_tokens = (
            native_max_output_tokens
            if configured_castfsr_tokens is None
            else configured_castfsr_tokens
        )
        self.castfsr_retry_max_output_tokens = (
            native_retry_max_output_tokens
            if configured_castfsr_retry_tokens is None
            else configured_castfsr_retry_tokens
        )
        self.temperature = getattr(configs, "castfsr_temperature", 0.6)
        self.top_p = getattr(configs, "castfsr_top_p", 0.7)
        self.reflection_max_rounds = getattr(
            configs, "castfsr_reflection_max_rounds", 3
        )
        if self.predictor_timeout <= 0:
            raise ValueError("castfsr_predictor_timeout must be positive")
        if (
            self.stage_max_output_tokens <= 0
            or self.castfsr_max_output_tokens <= 0
            or self.castfsr_retry_max_output_tokens <= 0
        ):
            raise ValueError("CastFSR output token limits must be positive")
        if self.castfsr_retry_max_output_tokens < max(
            self.stage_max_output_tokens, self.castfsr_max_output_tokens
        ):
            raise ValueError(
                "castfsr_retry_max_output_tokens must be at least both CastFSR "
                "request budgets"
            )
        if not 0 <= self.temperature <= 2:
            raise ValueError("castfsr_temperature must be between 0 and 2")
        if not 0 < self.top_p <= 1:
            raise ValueError("castfsr_top_p must be in (0, 1]")
        if not 1 <= self.reflection_max_rounds <= 3:
            raise ValueError("castfsr_reflection_max_rounds must be between 1 and 3")
        if self.expert_train_epochs <= 0:
            raise ValueError("train_epochs must be positive for local CastFSR experts")
        if self.expert_patience <= 0:
            raise ValueError("patience must be positive for local CastFSR experts")
        if self.expert_learning_rate <= 0:
            raise ValueError("learning_rate must be positive for local CastFSR experts")

        self._preflight_lock = Lock()
        self._preflight_result = None
        self._trace_lock = Lock()

        cache_namespace = getattr(configs, "llm_cache_namespace", None)
        self.trace_dir = None
        if cache_namespace:
            self.trace_dir = (
                Path(getattr(configs, "results", "./results"))
                / cache_namespace
                / "castfsr_traces"
            )
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def _expert_artifact_dir(self, expert_name):
        return self.expert_checkpoint_dir / expert_name

    def _expert_config(self, expert_name):
        config = {
            "format_version": 1,
            "implementation": "world2ts_eval",
            "expert": expert_name,
            "dataset_name": self.dataset_name,
            "series_id": self.series_id,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "input_scaling": "world2ts_train_split_standard_scaler",
        }
        if expert_name == "dlinear":
            config["moving_average"] = self.dlinear_moving_average
        elif expert_name == "patchtst":
            config.update(self.patchtst_config)
        return config

    def _load_local_expert_checkpoints(self):
        for expert_name, expert in self.local_experts.items():
            artifact_dir = self._expert_artifact_dir(expert_name)
            config_path = artifact_dir / "config.json"
            checkpoint_path = artifact_dir / "checkpoint.pth"
            if not config_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing local CastFSR {expert_name} artifact under {artifact_dir}; "
                    "run CastFSRZero once with --is_training 1 and include this expert"
                )
            artifact_config = json.loads(config_path.read_text(encoding="utf-8"))
            expected = self._expert_config(expert_name)
            mismatches = {
                key: (artifact_config.get(key), value)
                for key, value in expected.items()
                if artifact_config.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"Incompatible local CastFSR {expert_name} config: {mismatches}"
                )
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            expert.load_state_dict(state, strict=True)
            expert.eval()
        self._local_experts_ready = True

    @staticmethod
    def _expert_batch(batch, device, pred_len):
        history = batch[0].float().to(device)
        target = batch[1].float().to(device)[:, -pred_len:, :]
        return history, target

    @staticmethod
    def _run_local_expert(expert_name, expert, history):
        if expert_name == "patchtst":
            return expert.forecast(history)[:, -expert.pred_len :, :]
        return expert(history)

    def _validation_loss_for_expert(
        self, expert_name, expert, validation_loader, criterion
    ):
        expert.eval()
        device = next(expert.parameters()).device
        losses = []
        with torch.no_grad():
            for batch in validation_loader:
                history, target = self._expert_batch(batch, device, self.pred_len)
                prediction = self._run_local_expert(expert_name, expert, history)
                losses.append(criterion(prediction, target).item())
        if not losses:
            raise ValueError("CastFSR local expert validation loader is empty")
        return float(np.mean(losses))

    def fit_numeric_experts(self, train_loader, validation_loader):
        """Train the requested DLinear/PatchTST experts without invoking the LLM."""
        if not self.local_expert_names:
            raise ValueError(
                "CastFSRZero training requires dlinear and/or patchtst in "
                "--castfsr_predictors"
            )
        criterion = nn.MSELoss()
        training_summary = {}
        for expert_name, expert in self.local_experts.items():
            device = next(expert.parameters()).device
            optimizer = optim.Adam(
                expert.parameters(), lr=self.expert_learning_rate
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.expert_lr_decay, patience=1
            )
            artifact_dir = self._expert_artifact_dir(expert_name)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = artifact_dir / "checkpoint.pth"
            best_loss = None
            stale_epochs = 0
            epochs_completed = 0
            for epoch in range(self.expert_train_epochs):
                started = time.time()
                expert.train()
                train_losses = []
                for batch in train_loader:
                    history, target = self._expert_batch(
                        batch, device, self.pred_len
                    )
                    optimizer.zero_grad()
                    prediction = self._run_local_expert(
                        expert_name, expert, history
                    )
                    loss = criterion(prediction, target)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                if not train_losses:
                    raise ValueError("CastFSR local expert training loader is empty")
                train_loss = float(np.mean(train_losses))
                validation_loss = self._validation_loss_for_expert(
                    expert_name, expert, validation_loader, criterion
                )
                scheduler.step(validation_loss)
                epochs_completed = epoch + 1
                print(
                    f"CastFSR {expert_name} epoch {epoch + 1}/"
                    f"{self.expert_train_epochs} train={train_loss:.6f} "
                    f"val={validation_loss:.6f} "
                    f"time={time.time() - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                if best_loss is None or validation_loss < best_loss:
                    best_loss = validation_loss
                    stale_epochs = 0
                    torch.save(expert.state_dict(), checkpoint_path)
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.expert_patience:
                        break
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            expert.load_state_dict(state, strict=True)
            expert.eval()
            (artifact_dir / "config.json").write_text(
                json.dumps(
                    self._expert_config(expert_name),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            training_summary[expert_name] = {
                "best_validation_loss": best_loss,
                "epochs_completed": epochs_completed,
                "artifact_dir": str(artifact_dir.resolve()),
            }
        self._local_experts_ready = True
        self._preflight_result = None
        return training_summary

    def _local_expert_identity(self):
        identity = {}
        for expert_name in self.local_expert_names:
            checkpoint = self._expert_artifact_dir(expert_name) / "checkpoint.pth"
            if checkpoint.is_file():
                stat = checkpoint.stat()
                identity[expert_name] = {
                    "path": str(checkpoint.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            else:
                identity[expert_name] = {"status": "not_trained"}
        return identity

    def _coordinator_json(self, stage, user_prompt, max_output_tokens):
        reasoning_effort = (
            "max" if self.reasoning_effort.casefold() in {"xhigh", "max"} else "high"
        )
        limits = [int(max_output_tokens)]
        if self.castfsr_retry_max_output_tokens > max_output_tokens:
            # Native DirectLLM has one retry.  CastFSR has several coordinator
            # stages per window, so allow a second retry for transient empty or
            # malformed provider responses without changing either token cap.
            limits.extend(
                [
                    int(self.castfsr_retry_max_output_tokens),
                    int(self.castfsr_retry_max_output_tokens),
                ]
            )
        last_error = None
        for attempt, token_limit in enumerate(limits, start=1):
            try:
                response = self._post(
                    {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": COORDINATOR_SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": reasoning_effort,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_tokens": token_limit,
                        "stream": False,
                    }
                )
            except LLMTransportError as error:
                raise RuntimeError(f"CastFSR {stage} transport failed: {error}") from error
            choices = response.get("choices") or []
            if not choices:
                last_error = ValueError(f"CastFSR {stage} returned no choices")
            else:
                choice = choices[0]
                if choice.get("finish_reason") == "length":
                    last_error = ValueError(
                        f"CastFSR {stage} reached max_tokens={token_limit}"
                    )
                else:
                    content = (choice.get("message") or {}).get("content")
                    try:
                        return _load_json_output(content)
                    except Exception as error:
                        last_error = ValueError(
                            f"Invalid CastFSR {stage} JSON: {error}"
                        )
            if attempt < len(limits):
                self._increment_api_usage("retries")
                print(
                    f"CastFSR {stage} coordinator response failed at "
                    f"max_tokens={token_limit}; retrying with "
                    f"max_tokens={limits[attempt]} ({last_error})",
                    file=sys.stderr,
                    flush=True,
                )
        raise last_error

    def _predictor_health(self):
        local_ready = {
            name: self._local_experts_ready for name in self.local_expert_names
        }
        local_details = {
            name: f"in-process checkpoint={self._expert_artifact_dir(name)}"
            for name in self.local_expert_names
        }
        if not self.predictor_preflight:
            return {
                "status": "disabled",
                "predictors": list(self.predictors),
                "ready": local_ready,
                "details": local_details,
            }
        if self._preflight_result is not None:
            return self._preflight_result
        with self._preflight_lock:
            if self._preflight_result is not None:
                return self._preflight_result
            if not self.remote_predictors:
                if not self._local_experts_ready:
                    raise RuntimeError("Local CastFSR experts have not been trained")
                self._preflight_result = {
                    "status": "healthy",
                    "predictors": list(self.predictors),
                    "ready": local_ready,
                    "details": local_details,
                    "transport": "in_process",
                }
                return self._preflight_result
            request = Request(
                f"{self.predictor_url}/health?{urlencode({'dataset_name': self.dataset_name})}",
                headers={"Accept": "application/json"},
                method="GET",
            )
            try:
                with urlopen(request, timeout=self.predictor_timeout) as response:
                    health = json.loads(response.read().decode("utf-8"))
            except Exception as error:
                raise RuntimeError(
                    f"CastFSR predictor preflight failed at {self.predictor_url}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            advertised = {
                str(item).casefold() for item in health.get("predictors", [])
            }
            missing = sorted(set(self.remote_predictors) - advertised)
            if missing:
                raise RuntimeError(
                    f"CastFSR predictor service does not advertise {missing}; "
                    f"advertised={sorted(advertised)}"
                )
            readiness = health.get("ready") or {}
            unavailable = [
                predictor
                for predictor in self.remote_predictors
                if predictor in readiness and not readiness[predictor]
            ]
            if unavailable:
                details = health.get("details") or {}
                raise RuntimeError(
                    f"CastFSR predictors are not ready: {unavailable}; details={details}"
                )
            health["predictors"] = list(self.predictors)
            health.setdefault("ready", {}).update(local_ready)
            health.setdefault("details", {}).update(local_details)
            self._preflight_result = health
            return health

    def _request_predictor(
        self,
        predictor,
        history_timestamps,
        forecast_timestamps,
        history,
    ):
        self._predictor_health()
        if predictor in self.local_experts:
            if not self._local_experts_ready:
                raise RuntimeError(
                    f"Local CastFSR expert {predictor} has not been trained"
                )
            if len(history) != self.seq_len:
                raise ValueError(
                    f"Local CastFSR expert {predictor} expects history length "
                    f"{self.seq_len}, got {len(history)}"
                )
            expert = self.local_experts[predictor]
            parameter = next(expert.parameters())
            tensor = torch.tensor(
                history,
                dtype=parameter.dtype,
                device=parameter.device,
            ).reshape(1, -1, 1)
            with self._expert_inference_lock, torch.no_grad():
                expert.eval()
                values = self._run_local_expert(predictor, expert, tensor)
            values = values[0, :, 0].detach().cpu().tolist()
            return _validated_forecast(values, self.pred_len)
        payload = {
            "timestamps": history_timestamps,
            "values": history,
            "series_id": self.series_id,
            "prediction_length": self.pred_len,
            "model_name": predictor,
            "dataset_name": self.dataset_name,
            "historical_covariate_timestamps": history_timestamps,
            "historical_covariates": {},
            "future_exogenous_timestamps": forecast_timestamps,
            "future_exogenous": {},
        }
        request = Request(
            f"{self.predictor_url}/predict",
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.predictor_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"CastFSR predictor service returned HTTP {error.code}: {detail}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"CastFSR predictor service failed at {self.predictor_url}: "
                f"{type(error).__name__}: {error}"
            ) from error
        model_used = str(result.get("model_used", "")).casefold()
        if model_used != predictor:
            raise ValueError(
                f"CastFSR predictor service used {model_used!r}, expected {predictor!r}"
            )
        returned_timestamps = result.get("timestamps")
        if not isinstance(returned_timestamps, list) or len(returned_timestamps) != self.pred_len:
            raise ValueError(
                "CastFSR predictor service must return one timestamp per forecast value"
            )
        expected = [pd.Timestamp(value) for value in forecast_timestamps]
        returned = [pd.Timestamp(value) for value in returned_timestamps]
        if returned != expected:
            raise ValueError(
                "CastFSR predictor timestamps do not match the evaluator forecast grid: "
                f"returned={returned_timestamps}, expected={forecast_timestamps}"
            )
        return _validated_forecast(result.get("values"), self.pred_len)

    def _write_trace(self, cache_key, trace):
        if self.trace_dir is None:
            return
        with self._trace_lock:
            trace_path = self.trace_dir / f"{cache_key}.json"
            payload = (
                json.dumps(trace, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
            for attempt in range(3):
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                temporary_path = trace_path.with_name(
                    f".{trace_path.stem[:16]}.{get_ident()}.tmp"
                )
                try:
                    temporary_path.write_text(payload, encoding="utf-8")
                    temporary_path.replace(trace_path)
                    return
                except FileNotFoundError:
                    if attempt == 2:
                        raise
                    time.sleep(0.01 * (attempt + 1))
                finally:
                    temporary_path.unlink(missing_ok=True)

    def _forecast(
        self,
        history,
        metadata,
        events,
        history_time_range,
        forecast_time_range,
    ):
        history_timestamps = _timestamps_from_range(
            history_time_range, len(history), "history_time_range"
        )
        forecast_timestamps = _timestamps_from_range(
            forecast_time_range, self.pred_len, "forecast_time_range"
        )
        evidence = {
            "series_id": self.series_id,
            "history_timestamps": history_timestamps,
            "forecast_timestamps": forecast_timestamps,
            "normalized_history": [round(float(value), 8) for value in history],
            "metadata": metadata or "",
            "historical_pre_forecast_events": events or "",
            "event_time_role": "historical_only_not_future_known",
            "future_known_calendar": _calendar_context(forecast_timestamps),
            "future_known_exogenous": {},
            "future_known_exogenous_status": "unavailable_do_not_invent",
        }
        request_identity = {
            "protocol_version": PROTOCOL_VERSION,
            "predictor_url": self.predictor_url,
            "predictors": self.predictors,
            "local_experts": self._local_expert_identity(),
            "dataset_name": self.dataset_name,
            "stage_max_output_tokens": self.stage_max_output_tokens,
            "castfsr_max_output_tokens": self.castfsr_max_output_tokens,
            "castfsr_retry_max_output_tokens": self.castfsr_retry_max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reflection_max_rounds": self.reflection_max_rounds,
            "evidence": evidence,
        }
        canonical_request = json.dumps(
            request_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        cache_key = self._cache_key(COORDINATOR_SYSTEM, canonical_request)
        cached_forecast = self._read_cached_forecast(cache_key)
        if cached_forecast is not None:
            return cached_forecast
        predictor_health = self._predictor_health()

        feature_call = self._coordinator_json(
            "Stage 1a Fast diagnostic profiling",
            "Stage 1a / Fast diagnostic profiling. Select exactly one feature tool "
            "that resolves the largest remaining uncertainty in this sample. Return "
            f'{{"tool":"one of {FEATURE_TOOLS}","reason":"brief reason"}}.\n'
            "Do not select a predictor and do not forecast.\n"
            f"Evidence:\n{json.dumps(evidence, ensure_ascii=False)}",
            self.stage_max_output_tokens,
        )
        tool_name = str(feature_call.get("tool", ""))
        if tool_name not in FEATURE_TOOLS:
            raise ValueError(
                f"CastFSR selected invalid feature tool {tool_name!r}; "
                f"expected one of {FEATURE_TOOLS}"
            )
        feature_result = extract_feature(
            tool_name, history, seasonal_period=self.seasonal_period
        )

        predictor_call = self._coordinator_json(
            "Stage 1b Fast predictor routing",
            "Stage 1b / Fast predictor routing. Use the diagnostic result to select "
            "exactly one available numeric predictor. The pool descriptions are "
            f"{json.dumps({name: PREDICTOR_DESCRIPTIONS[name] for name in self.predictors})}. "
            "Return "
            f'{{"predictor":"one of {self.predictors}","reason":"brief reason"}}.\n'
            f"Evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n"
            f"Selected feature tool: {tool_name}\n"
            f"Feature result: {json.dumps(feature_result)}",
            self.stage_max_output_tokens,
        )
        predictor = str(predictor_call.get("predictor", "")).casefold()
        if predictor not in self.predictors:
            raise ValueError(
                f"CastFSR selected unavailable predictor {predictor!r}; "
                f"available predictors are {self.predictors}"
            )
        numeric_prior = self._request_predictor(
            predictor,
            history_timestamps,
            forecast_timestamps,
            [float(value) for value in history],
        )

        slow_call = self._coordinator_json(
            "Stage 2 Slow deliberative reasoning",
            "Stage 2 / Slow deliberative reasoning. Produce a candidate forecast. "
            "Treat the numeric output as a prior. Assess each context item's direction, "
            "reliability, lag, and temporal scope; retain prior segments when no visible "
            "evidence supports a change. Calendar timestamps are future-known, while "
            "metadata and event text are historical context and cannot by themselves "
            "authorize a horizon-specific adjustment. "
            f"Return exactly {self.pred_len} finite numbers as "
            '{"forecast":[...],"adjustments":["localized changes or none"],'
            '"rationale":"brief evidence-grounded reason"}.\n'
            f"Evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n"
            f"Feature result ({tool_name}): {json.dumps(feature_result)}\n"
            f"Predictor: {predictor}\nNumeric prior: {json.dumps(numeric_prior)}",
            self.castfsr_max_output_tokens,
        )
        slow_forecast = _validated_forecast(slow_call.get("forecast"), self.pred_len)
        candidate = slow_forecast
        reflection_rounds = []
        for round_number in range(1, self.reflection_max_rounds + 1):
            deterministic_audit = assess_forecast(
                history, candidate, seasonal_period=self.seasonal_period
            )
            reflect_call = self._coordinator_json(
                f"Stage 3 Reflect round {round_number}",
                "Stage 3 / Reflective evaluation. Audit the candidate segment by "
                "segment against temporal regularity, contextual support, domain "
                "constraints actually present in metadata, timestamp correctness, and "
                "numerical feasibility. Keep reliable segments and repair only "
                "unsupported or invalid segments. Do not invent target bounds. "
                f"Return exactly {self.pred_len} finite numbers as "
                '{"forecast":[...],"repairs":["localized repairs or none"],'
                '"rationale":"brief audit result"}.\n'
                f"Evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n"
                f"Numeric prior ({predictor}): {json.dumps(numeric_prior)}\n"
                f"Slow/current candidate: {json.dumps(candidate)}\n"
                "Deterministic target-blind audit: "
                f"{json.dumps(deterministic_audit)}",
                self.castfsr_max_output_tokens,
            )
            candidate = _validated_forecast(
                reflect_call.get("forecast"), self.pred_len
            )
            post_audit = assess_forecast(
                history, candidate, seasonal_period=self.seasonal_period
            )
            reflection_rounds.append(
                {
                    "round": round_number,
                    "pre_audit": deterministic_audit,
                    "response": reflect_call,
                    "post_audit": post_audit,
                }
            )
            if not post_audit["triggered"]:
                break
        final_forecast = candidate
        trace = {
            "protocol_version": PROTOCOL_VERSION,
            "model": self.model_name,
            "predictor_health": predictor_health,
            "evidence": evidence,
            "feature_selection": feature_call,
            "feature_result": feature_result,
            "predictor_selection": predictor_call,
            "numeric_prior": numeric_prior,
            "slow_candidate": slow_call,
            "reflection_rounds": reflection_rounds,
            "reflection_exhausted": bool(
                reflection_rounds[-1]["post_audit"]["triggered"]
            ),
            "reflect_final": reflection_rounds[-1]["response"],
        }
        try:
            self._write_trace(cache_key, trace)
        except OSError as error:
            # Trace files are diagnostic artifacts; do not discard a valid
            # forecast because a transient filesystem race prevented one from
            # being persisted.  Keep the failure visible for later inspection.
            print(
                f"CastFSR trace write failed for {cache_key}; continuing with "
                f"forecast cache ({type(error).__name__}: {error})",
                file=sys.stderr,
                flush=True,
            )
        self._write_cached_forecast(cache_key, final_forecast)
        return final_forecast

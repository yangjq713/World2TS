#!/usr/bin/env python3
"""Local unified numeric-predictor service for CastFSR-Zero.

The HTTP contract and expert pool follow the released CastFSR service at commit
``def273e0be9df1cea3e9302189c4ed9eecee5af4``. ARIMA needs no model artifact;
DLinear, PatchTST, and iTransformer use Time-Series-Library checkpoints; and
Chronos2 uses a local pretrained model directory. Missing assets are reported as
hard errors and are never replaced by another predictor.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from threading import Lock
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PREDICTOR_NAMES = ("arima", "dlinear", "patchtst", "itransformer", "chronos2")
DEFAULT_LOOKBACK = 96
DEFAULT_HORIZON = 24
MODELS_ROOT = Path(
    os.environ.get(
        "CASTFSR_MODELS_ROOT",
        Path(__file__).resolve().parent / "castfsr_models",
    )
).expanduser().resolve()
HF_CACHE_ROOT = Path(__file__).resolve().parent / "hf"
_DEVICE = "cpu"
_CACHE: dict[tuple[str, str], torch.nn.Module] = {}
_CHRONOS_PIPELINE: Any = None
_CHRONOS_LOAD_LOCK = Lock()


class PredictRequest(BaseModel):
    timestamps: list[str]
    values: list[float]
    series_id: str = "series_0"
    prediction_length: int = DEFAULT_HORIZON
    model_name: str = "patchtst"
    dataset_name: str | None = None
    historical_covariate_timestamps: list[str] = Field(default_factory=list)
    historical_covariates: dict[str, list[float]] = Field(default_factory=dict)
    future_exogenous_timestamps: list[str] = Field(default_factory=list)
    future_exogenous: dict[str, list[float]] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    timestamps: list[str]
    values: list[float]
    model_used: str
    dataset_used: str | None = None
    status: str = "success"


class TSLForecastWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, label_len: int, pred_len: int):
        super().__init__()
        self.model = model
        self.label_len = label_len
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        x_mark = torch.zeros((batch, seq_len, 4), dtype=x.dtype, device=x.device)
        decoder_length = self.label_len + self.pred_len
        x_decoder = torch.zeros(
            (batch, decoder_length, channels), dtype=x.dtype, device=x.device
        )
        x_mark_decoder = torch.zeros(
            (batch, decoder_length, 4), dtype=x.dtype, device=x.device
        )
        output = self.model(x, x_mark, x_decoder, x_mark_decoder)
        if isinstance(output, tuple):
            output = output[0]
        return output[:, -self.pred_len :, :]


def _dataset_key(dataset_name: str | None) -> str:
    return (dataset_name or "default").strip().upper()


def _artifact_dir(model_name: str, dataset_name: str | None) -> Path:
    return MODELS_ROOT / model_name / _dataset_key(dataset_name)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _tsl_model(model_name: str, config: dict[str, Any]) -> torch.nn.Module:
    tsl_root = config.get("tsl_library_path") or os.environ.get("TSL_LIBRARY_PATH")
    if not tsl_root:
        raise RuntimeError("Set TSL_LIBRARY_PATH to a Time-Series-Library checkout")
    root = str(Path(tsl_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module_names = {
        "dlinear": "DLinear",
        "patchtst": "PatchTST",
        "itransformer": "iTransformer",
    }
    defaults: dict[str, Any] = {
        "task_name": "long_term_forecast",
        "seq_len": DEFAULT_LOOKBACK,
        "label_len": DEFAULT_LOOKBACK // 2,
        "pred_len": DEFAULT_HORIZON,
        "enc_in": 1,
        "dec_in": int(config.get("enc_in", 1)),
        "c_out": int(config.get("enc_in", 1)),
        "d_model": 128,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "factor": 3,
        "dropout": 0.1,
        "activation": "gelu",
        "embed": "timeF",
        "freq": "h",
        "output_attention": False,
        "use_norm": True,
        "patch_len": 16,
        "moving_avg": 25,
        "individual": False,
    }
    defaults.update(config)
    args = SimpleNamespace(**defaults)
    base = importlib.import_module(f"models.{module_names[model_name]}").Model(args)
    return TSLForecastWrapper(base, args.label_len, args.pred_len)


def _load_model(
    model_name: str, dataset_name: str | None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    key = (model_name, _dataset_key(dataset_name))
    artifact_dir = _artifact_dir(model_name, dataset_name)
    config = _load_json(artifact_dir / "config.json")
    if key not in _CACHE:
        checkpoint = artifact_dir / "checkpoint.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing model checkpoint: {checkpoint}")
        model = _tsl_model(model_name, config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(_DEVICE).eval()
        _CACHE[key] = model
    return _CACHE[key], config


def _history_matrix(request: PredictRequest, seq_len: int) -> np.ndarray:
    columns = [np.asarray(request.values, dtype=np.float32)]
    for name in sorted(request.historical_covariates):
        values = np.asarray(request.historical_covariates[name], dtype=np.float32)
        if len(values) != len(columns[0]):
            raise ValueError(f"Historical covariate {name!r} is not target-aligned")
        columns.append(values)
    matrix = np.column_stack(columns)
    if len(matrix) < seq_len:
        raise ValueError(f"Need at least {seq_len} observations, received {len(matrix)}")
    return matrix[-seq_len:]


def _predict_arima(request: PredictRequest) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    values = np.asarray(request.values, dtype=np.float64)
    if len(values) < 8:
        raise ValueError("ARIMA requires at least eight observations")
    order = tuple(
        int(value)
        for value in os.environ.get("CASTFSR_ARIMA_ORDER", "2,1,2").split(",")
    )
    if len(order) != 3:
        raise ValueError("CASTFSR_ARIMA_ORDER must contain p,d,q")
    result = ARIMA(values, order=order).fit()
    return np.asarray(result.forecast(request.prediction_length), dtype=np.float64)


def _chronos2_model_path() -> Path:
    configured = os.environ.get("CHRONOS2_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    repository_cache = HF_CACHE_ROOT / "models--amazon--chronos-2"
    revision_path = repository_cache / "refs/main"
    if not revision_path.is_file():
        return repository_cache / "snapshots/MISSING_MAIN_REVISION"
    revision = revision_path.read_text(encoding="utf-8").strip()
    if not revision:
        return repository_cache / "snapshots/EMPTY_MAIN_REVISION"
    return (repository_cache / "snapshots" / revision).resolve()


def _load_chronos2() -> Any:
    global _CHRONOS_PIPELINE
    if _CHRONOS_PIPELINE is not None:
        return _CHRONOS_PIPELINE
    # Uvicorn serves synchronous endpoints in a thread pool.  Serialize the
    # first load so concurrent cold requests cannot create multiple Accelerate
    # dispatches on the same CUDA device (which can leave meta tensors behind).
    with _CHRONOS_LOAD_LOCK:
        if _CHRONOS_PIPELINE is not None:
            return _CHRONOS_PIPELINE
        model_path = _chronos2_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Chronos2 weights: {model_path}")
        from chronos import BaseChronosPipeline

        _CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(
            str(model_path), device_map=_DEVICE
        )
    return _CHRONOS_PIPELINE


def _chronos_covariates(
    request: PredictRequest,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    columns = sorted(set(request.historical_covariates) & set(request.future_exogenous))
    if not columns:
        return (
            [],
            np.empty((len(request.values), 0)),
            np.empty((request.prediction_length, 0)),
        )
    historical = np.column_stack(
        [np.asarray(request.historical_covariates[name]) for name in columns]
    )
    future = np.column_stack(
        [np.asarray(request.future_exogenous[name]) for name in columns]
    )
    if historical.shape != (len(request.values), len(columns)):
        raise ValueError("Chronos2 historical covariates must align with target history")
    if future.shape != (request.prediction_length, len(columns)):
        raise ValueError("Chronos2 future covariates must cover the complete horizon")
    if len(request.future_exogenous_timestamps) != request.prediction_length:
        raise ValueError("Chronos2 future timestamps must cover the complete horizon")
    if not np.isfinite(historical).all() or not np.isfinite(future).all():
        raise ValueError("Chronos2 covariates contain non-finite values")
    return columns, historical, future


def _predict_chronos2(request: PredictRequest) -> np.ndarray:
    pipeline = _load_chronos2()
    columns, historical, future = _chronos_covariates(request)
    if columns:
        item_id = request.series_id or "series_0"
        context_data: dict[str, Any] = {
            "item_id": [item_id] * len(request.values),
            "timestamp": pd.to_datetime(request.timestamps, errors="raise"),
            "target": np.asarray(request.values, dtype=np.float64),
        }
        future_data: dict[str, Any] = {
            "item_id": [item_id] * request.prediction_length,
            "timestamp": pd.to_datetime(
                request.future_exogenous_timestamps, errors="raise"
            ),
        }
        for index, name in enumerate(columns):
            context_data[name] = historical[:, index]
            future_data[name] = future[:, index]
        frame = pipeline.predict_df(
            pd.DataFrame(context_data),
            future_df=pd.DataFrame(future_data),
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=request.prediction_length,
            quantile_levels=[0.5],
            batch_size=1,
            context_length=len(request.values),
            cross_learning=False,
            validate_inputs=True,
        )
        value_column = next(
            (
                name
                for name in ("0.5", "target_0.5", "predictions")
                if name in frame.columns
            ),
            None,
        )
        if value_column is None:
            raise ValueError(f"Chronos2 output has no median column: {list(frame.columns)}")
        return np.asarray(frame[value_column], dtype=np.float64).reshape(-1)

    _, point_forecasts = pipeline.predict_quantiles(
        inputs=[np.asarray(request.values, dtype=np.float32)],
        prediction_length=request.prediction_length,
        quantile_levels=[0.5],
        batch_size=1,
        cross_learning=False,
        limit_prediction_length=True,
    )
    values = point_forecasts[0]
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    return values.reshape(-1)


def _predict_neural(request: PredictRequest) -> np.ndarray:
    model, config = _load_model(request.model_name, request.dataset_name)
    sequence_length = int(config.get("seq_len", DEFAULT_LOOKBACK))
    matrix = _history_matrix(request, sequence_length)
    tensor = torch.from_numpy(matrix).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        output = model(tensor).detach().cpu().numpy()[0]
    if output.ndim == 2:
        output = output[:, int(config.get("target_index", 0))]
    output = np.asarray(output, dtype=np.float64).reshape(-1)
    if len(output) < request.prediction_length:
        raise ValueError(
            f"Checkpoint produced {len(output)} values, expected {request.prediction_length}"
        )
    return output[: request.prediction_length]


def _future_timestamps(request: PredictRequest) -> list[str]:
    if len(request.future_exogenous_timestamps) >= request.prediction_length:
        return request.future_exogenous_timestamps[: request.prediction_length]
    parsed = pd.to_datetime(request.timestamps, errors="raise")
    if len(parsed) < 2:
        raise ValueError("At least two timestamps are required to infer frequency")
    delta = parsed[-1] - parsed[-2]
    return [
        (parsed[-1] + delta * step).strftime("%Y-%m-%d %H:%M:%S")
        for step in range(1, request.prediction_length + 1)
    ]


def _readiness(dataset_name: str | None) -> tuple[dict[str, bool], dict[str, str]]:
    ready: dict[str, bool] = {}
    details: dict[str, str] = {}
    ready["arima"] = importlib.util.find_spec("statsmodels") is not None
    details["arima"] = "ready" if ready["arima"] else "statsmodels is not installed"
    chronos_path = _chronos2_model_path()
    ready["chronos2"] = (
        importlib.util.find_spec("chronos") is not None and chronos_path.exists()
    )
    details["chronos2"] = (
        f"weights={chronos_path.resolve()}"
        if ready["chronos2"]
        else f"missing dependency or weights={chronos_path.resolve()}"
    )
    tsl_path = os.environ.get("TSL_LIBRARY_PATH")
    for name in ("dlinear", "patchtst", "itransformer"):
        artifact = _artifact_dir(name, dataset_name)
        ready[name] = bool(
            tsl_path
            and Path(tsl_path).expanduser().is_dir()
            and (artifact / "config.json").is_file()
            and (artifact / "checkpoint.pth").is_file()
        )
        details[name] = (
            f"artifact={artifact}; TSL_LIBRARY_PATH={tsl_path or 'unset'}"
        )
    return ready, details


app = FastAPI(title="CastFSR Predictor Service", version="2.0")


@app.get("/health")
def health(dataset_name: str | None = None) -> dict[str, Any]:
    ready, details = _readiness(dataset_name)
    return {
        "status": "healthy",
        "device": _DEVICE,
        "predictors": list(PREDICTOR_NAMES),
        "ready": ready,
        "details": details,
        "models_root": str(MODELS_ROOT),
        "dataset_name": _dataset_key(dataset_name),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model_name = request.model_name.lower().strip()
    if model_name not in PREDICTOR_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown predictor: {model_name}")
    request.model_name = model_name
    try:
        if model_name == "arima":
            values = _predict_arima(request)
        elif model_name == "chronos2":
            values = _predict_chronos2(request)
        else:
            values = _predict_neural(request)
        if len(values) != request.prediction_length:
            raise ValueError(
                f"Predictor returned {len(values)} values, expected {request.prediction_length}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Predictor returned non-finite values")
        return PredictResponse(
            timestamps=_future_timestamps(request),
            values=[float(value) for value in values],
            model_used=model_name,
            dataset_used=_dataset_key(request.dataset_name),
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def main() -> None:
    global MODELS_ROOT, _DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8996)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models_root", default=None)
    parser.add_argument("--tsl_library_path", default=None)
    parser.add_argument("--chronos2_model_path", default=None)
    args = parser.parse_args()
    if args.models_root:
        MODELS_ROOT = Path(args.models_root).expanduser().resolve()
    if args.tsl_library_path:
        os.environ["TSL_LIBRARY_PATH"] = str(
            Path(args.tsl_library_path).expanduser().resolve()
        )
    if args.chronos2_model_path:
        os.environ["CHRONOS2_MODEL_PATH"] = str(
            Path(args.chronos2_model_path).expanduser().resolve()
        )
    requested_device = args.device.lower()
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        _DEVICE = requested_device if ":" in requested_device else "cuda:0"
    else:
        _DEVICE = "cpu"
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

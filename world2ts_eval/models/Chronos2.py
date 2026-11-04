import numpy as np
import torch
from torch import nn


TIME_COVARIATE_NAMES = ("month", "day", "day_of_week", "hour")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class Model(nn.Module):
    """Zero-shot Chronos-2 adapter using target and known calendar features."""

    requires_training = False

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.model_path = getattr(configs, "chronos2_model_path", "amazon/chronos-2")
        self.cache_dir = getattr(configs, "chronos2_cache_dir", None)
        self.local_files_only = getattr(configs, "chronos2_local_files_only", False)
        self.use_time_covariates = getattr(
            configs, "chronos2_use_time_covariates", True
        )
        self.inference_batch_size = getattr(configs, "chronos2_batch_size", 256)
        self.context_length = getattr(configs, "chronos2_context_length", None)
        self.torch_dtype = getattr(configs, "chronos2_torch_dtype", "auto")
        if not 0 < self.pred_len <= 1024:
            raise ValueError("Chronos2 pred_len must be between 1 and 1024")
        if self.inference_batch_size <= 0:
            raise ValueError("chronos2_batch_size must be positive")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("chronos2_context_length must be positive")
        if self.torch_dtype not in {"auto", *DTYPES}:
            raise ValueError(
                "chronos2_torch_dtype must be auto, float32, float16, or bfloat16"
            )
        self.backend = None

    def _load_backend(self, device):
        if self.backend is not None:
            return self.backend
        try:
            from chronos import Chronos2Pipeline
        except ImportError as error:
            raise RuntimeError(
                "Chronos2 requires the optional foundation-model dependencies; "
                "install world2ts_eval/requirements-foundation-models.txt"
            ) from error

        load_arguments = {
            "device_map": str(device),
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if self.torch_dtype != "auto":
            load_arguments["torch_dtype"] = DTYPES[self.torch_dtype]
        backend = Chronos2Pipeline.from_pretrained(
            self.model_path,
            **load_arguments,
        )
        backend.model.eval()
        self.backend = backend
        return backend

    def _build_inputs(self, x_enc, x_mark_enc, x_mark_dec):
        histories = x_enc.detach().cpu().numpy()[:, :, 0]
        if not np.isfinite(histories).all():
            raise ValueError("Chronos2 history contains a non-finite value")
        if not self.use_time_covariates:
            return [history for history in histories]

        expected_history_shape = (x_enc.shape[0], x_enc.shape[1], 4)
        if x_mark_enc is None or tuple(x_mark_enc.shape) != expected_history_shape:
            actual_shape = None if x_mark_enc is None else tuple(x_mark_enc.shape)
            raise ValueError(
                "Chronos2 expected history calendar features with shape "
                f"{expected_history_shape}, got {actual_shape}"
            )
        expected_future_shape = (x_enc.shape[0], self.pred_len, 4)
        if x_mark_dec is None or x_mark_dec.shape[0] != x_enc.shape[0]:
            actual_shape = None if x_mark_dec is None else tuple(x_mark_dec.shape)
            raise ValueError(
                "Chronos2 expected decoder calendar features for the input batch, "
                f"got {actual_shape}"
            )
        future_marks = x_mark_dec[:, -self.pred_len :, :]
        if tuple(future_marks.shape) != expected_future_shape:
            raise ValueError(
                "Chronos2 expected future calendar features with shape "
                f"{expected_future_shape}, got {tuple(future_marks.shape)}"
            )
        past_marks = x_mark_enc.detach().cpu().numpy()
        future_marks = future_marks.detach().cpu().numpy()
        if not np.isfinite(past_marks).all() or not np.isfinite(future_marks).all():
            raise ValueError("Chronos2 calendar features contain a non-finite value")

        return [
            {
                "target": histories[index],
                "past_covariates": {
                    name: past_marks[index, :, feature_index]
                    for feature_index, name in enumerate(TIME_COVARIATE_NAMES)
                },
                "future_covariates": {
                    name: future_marks[index, :, feature_index]
                    for feature_index, name in enumerate(TIME_COVARIATE_NAMES)
                },
            }
            for index in range(x_enc.shape[0])
        ]

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
        del x_dec, metadata, events, history_time_range, forecast_time_range
        if x_enc.ndim != 3 or x_enc.shape[-1] != 1:
            raise ValueError("Chronos2 supports one target variable per evaluation")
        inputs = self._build_inputs(x_enc, x_mark_enc, x_mark_dec)
        backend = self._load_backend(x_enc.device)
        predict_arguments = {
            "inputs": inputs,
            "prediction_length": self.pred_len,
            "quantile_levels": [0.5],
            "batch_size": self.inference_batch_size,
            "cross_learning": False,
            "limit_prediction_length": True,
        }
        if self.context_length is not None:
            predict_arguments["context_length"] = self.context_length
        _, point_forecasts = backend.predict_quantiles(**predict_arguments)
        if len(point_forecasts) != x_enc.shape[0]:
            raise RuntimeError(
                f"Chronos2 returned {len(point_forecasts)} forecasts, "
                f"expected {x_enc.shape[0]}"
            )
        forecasts = []
        for forecast in point_forecasts:
            forecast = torch.as_tensor(forecast).detach().cpu()
            if tuple(forecast.shape) != (1, self.pred_len):
                raise RuntimeError(
                    "Chronos2 returned point forecast shape "
                    f"{tuple(forecast.shape)}, expected {(1, self.pred_len)}"
                )
            forecasts.append(forecast[0])
        forecasts = torch.stack(forecasts)
        if not torch.isfinite(forecasts).all():
            raise RuntimeError("Chronos2 returned a non-finite forecast value")
        return forecasts.to(dtype=x_enc.dtype, device=x_enc.device).unsqueeze(-1)

import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.conditions import CONDITIONS, NUMERICAL_ONLY


def build_parser():
    parser = argparse.ArgumentParser(description="World2TS forecasting evaluation")
    parser.add_argument("--task_name", default="long_term_forecast")
    parser.add_argument("--is_training", type=int, choices=[0, 1], required=True)
    parser.add_argument("--model_id", default="experiment")
    parser.add_argument(
        "--model",
        choices=[
            "LastValue",
            "SeasonalNaive",
            "PatchTST",
            "DirectLLM",
            "CastFSRZero",
            "Chronos2",
        ],
        required=True,
    )

    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", default="data.arrow")
    parser.add_argument("--series_id", required=True)
    parser.add_argument("--freq", default="auto")
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        default=NUMERICAL_ONLY,
    )
    parser.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--inverse", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--top_k_events", type=int, default=10)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument(
        "--stride",
        type=int,
        default=24,
        help=(
            "validation/test rolling-window stride; training always uses stride 1; "
            "defaults to 24 for all frequencies"
        ),
    )
    parser.add_argument("--seasonal_period", type=int, default=7)

    parser.add_argument(
        "--llm_provider", choices=["openai", "deepseek"], default="openai"
    )
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--llm_api_url", default=None)
    parser.add_argument("--llm_keys_path", default=None)
    parser.add_argument("--llm_reasoning_effort", default="high")
    parser.add_argument("--llm_max_output_tokens", type=int, default=65536)
    parser.add_argument("--llm_retry_max_output_tokens", type=int, default=98304)
    parser.add_argument("--llm_concurrency", type=int, default=16)
    parser.add_argument("--llm_progress_interval", type=int, default=8)
    parser.add_argument("--llm_timeout", type=int, default=360)
    parser.add_argument("--llm_transport_retries", type=int, default=2)
    parser.add_argument("--llm_retry_backoff_seconds", type=float, default=5.0)
    parser.add_argument(
        "--llm_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse validated DirectLLM forecasts when the same setting is resumed",
    )
    parser.add_argument(
        "--llm_cache_dir",
        default=None,
        help="optional LLM forecast cache root; defaults inside the setting result dir",
    )

    parser.add_argument(
        "--castfsr_predictor_url",
        default="http://127.0.0.1:8996",
        help="base URL of the official CastFSR numeric predictor service",
    )
    parser.add_argument(
        "--castfsr_predictors",
        default="arima,dlinear,patchtst,itransformer,chronos2",
        help="comma-separated expert pool exposed by the predictor service",
    )
    parser.add_argument(
        "--castfsr_dataset_name",
        default=None,
        help="predictor checkpoint dataset name; defaults to series_id",
    )
    parser.add_argument("--castfsr_predictor_timeout", type=int, default=120)
    parser.add_argument(
        "--castfsr_use_local_experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train/load DLinear and PatchTST inside CastFSRZero instead of HTTP",
    )
    parser.add_argument(
        "--castfsr_expert_checkpoint_dir",
        default=None,
        help="checkpoint root for CastFSRZero's in-process DLinear/PatchTST experts",
    )
    parser.add_argument(
        "--castfsr_dlinear_moving_average",
        type=int,
        default=25,
        help="odd moving-average kernel used by the local DLinear expert",
    )
    parser.add_argument(
        "--castfsr_predictor_preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify the predictor service and requested pool before spending LLM calls",
    )
    parser.add_argument(
        "--castfsr_stage_max_output_tokens",
        type=int,
        default=None,
        help="first CastFSR coordinator request budget; defaults to llm_max_output_tokens",
    )
    parser.add_argument(
        "--castfsr_max_output_tokens",
        type=int,
        default=None,
        help="Slow/Reflect CastFSR coordinator request budget; defaults to llm_max_output_tokens",
    )
    parser.add_argument(
        "--castfsr_retry_max_output_tokens",
        type=int,
        default=None,
        help="CastFSR retry budget; defaults to llm_retry_max_output_tokens",
    )
    parser.add_argument(
        "--castfsr_reflection_max_rounds",
        type=int,
        default=3,
        help="maximum target-blind Reflect repair rounds (1-3)",
    )
    parser.add_argument("--castfsr_temperature", type=float, default=0.6)
    parser.add_argument("--castfsr_top_p", type=float, default=0.7)

    parser.add_argument("--chronos2_model_path", default="amazon/chronos-2")
    parser.add_argument(
        "--chronos2_cache_dir",
        default=str(Path(__file__).resolve().parent / "hf"),
        help="Hugging Face cache root; defaults to world2ts_eval/hf",
    )
    parser.add_argument(
        "--chronos2_local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--chronos2_use_time_covariates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--chronos2_batch_size", type=int, default=256)
    parser.add_argument("--chronos2_context_length", type=int, default=None)
    parser.add_argument(
        "--chronos2_torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )

    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--enc_in", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument("--embed", choices=["timeF", "fixed"], default="timeF")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--results", default="./results")
    parser.add_argument(
        "--use_gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for iteration in range(args.itr):
        setting = "_".join(
            [
                _safe_name(args.model_id),
                args.model,
                _safe_name(args.condition),
                _safe_name(args.series_id),
                f"sl{args.seq_len}",
                f"pl{args.pred_len}",
                "trst1",
                f"evst{args.stride if args.stride is not None else 'auto'}",
                f"i{iteration}",
            ]
        )
        args.llm_cache_namespace = setting
        experiment = Exp_Long_Term_Forecast(args)
        if args.is_training and experiment.model.requires_training:
            experiment.train(setting)
            experiment.test(setting)
        else:
            experiment.test(
                setting,
                load_checkpoint=bool(
                    not args.is_training and experiment.model.requires_training
                ),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

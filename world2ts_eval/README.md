# World2TS Eval

`world2ts_eval` is a compact, forecasting-only evaluation environment for
World2TS Arrow datasets and their published metadata/event context. Its flat
layout, forecasting experiment loop, `Dataset_Custom`, and PatchTST model are
adapted from [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library).
The upstream MIT license is preserved in `LICENSES/Time-Series-Library-MIT.txt`.

## Input

Each input directory contains a regular univariate or channel-separated Arrow
dataset and the three World2TS publication files:

```text
<dataset>/
  data.arrow
  dataset_info.json
  unified_event_series.jsonl
  unified_events.jsonl
```

The evaluator selects one `series_id` per run. Multichannel Arrow targets use
the World2TS IDs `<item_id>__target_<index>`.

`Dataset_Custom` uses a chronological 70/10/20 split and fits its scaler on the
training region only. Training windows always use stride 1; validation and test
windows use the requested rolling-window `stride`. Every sample also carries
the dataset/series description and the `top_k_events` most recent linked events
inside its numeric history window. Test datasets retain only windows containing
at least one such event; training and validation retain all numeric windows.

Source NaNs are interpolated on the timestamp axis before scaling: interior
gaps use linear interpolation, while leading and trailing gaps use the nearest
observed value. A series containing only NaNs is rejected.

## Install

Use a separate Python environment with a Torch build matching the machine:

```bash
pip install -r requirements.txt
```

Chronos-2 is an optional local foundation-model baseline. Install its heavier
dependencies only in the environment used to run it:

```bash
pip install -r requirements-foundation-models.txt
```

Chronos2 loads the official `amazon/chronos-2` 120M checkpoint through
`chronos-forecasting==2.3.1`. It returns the model's median forecast and uses
the history/future calendar features already shared by every evaluation
condition as known covariates. The free-text metadata and events are not valid
Chronos-2 covariates and are intentionally ignored. Its weights are downloaded
from Hugging Face on first use unless `--chronos2_local_files_only` is set. The
default Hugging Face cache root is `world2ts_eval/hf`; downloaded checkpoints
should remain under that directory.

LLM API keys are kept together in the Git-ignored `llm_keys.yaml`:

```yaml
api_keys:
  openai: ""
  deepseek: ""
api_urls:
  openai: "https://api.wintoken.win/v1/responses"  # optional relay
```

DirectLLM and CastFSRZero read this file from the evaluator root by default. Use
`--llm_keys_path` only when the shared key file is stored elsewhere.

DeepSeek DirectLLM uses `deepseek-v4-flash` with thinking enabled at `high`
reasoning effort. Forecasts run with 16 concurrent requests and a 360-second
request timeout by default, a
65,536-token output ceiling, and one retry at 98,304 tokens after a request,
truncation, empty-response, JSON, or forecast-validation failure. Use
`--llm_concurrency`, `--llm_timeout`,
`--llm_max_output_tokens`, and `--llm_retry_max_output_tokens` to override
these defaults.

CastFSRZero inherits these same two DeepSeek output budgets for every
coordinator stage by default. Its `--castfsr_stage_max_output_tokens`,
`--castfsr_max_output_tokens`, and `--castfsr_retry_max_output_tokens` options
remain available as explicit per-experiment overrides.

`run.py` also persists each validated DirectLLM or CastFSRZero forecast under
the current experiment setting. Rerunning the same command reuses completed forecasts and
requests only cache misses. Iterations remain independent because `i0`, `i1`,
and later iterations use separate cache namespaces; prompt and model settings
are part of each cache key. Use `--no-llm_cache` to force fresh requests or
`--llm_cache_dir` to place the cache under a different root.

## Run

Last-value evaluation:

```bash
python -u run.py \
  --is_training 0 \
  --model LastValue \
  --model_id dgs10_naive \
  --root_path ../store/online_discover/macro_fred_dgs10_daily \
  --series_id DGS10 \
  --seq_len 96 \
  --pred_len 12 \
  --stride 24
```

PatchTST training and evaluation:

```bash
python -u run.py \
  --is_training 1 \
  --model PatchTST \
  --model_id dgs10_patchtst \
  --root_path ../store/online_discover/macro_fred_dgs10_daily \
  --series_id DGS10 \
  --seq_len 96 \
  --pred_len 12 \
  --stride 24 \
  --train_epochs 10
```

`seq_len`, `pred_len`, and the validation/test rolling-window `stride` are CLI
parameters. Training always uses stride 1. When validation/test `stride` is
omitted, it defaults to 24 for all frequencies. With a prediction length of
24, successive forecast horizons do not overlap. PatchTST's internal patch
stride is a separate `--patch_stride` parameter.

`--condition` accepts `Numercial Only` (the default) or `With Context`. Naive,
PatchTST, and Chronos2 are numeric-only and therefore ignore the supplied text.
DirectLLM and CastFSRZero consume exactly the numeric, metadata, and event
inputs named by the selected condition. Observation and forecast time ranges
are shared numeric information and are supplied in both conditions where the
model interface supports them. Chronos2 should normally be run once as a
numeric foundation-model baseline rather than repeated across both conditions.

CastFSRZero is a paper-aligned adapter for the released
[CastFSR](https://github.com/Xiaoyu-Tao/CastFSR) Fast-Slow-Reflect workflow, not
a standalone Hugging Face checkpoint. It uses DeepSeek V4 Flash as the
coordinator, executes the released feature/predictor-selection protocol, calls
one selected numeric expert, and then performs Slow contextual adjustment and
target-blind Reflect auditing. DLinear and PatchTST are instantiated, trained,
checkpointed, and invoked in-process by CastFSRZero. ARIMA, iTransformer, and
Chronos-2 use the unified HTTP predictor service. Each instance selects exactly
one expert from the configured pool; a pool may route different instances to
different experts, but the official protocol is not an ensemble.

Train the in-process DLinear and PatchTST experts from the CastFSRZero entry
point before inference on a new series:

```bash
python -u run.py --is_training 1 --model CastFSRZero \
  --model_id dgs10_castfsr_experts --root_path <dataset> --series_id DGS10 \
  --llm_provider deepseek --castfsr_predictors dlinear,patchtst \
  --condition "Numercial Only" --seq_len 96 --pred_len 24 --stride 24 \
  --train_epochs 10
```

Training reads only the numeric history and target from the train/validation
splits, so the resulting artifacts are shared by `Numercial Only` and
`With Context`. By default they are written under
`checkpoints/castfsr_experts/<dataset>/<series>/sl<seq_len>_pl<pred_len>/`.
Inference loads them automatically and fails explicitly when an artifact or
its recorded architecture is incompatible.

Install and start the in-repository service when the configured pool includes
ARIMA, iTransformer, or Chronos-2:

```bash
pip install -r requirements-castfsr.txt
python castfsr_predictor_service.py --device cpu --port 8996 \
  --chronos2_model_path <local-chronos2-directory>
```

The service reproduces the released HTTP contract and
`arima,dlinear,patchtst,itransformer,chronos2` pool. ARIMA needs no checkpoint
and Chronos-2 requires its pretrained weights. iTransformer requires a
Time-Series-Library checkout plus a compatible dataset-specific checkpoint under
`castfsr_models/<model>/<DATASET>/{config.json,checkpoint.pth}`; Chronos-2
uses its configured pretrained model path. DLinear and PatchTST can still be
routed through this service with `--no-castfsr_use_local_experts`, but the
default is the in-process implementation described above.
`/health?dataset_name=<DATASET>` reports per-expert readiness. The evaluator
performs this preflight before spending LLM calls and validates the predictor's
returned timestamp grid.

Missing services, experts, or checkpoints are hard errors and are never
replaced silently. World2TS event text is passed as historical context only,
with no claim that it is future-known exogenous data. Forecast timestamps
contribute future-known calendar fields; no other future exogenous values are
invented.

CastFSRZero evaluation:

```bash
python -u run.py --is_training 0 --model CastFSRZero \
  --model_id dgs10_castfsr_zero --root_path <dataset> --series_id DGS10 \
  --llm_provider deepseek --castfsr_predictors arima,dlinear,patchtst,chronos2 \
  --condition "With Context" --seq_len 96 --pred_len 24 --stride 24
```

Validated final forecasts use the same resume cache as DirectLLM, in a separate
`castfsr_cache` directory. Per-window tool selection, feature output, selected
numeric expert and prior, Slow candidate, deterministic audit diagnostics, and
all Reflect rounds are written to `castfsr_traces` under the experiment result
directory. Reflect uses at most three rounds by default; change this with
`--castfsr_reflection_max_rounds`.

Chronos-2 evaluation:

```bash
python -u run.py --is_training 0 --model Chronos2 \
  --model_id dgs10_chronos2 --root_path <dataset> --series_id DGS10 \
  --condition "Numercial Only" --seq_len 96 --pred_len 24 --stride 24
```

Use `--no-chronos2_use_time_covariates` for a target-only ablation. The default
does not enable cross-learning across rolling windows, because later windows
can contain values lying in earlier windows' forecast horizons.

Outputs are written under:

```text
checkpoints/<setting>/checkpoint.pth
results/<setting>/pred.npy
results/<setting>/true.npy
results/<setting>/metrics.json
```

The initial metrics are MAE, MSE, RMSE, MAPE, and sMAPE. MAPE and sMAPE are
stored as ratios rather than percentages. DirectLLM and CastFSRZero results
also include an `api_usage` object that aggregates requests, retries, failures, prompt/input
tokens, completion/output tokens, reasoning tokens, cache usage, and total
tokens across every API attempt.

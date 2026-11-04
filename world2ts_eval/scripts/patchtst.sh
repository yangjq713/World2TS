#!/usr/bin/env bash
set -euo pipefail

root_path=${1:?"usage: patchtst.sh ROOT_PATH SERIES_ID [extra run.py args]"}
series_id=${2:?"usage: patchtst.sh ROOT_PATH SERIES_ID [extra run.py args]"}
shift 2

python -u run.py \
  --is_training 1 \
  --model PatchTST \
  --model_id patchtst \
  --root_path "$root_path" \
  --series_id "$series_id" \
  "$@"

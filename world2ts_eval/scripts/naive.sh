#!/usr/bin/env bash
set -euo pipefail

root_path=${1:?"usage: naive.sh ROOT_PATH SERIES_ID [extra run.py args]"}
series_id=${2:?"usage: naive.sh ROOT_PATH SERIES_ID [extra run.py args]"}
shift 2

python -u run.py \
  --is_training 0 \
  --model SeasonalNaive \
  --model_id seasonal_naive \
  --root_path "$root_path" \
  --series_id "$series_id" \
  "$@"

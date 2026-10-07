#!/usr/bin/env bash
set -euo pipefail

MIF_LAUNCH_HF_JOB="${MIF_LAUNCH_HF_JOB:-0}"
MIF_GIT_URL="${MIF_GIT_URL:-https://github.com/SaibaWaipu/make-it-flash.git}"
MIF_GIT_REF="${MIF_GIT_REF:-gdn-4.1-pilot}"
MIF_GIT_COMMIT="${MIF_GIT_COMMIT:-}"
MIF_OUTPUT_REPO="${MIF_OUTPUT_REPO:-}"
MIF_APPROVED_BUDGET_USD="${MIF_APPROVED_BUDGET_USD:-}"
MIF_CONFIRMED_CUMULATIVE_SPENT_USD="${MIF_CONFIRMED_CUMULATIVE_SPENT_USD:-}"
ENGINEERING_CUMULATIVE_CAP_USD="9.00"

HF_FLAVOR="${HF_FLAVOR:-a100-large}"
HF_TIMEOUT="${HF_TIMEOUT:-3h}"
HF_JOB_IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime}"
MIF_TOKENS="${MIF_TOKENS:-10000}"
MIF_MAX_SEQ_LEN="${MIF_MAX_SEQ_LEN:-1024}"
MIF_FIT_EPOCHS="${MIF_FIT_EPOCHS:-1}"
MIF_MAX_STEPS="${MIF_MAX_STEPS:-20}"
MIF_LEARNING_RATE="${MIF_LEARNING_RATE:-1e-4}"
MIF_LAYER="${MIF_LAYER:-0}"
MIF_MODEL="${MIF_MODEL:-llm-jp/llm-jp-4.1-32b-a3b-thinking}"
MIF_MODEL_REVISION="${MIF_MODEL_REVISION:-cda260706786758045e5e96bf4d738bbc01155b5}"
MIF_JOB_NAME="${MIF_JOB_NAME:-llm-jp-41-gdn-layer-${MIF_LAYER}}"

case "$MIF_LAUNCH_HF_JOB" in
  0) ;;
  1) ;;
  *) echo "MIF_LAUNCH_HF_JOB must be 0 (dry-run) or 1 (launch)" >&2; exit 2 ;;
esac

HARDWARE_JSON="$(hf jobs hardware --json)"
COST_INFO="$(HF_HARDWARE_JSON="$HARDWARE_JSON" python - "$HF_FLAVOR" "$HF_TIMEOUT" <<'PY'
import json
import os
import re
import sys
from decimal import Decimal, ROUND_CEILING

flavor, timeout = sys.argv[1:]
hardware = json.loads(os.environ["HF_HARDWARE_JSON"])
entry = next((item for item in hardware if item["name"] == flavor), None)
if entry is None:
    raise SystemExit(f"unknown HF Jobs hardware flavor: {flavor}")
match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhd])", timeout)
if match is None:
    raise SystemExit("HF_TIMEOUT must be a number followed by s, m, h, or d")
amount, unit = match.groups()
seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
seconds = Decimal(amount) * seconds_per_unit
rate = Decimal(re.sub(r"[^0-9.]", "", entry["cost/hour"]))
estimate = (rate * seconds / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
print(f"{rate}\t{estimate}")
PY
)"
IFS=$'\t' read -r HF_RATE_USD_PER_HOUR HF_ESTIMATED_COST_USD <<<"$COST_INFO"
printf 'HF Job plan: flavor=%s timeout=%s rate=$%s/hour estimated_max_cost=$%s tokens=%s max_seq_len=%s layer=%s max_steps=%s\n' \
  "$HF_FLAVOR" "$HF_TIMEOUT" "$HF_RATE_USD_PER_HOUR" "$HF_ESTIMATED_COST_USD" "$MIF_TOKENS" "$MIF_MAX_SEQ_LEN" "$MIF_LAYER" "$MIF_MAX_STEPS"
printf 'Cumulative engineering compute cap: $%s (pilot included)\n' "$ENGINEERING_CUMULATIVE_CAP_USD"

if [[ "$MIF_LAUNCH_HF_JOB" == "0" ]]; then
  echo "DRY RUN ONLY: no HF Job was submitted. Launch requires explicit approval, per-job budget, and invoice-confirmed cumulative prior spend."
  exit 0
fi

: "${MIF_APPROVED_BUDGET_USD:?Set MIF_APPROVED_BUDGET_USD to the approved per-job spending limit}"
: "${MIF_CONFIRMED_CUMULATIVE_SPENT_USD:?Set the invoice-confirmed prior spend, including the original pilot}"
: "${MIF_OUTPUT_REPO:?Set MIF_OUTPUT_REPO to a private Hub model repo for the fitted layer}"
: "${MIF_GIT_COMMIT:?Set MIF_GIT_COMMIT to the exact approved source commit}"
if [[ ! "$MIF_GIT_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "MIF_GIT_COMMIT must be an exact 40-character commit SHA" >&2
  exit 2
fi
if [[ ! "$MIF_MODEL_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "MIF_MODEL_REVISION must be a pinned 40-character commit SHA" >&2
  exit 2
fi
if ! python - "$MIF_APPROVED_BUDGET_USD" "$HF_ESTIMATED_COST_USD" "$MIF_CONFIRMED_CUMULATIVE_SPENT_USD" "$ENGINEERING_CUMULATIVE_CAP_USD" <<'PY'
from decimal import Decimal, InvalidOperation
import sys

try:
    budget, estimate, prior_spend, cap = map(Decimal, sys.argv[1:])
except InvalidOperation:
    raise SystemExit("budget, estimate, prior spend, and cap must be valid dollar amounts")
if not all(value.is_finite() for value in (budget, estimate, prior_spend, cap)):
    raise SystemExit("cost guard values must be finite dollar amounts")
if budget <= 0 or prior_spend < 0 or prior_spend > cap:
    raise SystemExit(f"refusing launch: invalid budget or confirmed prior spend ${prior_spend} against cap ${cap}")
remaining = cap - prior_spend
if budget > remaining:
    raise SystemExit(f"refusing launch: per-job budget ${budget} exceeds remaining cumulative cap ${remaining}")
if estimate > budget:
    raise SystemExit(f"refusing launch: estimated ${estimate} exceeds approved per-job budget ${budget}")
PY
then
  echo "Cost guard refused HF Job launch." >&2
  exit 3
fi

hf jobs run --detach \
  --label "name=$MIF_JOB_NAME" \
  --label "purpose=llm-jp-41-gdn-pilot" \
  --label "cost_guarded=true" \
  --flavor "$HF_FLAVOR" \
  --timeout "$HF_TIMEOUT" \
  --secrets HF_TOKEN \
  --env "MIF_GIT_URL=$MIF_GIT_URL" \
  --env "MIF_GIT_REF=$MIF_GIT_REF" \
  --env "MIF_GIT_COMMIT=$MIF_GIT_COMMIT" \
  --env "MIF_OUTPUT_REPO=$MIF_OUTPUT_REPO" \
  --env "MIF_TOKENS=$MIF_TOKENS" \
  --env "MIF_MAX_SEQ_LEN=$MIF_MAX_SEQ_LEN" \
  --env "MIF_FIT_EPOCHS=$MIF_FIT_EPOCHS" \
  --env "MIF_MAX_STEPS=$MIF_MAX_STEPS" \
  --env "MIF_LEARNING_RATE=$MIF_LEARNING_RATE" \
  --env "MIF_LAYER=$MIF_LAYER" \
  --env "MIF_MODEL=$MIF_MODEL" \
  --env "MIF_MODEL_REVISION=$MIF_MODEL_REVISION" \
  "$HF_JOB_IMAGE" bash -c 'set -euo pipefail
  if ! command -v git >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "git and apt-get are unavailable in the selected job image" >&2
      exit 127
    fi
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git
  fi
  git clone --depth 1 --branch "$MIF_GIT_REF" "$MIF_GIT_URL" /workspace/make_it_flash
  cd /workspace/make_it_flash
  source_commit="$(git rev-parse HEAD)"
  printf "Source commit: %s\n" "$source_commit"
  if [[ "$source_commit" != "$MIF_GIT_COMMIT" ]]; then
    echo "cloned source commit does not match approved commit" >&2
    exit 4
  fi
  python -m pip install -e .
  make-it-flash prepare --model "$MIF_MODEL" --model-revision "$MIF_MODEL_REVISION" --output-dir artifacts/data --max-tokens "$MIF_TOKENS" --max-seq-len "$MIF_MAX_SEQ_LEN"
  make-it-flash cache --data-file artifacts/data/calibration.jsonl --output-dir artifacts/cache --layers "$MIF_LAYER"
  make-it-flash fit --cache-dir artifacts/cache --output-dir artifacts/gdn --layer "$MIF_LAYER" --epochs "$MIF_FIT_EPOCHS" --max-steps "$MIF_MAX_STEPS" --learning-rate "$MIF_LEARNING_RATE"
  python scripts/publish_hf_artifact.py artifacts/gdn'

#!/usr/bin/env bash
# Full 24 GDN + 8 QSA conversion: cost-checked dry-run by default.
set -euo pipefail

MIF_LAUNCH_HF_JOB="${MIF_LAUNCH_HF_JOB:-0}"
MIF_GIT_URL="${MIF_GIT_URL:-https://github.com/SaibaWaipu/make-it-flash.git}"
MIF_GIT_REF="${MIF_GIT_REF:-gdn-4.1-pilot}"
MIF_GIT_COMMIT="${MIF_GIT_COMMIT:-}"
MIF_MODEL_REVISION="${MIF_MODEL_REVISION:-cda260706786758045e5e96bf4d738bbc01155b5}"
MIF_OUTPUT_REPO="${MIF_OUTPUT_REPO:-RemydreScarlet/llm-jp-4.1-flash-next}"
MIF_APPROVED_INCREMENTAL_BUDGET_USD="${MIF_APPROVED_INCREMENTAL_BUDGET_USD:-}"
MIF_EXPECTED_OUTPUT_REPO_SHA="${MIF_EXPECTED_OUTPUT_REPO_SHA:-}"
HF_FLAVOR="${HF_FLAVOR:-a100-large}"
HF_TIMEOUT="${HF_TIMEOUT:-3h}"
HF_JOB_IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime}"
MIF_TOKENS="${MIF_TOKENS:-100000}"
MIF_MAX_SEQ_LEN="${MIF_MAX_SEQ_LEN:-4096}"
MIF_FIT_EPOCHS="${MIF_FIT_EPOCHS:-3}"
MIF_MAX_STEPS="${MIF_MAX_STEPS:-200}"
MIF_VALIDATION_FRACTION="${MIF_VALIDATION_FRACTION:-0.1}"
MIF_DATASET_REVISION="${MIF_DATASET_REVISION:-main}"
MIF_EVAL_DATASET="${MIF_EVAL_DATASET:-}"
MIF_EVAL_SPLIT="${MIF_EVAL_SPLIT:-}"
MIF_EVAL_DATASET_REVISION="${MIF_EVAL_DATASET_REVISION:-main}"
MIF_EVAL_TOKENS="${MIF_EVAL_TOKENS:-10000}"
MIF_ALLOW_UNEVALUATED_UPLOAD="${MIF_ALLOW_UNEVALUATED_UPLOAD:-0}"
MIF_RESUME="${MIF_RESUME:-0}"

case "$MIF_LAUNCH_HF_JOB" in 0|1) ;; *) echo 'MIF_LAUNCH_HF_JOB must be 0 or 1' >&2; exit 2;; esac
case "$MIF_RESUME" in 0|1) ;; *) echo 'MIF_RESUME must be 0 or 1' >&2; exit 2;; esac
case "$MIF_ALLOW_UNEVALUATED_UPLOAD" in 0|1) ;; *) echo 'MIF_ALLOW_UNEVALUATED_UPLOAD must be 0 or 1' >&2; exit 2;; esac
hardware_json="$(hf jobs hardware --json)"
cost_info="$(HF_HARDWARE_JSON="$hardware_json" python - "$HF_FLAVOR" "$HF_TIMEOUT" <<'PY'
import json, os, re, sys
from decimal import Decimal, ROUND_CEILING
flavor, timeout = sys.argv[1:]
entry = next((x for x in json.loads(os.environ['HF_HARDWARE_JSON']) if x['name'] == flavor), None)
if entry is None: raise SystemExit('unknown hardware flavor')
match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([smhd])', timeout)
if match is None: raise SystemExit('HF_TIMEOUT must be number followed by s/m/h/d')
amount, unit = match.groups()
seconds = Decimal(amount) * {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[unit]
rate = Decimal(re.sub('[^0-9.]', '', entry['cost/hour']))
maximum = (rate * seconds / 3600).quantize(Decimal('0.01'), rounding=ROUND_CEILING)
print(f'{rate}\t{maximum}')
PY
)"
IFS=$'\t' read -r hf_rate max_cost <<<"$cost_info"
printf 'FULL 32-LAYER HF Job plan: %s (%s), max compute cost $%s, model=%s, output=%s\n' \
  "$HF_FLAVOR" "$HF_TIMEOUT" "$max_cost" "$MIF_MODEL_REVISION" "$MIF_OUTPUT_REPO"
echo 'Includes prepare, eight teacher loads (one QSA attention layer each), 32 fits, assembly, and optionally evaluation.'
echo 'No quality guarantee or guaranteed completion within timeout. Paid GPU time includes downloads and installation.'
if [[ "$MIF_LAUNCH_HF_JOB" == 0 ]]; then
  echo 'DRY RUN ONLY: no HF Job submitted.'
  exit 0
fi
: "${MIF_GIT_COMMIT:?Pin an uploaded, reviewed full-run source commit}"
: "${MIF_APPROVED_INCREMENTAL_BUDGET_USD:?Approve this job incremental compute cost}"
: "${MIF_EXPECTED_OUTPUT_REPO_SHA:?Pin the existing private output repo commit to avoid overwriting unrelated content}"
[[ "$MIF_GIT_COMMIT" =~ ^[0-9a-fA-F]{40}$ && "$MIF_MODEL_REVISION" =~ ^[0-9a-fA-F]{40}$ ]] || { echo 'Source and teacher revisions must be pinned 40-character SHAs' >&2; exit 2; }
[[ "$MIF_EXPECTED_OUTPUT_REPO_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || { echo 'Existing output repo SHA must be pinned' >&2; exit 2; }
python - "$MIF_APPROVED_INCREMENTAL_BUDGET_USD" "$max_cost" <<'PY'
from decimal import Decimal, InvalidOperation
import sys
try: approved, maximum = map(Decimal, sys.argv[1:])
except InvalidOperation: raise SystemExit('budget must be numeric')
if not approved.is_finite() or approved <= 0 or approved > 9 or maximum > approved:
    raise SystemExit(f'refusing job: max ${maximum} exceeds approved incremental budget ${approved}, or approved > $9')
PY
[[ "$MIF_OUTPUT_REPO" == 'RemydreScarlet/llm-jp-4.1-flash-next' ]] || { echo 'Output repo must be the approved private RemydreScarlet repo' >&2; exit 2; }
if [[ -z "$MIF_EVAL_DATASET" || -z "$MIF_EVAL_SPLIT" ]]; then
  [[ "$MIF_ALLOW_UNEVALUATED_UPLOAD" == 1 ]] || { echo 'Provide an independent evaluation dataset/split, or explicitly permit unevaluated upload' >&2; exit 2; }
fi
python - "$MIF_OUTPUT_REPO" "$MIF_EXPECTED_OUTPUT_REPO_SHA" <<'PY'
import sys
from huggingface_hub import HfApi
info = HfApi().model_info(sys.argv[1])
if not info.private or info.sha != sys.argv[2]: raise SystemExit('private repo or expected base SHA mismatch')
PY
if [[ "$MIF_RESUME" == 1 ]]; then
  echo 'Remote resume requires persisted calibration/caches; ephemeral Jobs do not preserve them. Refusing unsupported resume.' >&2
  exit 2
fi
hf jobs run --detach --namespace RemydreScarlet --flavor "$HF_FLAVOR" --timeout "$HF_TIMEOUT" \
  --label name=llm-jp-41-flash-next-full --label purpose=full-32-layer-local-distillation \
  --secrets HF_TOKEN --env "MIF_GIT_URL=$MIF_GIT_URL" --env "MIF_GIT_REF=$MIF_GIT_REF" \
  --env "MIF_GIT_COMMIT=$MIF_GIT_COMMIT" --env "MIF_MODEL_REVISION=$MIF_MODEL_REVISION" \
  --env "MIF_OUTPUT_REPO=$MIF_OUTPUT_REPO" --env "MIF_EXPECTED_OUTPUT_REPO_SHA=$MIF_EXPECTED_OUTPUT_REPO_SHA" \
  --env "MIF_TOKENS=$MIF_TOKENS" --env "MIF_MAX_SEQ_LEN=$MIF_MAX_SEQ_LEN" \
  --env "MIF_FIT_EPOCHS=$MIF_FIT_EPOCHS" --env "MIF_MAX_STEPS=$MIF_MAX_STEPS" \
  --env "MIF_VALIDATION_FRACTION=$MIF_VALIDATION_FRACTION" --env "MIF_DATASET_REVISION=$MIF_DATASET_REVISION" \
  --env "MIF_EVAL_DATASET=$MIF_EVAL_DATASET" --env "MIF_EVAL_SPLIT=$MIF_EVAL_SPLIT" \
  --env "MIF_EVAL_DATASET_REVISION=$MIF_EVAL_DATASET_REVISION" --env "MIF_EVAL_TOKENS=$MIF_EVAL_TOKENS" \
  --env "MIF_ALLOW_UNEVALUATED_UPLOAD=$MIF_ALLOW_UNEVALUATED_UPLOAD" \
  "$HF_JOB_IMAGE" bash -c 'set -euo pipefail
  if ! command -v git >/dev/null; then apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git; fi
  git clone --depth 1 --branch "$MIF_GIT_REF" "$MIF_GIT_URL" /workspace/make_it_flash
  cd /workspace/make_it_flash
  [[ "$(git rev-parse HEAD)" == "$MIF_GIT_COMMIT" ]] || { echo "source commit mismatch" >&2; exit 4; }
  python -m pip install -e .
  make-it-flash prepare --model-revision "$MIF_MODEL_REVISION" --dataset-revision "$MIF_DATASET_REVISION" --output-dir artifacts/full-data --max-tokens "$MIF_TOKENS" --max-seq-len "$MIF_MAX_SEQ_LEN"
  make-it-flash full-run --data-file artifacts/full-data/calibration.jsonl --output-dir artifacts/full-run --epochs "$MIF_FIT_EPOCHS" --max-steps "$MIF_MAX_STEPS" --validation-fraction "$MIF_VALIDATION_FRACTION"
  if [[ -n "$MIF_EVAL_DATASET" && -n "$MIF_EVAL_SPLIT" ]]; then
    make-it-flash prepare --model-revision "$MIF_MODEL_REVISION" --dataset "$MIF_EVAL_DATASET" --dataset-revision "$MIF_EVAL_DATASET_REVISION" --split "$MIF_EVAL_SPLIT" --output-dir artifacts/heldout --max-tokens "$MIF_EVAL_TOKENS" --max-seq-len "$MIF_MAX_SEQ_LEN"
    make-it-flash evaluate --model-revision "$MIF_MODEL_REVISION" --data-file artifacts/heldout/calibration.jsonl --overlay-dir artifacts/full-run/overlay --output-file artifacts/full-run/evaluation.json
  fi
  if [[ -f artifacts/full-run/evaluation.json ]]; then
    python scripts/publish_full_hf_artifact.py artifacts/full-run/overlay artifacts/full-run/evaluation.json
  else
    [[ "$MIF_ALLOW_UNEVALUATED_UPLOAD" == 1 ]] || { echo "independent evaluation report is required" >&2; exit 6; }
    python scripts/publish_full_hf_artifact.py artifacts/full-run/overlay
  fi'

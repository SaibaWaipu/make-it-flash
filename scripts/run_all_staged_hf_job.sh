#!/usr/bin/env bash
# Submit one long-running HF Job that fits 32 layers sequentially and commits each one.
set -euo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_ROOT"
export PYTHONPATH="$SCRIPT_ROOT/src:$SCRIPT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MIF_LAUNCH_HF_JOB="${MIF_LAUNCH_HF_JOB:-0}"
MIF_RUN_ID="${MIF_RUN_ID:-}"
MIF_GIT_URL="${MIF_GIT_URL:-https://github.com/SaibaWaipu/make-it-flash.git}"
MIF_GIT_REF="${MIF_GIT_REF:-gdn-4.1-pilot}"
MIF_GIT_COMMIT="${MIF_GIT_COMMIT:-}"
MIF_OUTPUT_REPO="${MIF_OUTPUT_REPO:-RemydreScarlet/llm-jp-4.1-flash-next}"
MIF_EXPECTED_OUTPUT_REPO_SHA="${MIF_EXPECTED_OUTPUT_REPO_SHA:-}"
MIF_APPROVED_INCREMENTAL_BUDGET_USD="${MIF_APPROVED_INCREMENTAL_BUDGET_USD:-}"
MIF_CONFIRMED_THIS_RUN_SPEND_USD="${MIF_CONFIRMED_THIS_RUN_SPEND_USD:-}"
MIF_CALIBRATION_DATA_SHA="${MIF_CALIBRATION_DATA_SHA:-aa0500f9d21a4251d9dbf9e7158857cec78f39ae4b902a986c7a5abac6fde706}"
MIF_CALIBRATION_MANIFEST_SHA="${MIF_CALIBRATION_MANIFEST_SHA:-a728725277d56c6d2a7d639d1e4ffb9677bd907474a2e6906bf134096bc4c452}"
MIF_CALIBRATION_REPO="${MIF_CALIBRATION_REPO:-RemydreScarlet/llm-jp-4.1-flash-next-calibration}"
MIF_CALIBRATION_REVISION="${MIF_CALIBRATION_REVISION:-22c44496a7d708bca4986f80f1478e51aebad67a}"
MIF_RESUME="${MIF_RESUME:-0}"
HF_FLAVOR="${HF_FLAVOR:-a100-large}"
HF_TIMEOUT="${HF_TIMEOUT:-3h}"
HF_JOB_IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime}"
MIF_FIT_EPOCHS="${MIF_FIT_EPOCHS:-3}"
MIF_MAX_STEPS="${MIF_MAX_STEPS:-200}"
MIF_VALIDATION_FRACTION="${MIF_VALIDATION_FRACTION:-0.1}"

case "$MIF_LAUNCH_HF_JOB" in 0|1) ;; *) echo 'MIF_LAUNCH_HF_JOB must be 0 or 1' >&2; exit 2;; esac
case "$MIF_RESUME" in 0|1) ;; *) echo 'MIF_RESUME must be 0 or 1' >&2; exit 2;; esac
: "${MIF_RUN_ID:?Choose a stable run ID (required even for dry-run)}"
python - "$MIF_RUN_ID" <<'PY'
import sys
from scripts.publish_staged_layer import artifact_paths
for layer in range(32):
    prefix, checkpoint, metrics = artifact_paths(sys.argv[1], layer)
    print(f'{layer:02d}: {prefix}/{checkpoint} + {metrics}')
PY
hardware_json="$(hf jobs hardware --json)"
cost_info="$(HF_HARDWARE_JSON="$hardware_json" python - "$HF_FLAVOR" "$HF_TIMEOUT" <<'PY'
import json, os, re, sys
from decimal import Decimal, ROUND_CEILING
entry = next((e for e in json.loads(os.environ['HF_HARDWARE_JSON']) if e['name'] == sys.argv[1]), None)
if entry is None: raise SystemExit('unknown GPU flavor')
match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([smhd])', sys.argv[2])
if match is None: raise SystemExit('HF_TIMEOUT must use s/m/h/d')
amount, unit = match.groups()
seconds = Decimal(amount) * {'s':1,'m':60,'h':3600,'d':86400}[unit]
rate = Decimal(re.sub('[^0-9.]', '', entry['cost/hour']))
maximum = (rate * seconds / 3600).quantize(Decimal('0.01'), rounding=ROUND_CEILING)
print(f'{rate}\t{maximum}')
PY
)"
IFS=$'\t' read -r hf_rate max_cost <<<"$cost_info"
printf 'ONE HF JOB: 32 sequential layer fits at %s, $%s/hour; maximum compute cost $%s.\n' "$HF_FLAVOR" "$hf_rate" "$max_cost"
echo 'Each fit is validated and atomically committed to the private model repo before the next layer starts.'
echo 'Full schedule uses eight teacher captures. A timeout/failure preserves completed commits; remaining layers can resume with the same run ID.'
if [[ "$MIF_LAUNCH_HF_JOB" == 0 ]]; then echo 'DRY RUN ONLY: no HF Job submitted and no Hub write.'; exit 0; fi
: "${MIF_GIT_COMMIT:?Pin the uploaded source commit}"
: "${MIF_EXPECTED_OUTPUT_REPO_SHA:?Pin the current private model repo parent commit}"
: "${MIF_APPROVED_INCREMENTAL_BUDGET_USD:?Set approved total budget for this single job (maximum $9)}"
: "${MIF_CONFIRMED_THIS_RUN_SPEND_USD:?Enter confirmed billed spend so far in this run}"
python - "$MIF_GIT_COMMIT" "$MIF_EXPECTED_OUTPUT_REPO_SHA" "$MIF_CALIBRATION_REVISION" "$MIF_CALIBRATION_DATA_SHA" "$MIF_CALIBRATION_MANIFEST_SHA" "$MIF_APPROVED_INCREMENTAL_BUDGET_USD" "$MIF_CONFIRMED_THIS_RUN_SPEND_USD" "$max_cost" <<'PY'
from decimal import Decimal, InvalidOperation
import re, sys
source, output, calibration, data, manifest, budget, spent, maximum = sys.argv[1:]
if any(re.fullmatch(r'[0-9a-fA-F]{40}', sha) is None for sha in (source, output, calibration)):
    raise SystemExit('source/output/dataset revisions must be pinned 40-character SHAs')
if any(re.fullmatch(r'[0-9a-f]{64}', sha) is None for sha in (data, manifest)):
    raise SystemExit('calibration digests must be pinned 64-character SHA-256 values')
try: approved, cumulative, estimate = map(Decimal, (budget, spent, maximum))
except InvalidOperation: raise SystemExit('budget values must be decimal')
if (any(not x.is_finite() for x in (approved,cumulative,estimate)) or approved <= 0
        or cumulative < 0 or approved + cumulative > 9 or estimate > approved):
    raise SystemExit('refusing: this job maximum plus confirmed previous spend exceeds the approved $9 ceiling')
PY
[[ "$MIF_OUTPUT_REPO" == 'RemydreScarlet/llm-jp-4.1-flash-next' ]] || { echo 'Output repo must be the approved private repo' >&2; exit 2; }
python - "$MIF_OUTPUT_REPO" "$MIF_EXPECTED_OUTPUT_REPO_SHA" "$MIF_CALIBRATION_REPO" "$MIF_CALIBRATION_REVISION" <<'PY'
import sys
from huggingface_hub import HfApi
api=HfApi()
model=api.model_info(sys.argv[1])
if not model.private or model.sha != sys.argv[2]: raise SystemExit('private model repo or starting HEAD mismatch')
data=api.dataset_info(sys.argv[3], revision=sys.argv[4])
if not data.private or data.sha != sys.argv[4]: raise SystemExit('calibration repo must be private and pinned')
PY
hf jobs run --detach --namespace RemydreScarlet --flavor "$HF_FLAVOR" --timeout "$HF_TIMEOUT" \
  --label "name=flash-next-all-layers-$MIF_RUN_ID" --label "purpose=single-job-progressive-32-layer-fit" \
  --secrets HF_TOKEN --env "MIF_GIT_URL=$MIF_GIT_URL" --env "MIF_GIT_REF=$MIF_GIT_REF" \
  --env "MIF_GIT_COMMIT=$MIF_GIT_COMMIT" --env "MIF_RUN_ID=$MIF_RUN_ID" \
  --env "MIF_EXPECTED_OUTPUT_REPO_SHA=$MIF_EXPECTED_OUTPUT_REPO_SHA" \
  --env "MIF_CALIBRATION_DATA_SHA=$MIF_CALIBRATION_DATA_SHA" \
  --env "MIF_CALIBRATION_MANIFEST_SHA=$MIF_CALIBRATION_MANIFEST_SHA" \
  --env "MIF_CALIBRATION_REPO=$MIF_CALIBRATION_REPO" --env "MIF_CALIBRATION_REVISION=$MIF_CALIBRATION_REVISION" \
  --env "MIF_RESUME=$MIF_RESUME" --env "MIF_FIT_EPOCHS=$MIF_FIT_EPOCHS" \
  --env "MIF_MAX_STEPS=$MIF_MAX_STEPS" --env "MIF_VALIDATION_FRACTION=$MIF_VALIDATION_FRACTION" \
  "$HF_JOB_IMAGE" bash -c 'set -euo pipefail
    if ! command -v git >/dev/null 2>&1; then apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git; fi
    git clone --depth 1 --branch "$MIF_GIT_REF" "$MIF_GIT_URL" /workspace/make_it_flash
    cd /workspace/make_it_flash
    [[ "$(git rev-parse HEAD)" == "$MIF_GIT_COMMIT" ]] || { echo "source commit mismatch" >&2; exit 4; }
    python -m pip install -e .
    python - <<"PY"
import hashlib, os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
repo=os.environ["MIF_CALIBRATION_REPO"]; revision=os.environ["MIF_CALIBRATION_REVISION"]
info=HfApi().dataset_info(repo, revision=revision)
if not info.private or info.sha != revision: raise SystemExit("calibration repo must be private and pinned")
root=Path("artifacts/staged-data"); root.mkdir(parents=True, exist_ok=True)
for name, env in (("calibration.jsonl","MIF_CALIBRATION_DATA_SHA"),("data_manifest.json","MIF_CALIBRATION_MANIFEST_SHA")):
    payload=Path(hf_hub_download(repo, filename=name, repo_type="dataset", revision=revision)).read_bytes()
    if hashlib.sha256(payload).hexdigest() != os.environ[env]: raise SystemExit(f"pinned {name} hash mismatch")
    (root/name).write_bytes(payload)
PY
    args=(--data-file artifacts/staged-data/calibration.jsonl --output-dir artifacts/progressive-run
          --run-id "$MIF_RUN_ID" --expected-repo-sha "$MIF_EXPECTED_OUTPUT_REPO_SHA"
          --source-commit "$MIF_GIT_COMMIT" --epochs "$MIF_FIT_EPOCHS" --max-steps "$MIF_MAX_STEPS"
          --validation-fraction "$MIF_VALIDATION_FRACTION")
    if [[ "$MIF_RESUME" == 1 ]]; then args+=(--resume); fi
    python scripts/run_all_staged_conversion.py "${args[@]}"'

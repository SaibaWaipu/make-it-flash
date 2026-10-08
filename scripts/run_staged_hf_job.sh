#!/usr/bin/env bash
# Paid job for one specified layer; default is a read-only cost estimate.
set -euo pipefail

MIF_LAUNCH_HF_JOB="${MIF_LAUNCH_HF_JOB:-0}"
MIF_LAYER="${MIF_LAYER:-}"
MIF_RUN_ID="${MIF_RUN_ID:-}"
MIF_GIT_URL="${MIF_GIT_URL:-https://github.com/SaibaWaipu/make-it-flash.git}"
MIF_GIT_REF="${MIF_GIT_REF:-gdn-4.1-pilot}"
MIF_GIT_COMMIT="${MIF_GIT_COMMIT:-}"
MIF_MODEL_REVISION="${MIF_MODEL_REVISION:-cda260706786758045e5e96bf4d738bbc01155b5}"
MIF_OUTPUT_REPO="${MIF_OUTPUT_REPO:-RemydreScarlet/llm-jp-4.1-flash-next}"
MIF_EXPECTED_OUTPUT_REPO_SHA="${MIF_EXPECTED_OUTPUT_REPO_SHA:-}"
MIF_APPROVED_INCREMENTAL_BUDGET_USD="${MIF_APPROVED_INCREMENTAL_BUDGET_USD:-}"
MIF_CONFIRMED_THIS_RUN_SPEND_USD="${MIF_CONFIRMED_THIS_RUN_SPEND_USD:-}"
HF_FLAVOR="${HF_FLAVOR:-a100-large}"
HF_TIMEOUT="${HF_TIMEOUT:-30m}"
HF_JOB_IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime}"
MIF_FIT_EPOCHS="${MIF_FIT_EPOCHS:-3}"
MIF_MAX_STEPS="${MIF_MAX_STEPS:-200}"
MIF_VALIDATION_FRACTION="${MIF_VALIDATION_FRACTION:-0.1}"
MIF_CALIBRATION_DATA_SHA="${MIF_CALIBRATION_DATA_SHA:-}"
MIF_CALIBRATION_MANIFEST_SHA="${MIF_CALIBRATION_MANIFEST_SHA:-}"
MIF_CALIBRATION_REPO="${MIF_CALIBRATION_REPO:-}"
MIF_CALIBRATION_REVISION="${MIF_CALIBRATION_REVISION:-}"

case "$MIF_LAUNCH_HF_JOB" in 0|1) ;; *) echo 'MIF_LAUNCH_HF_JOB must be 0 or 1' >&2; exit 2;; esac
: "${MIF_LAYER:?Select a layer index 0-31 (required even for dry-run)}"
: "${MIF_RUN_ID:?Choose a stable run ID (required even for dry-run)}"
python - "$MIF_LAYER" "$MIF_RUN_ID" <<'PY'
import sys
from scripts.publish_staged_layer import artifact_paths
layer = int(sys.argv[1])
prefix, checkpoint, metrics = artifact_paths(sys.argv[2], layer)
print(f'Layer plan: {layer} -> {prefix}/{checkpoint} and {metrics}')
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
printf 'STAGED JOB: %s %s at $%s/h; this job maximum $%s.\n' "$HF_FLAVOR" "$HF_TIMEOUT" "$hf_rate" "$max_cost"
echo 'Each Job downloads pinned private calibration data, loads the 32B teacher, fits one layer, then conditionally uploads one atomic layer record.'
if [[ "$MIF_LAUNCH_HF_JOB" == 0 ]]; then echo 'DRY RUN ONLY: no job or Hub write.'; exit 0; fi
: "${MIF_GIT_COMMIT:?Pin the committed source branch before launch}"
: "${MIF_EXPECTED_OUTPUT_REPO_SHA:?Pin the private output repo parent commit}"
: "${MIF_APPROVED_INCREMENTAL_BUDGET_USD:?Set approved per-job budget <= $9}"
: "${MIF_CONFIRMED_THIS_RUN_SPEND_USD:?Enter confirmed billed spend for this run so far}"
: "${MIF_CALIBRATION_DATA_SHA:?Pin the intended calibration JSONL SHA for every layer}"
: "${MIF_CALIBRATION_MANIFEST_SHA:?Pin the intended calibration manifest SHA for every layer}"
: "${MIF_CALIBRATION_REPO:?Specify an authorized private calibration dataset repo}"
: "${MIF_CALIBRATION_REVISION:?Pin the private calibration repo commit}"
python - "$MIF_GIT_COMMIT" "$MIF_MODEL_REVISION" "$MIF_EXPECTED_OUTPUT_REPO_SHA" "$MIF_CALIBRATION_DATA_SHA" "$MIF_CALIBRATION_MANIFEST_SHA" "$MIF_CALIBRATION_REVISION" "$MIF_APPROVED_INCREMENTAL_BUDGET_USD" "$MIF_CONFIRMED_THIS_RUN_SPEND_USD" "$max_cost" <<'PY'
from decimal import Decimal, InvalidOperation
import re, sys
source, teacher, repo, data, manifest, calibration_revision, budget, spent, maximum = sys.argv[1:]
if any(re.fullmatch('[0-9a-fA-F]{40}', sha) is None for sha in (source,teacher,repo,calibration_revision)) or any(re.fullmatch('[0-9a-f]{64}', sha) is None for sha in (data,manifest)):
    raise SystemExit('all revisions and calibration digests must be pinned')
try: approved, cumulative, estimate = map(Decimal, (budget,spent,maximum))
except InvalidOperation: raise SystemExit('budget values must be decimal')
if any(not x.is_finite() for x in (approved,cumulative,estimate)) or approved <= 0 or cumulative < 0 or approved + cumulative > 9 or estimate > approved:
    raise SystemExit('refusing: per-job cost + confirmed spent exceeds the additional $9 budget')
PY
[[ "$MIF_OUTPUT_REPO" == 'RemydreScarlet/llm-jp-4.1-flash-next' ]] || { echo 'Output repo must be approved private repo' >&2; exit 2; }
python - "$MIF_OUTPUT_REPO" "$MIF_EXPECTED_OUTPUT_REPO_SHA" <<'PY'
import sys
from huggingface_hub import HfApi
info = HfApi().model_info(sys.argv[1])
if not info.private or info.sha != sys.argv[2]: raise SystemExit('private output repo or expected HEAD mismatch')
PY
hf jobs run --detach --namespace RemydreScarlet --flavor "$HF_FLAVOR" --timeout "$HF_TIMEOUT" \
  --label "name=flash-next-layer-$MIF_LAYER" --label "purpose=staged-local-attention-fit" \
  --secrets HF_TOKEN --env "MIF_GIT_URL=$MIF_GIT_URL" --env "MIF_GIT_REF=$MIF_GIT_REF" \
  --env "MIF_GIT_COMMIT=$MIF_GIT_COMMIT" --env "MIF_LAYER=$MIF_LAYER" --env "MIF_RUN_ID=$MIF_RUN_ID" \
  --env "MIF_MODEL_REVISION=$MIF_MODEL_REVISION" --env "MIF_OUTPUT_REPO=$MIF_OUTPUT_REPO" \
  --env "MIF_EXPECTED_OUTPUT_REPO_SHA=$MIF_EXPECTED_OUTPUT_REPO_SHA" \
  --env "MIF_CALIBRATION_DATA_SHA=$MIF_CALIBRATION_DATA_SHA" \
  --env "MIF_CALIBRATION_MANIFEST_SHA=$MIF_CALIBRATION_MANIFEST_SHA" \
  --env "MIF_CALIBRATION_REPO=$MIF_CALIBRATION_REPO" --env "MIF_CALIBRATION_REVISION=$MIF_CALIBRATION_REVISION" \
  --env "MIF_FIT_EPOCHS=$MIF_FIT_EPOCHS" --env "MIF_MAX_STEPS=$MIF_MAX_STEPS" \
  --env "MIF_VALIDATION_FRACTION=$MIF_VALIDATION_FRACTION" \
  "$HF_JOB_IMAGE" bash -c 'set -euo pipefail
    if ! command -v git >/dev/null 2>&1; then apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git; fi
    git clone --depth 1 --branch "$MIF_GIT_REF" "$MIF_GIT_URL" /workspace/make_it_flash
    cd /workspace/make_it_flash
    [[ "$(git rev-parse HEAD)" == "$MIF_GIT_COMMIT" ]] || { echo "source commit mismatch" >&2; exit 4; }
    python -m pip install -e .
    python - <<"PY"
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
repo=os.environ["MIF_CALIBRATION_REPO"]
revision=os.environ["MIF_CALIBRATION_REVISION"]
info=HfApi().dataset_info(repo, revision=revision)
if not info.private or info.sha != revision:
    raise SystemExit("calibration repo must be private and revision pinned")
root=Path("artifacts/staged-data")
root.mkdir(parents=True, exist_ok=True)
for name in ("calibration.jsonl", "data_manifest.json"):
    source=hf_hub_download(repo, filename=name, repo_type="dataset", revision=revision)
    (root/name).write_bytes(Path(source).read_bytes())
PY
    python - <<"PY"
import os
from make_it_flash.provenance import sha256_file
source="artifacts/staged-data/calibration.jsonl"
manifest="artifacts/staged-data/data_manifest.json"
if (sha256_file(source) != os.environ["MIF_CALIBRATION_DATA_SHA"] or sha256_file(manifest) != os.environ["MIF_CALIBRATION_MANIFEST_SHA"]):
    raise SystemExit("calibration data is not identical to the pinned run; no teacher loaded")
PY
    make-it-flash staged-layer --data-file artifacts/staged-data/calibration.jsonl --output-dir artifacts/staged-layer --layer "$MIF_LAYER" --epochs "$MIF_FIT_EPOCHS" --max-steps "$MIF_MAX_STEPS" --validation-fraction "$MIF_VALIDATION_FRACTION"
    python scripts/publish_staged_layer.py artifacts/staged-layer/fit --run-id "$MIF_RUN_ID" --layer "$MIF_LAYER" --expected-repo-sha "$MIF_EXPECTED_OUTPUT_REPO_SHA" --expected-source-commit "$MIF_GIT_COMMIT" --launch'

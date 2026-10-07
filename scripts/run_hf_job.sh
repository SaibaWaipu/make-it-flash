#!/usr/bin/env bash
set -euo pipefail

MIF_GIT_URL="${MIF_GIT_URL:-https://github.com/SaibaWaipu/make-it-flash.git}"
: "${MIF_OUTPUT_REPO:?Set MIF_OUTPUT_REPO to the private Hub model repo for the fitted layer}"

HF_FLAVOR="${HF_FLAVOR:-a100-large}"
HF_TIMEOUT="${HF_TIMEOUT:-4h}"
HF_JOB_IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime}"
MIF_TOKENS="${MIF_TOKENS:-100000}"
MIF_LAYER="${MIF_LAYER:-0}"
MIF_MODEL="${MIF_MODEL:-llm-jp/llm-jp-4.1-32b-a3b-thinking}"
MIF_MODEL_REVISION="${MIF_MODEL_REVISION:-cda260706786758045e5e96bf4d738bbc01155b5}"
MIF_GIT_REF="${MIF_GIT_REF:-main}"

hf jobs run --flavor "$HF_FLAVOR" --timeout "$HF_TIMEOUT" --secrets HF_TOKEN --env "MIF_GIT_URL=$MIF_GIT_URL" --env "MIF_GIT_REF=$MIF_GIT_REF" --env "MIF_OUTPUT_REPO=$MIF_OUTPUT_REPO" --env "MIF_TOKENS=$MIF_TOKENS" --env "MIF_LAYER=$MIF_LAYER" --env "MIF_MODEL=$MIF_MODEL" --env "MIF_MODEL_REVISION=$MIF_MODEL_REVISION" "$HF_JOB_IMAGE" bash -lc 'set -euo pipefail
  git clone --depth 1 --branch "$MIF_GIT_REF" "$MIF_GIT_URL" /workspace/make_it_flash
  cd /workspace/make_it_flash
  printf "Source commit: "
  git rev-parse HEAD
  python -m pip install -e .
  make-it-flash prepare --model "$MIF_MODEL" --model-revision "$MIF_MODEL_REVISION" --output-dir artifacts/data --max-tokens "$MIF_TOKENS"
  make-it-flash cache --data-file artifacts/data/calibration.jsonl --output-dir artifacts/cache --layers "$MIF_LAYER"
  make-it-flash fit --cache-dir artifacts/cache --output-dir artifacts/gdn --layer "$MIF_LAYER"
  python scripts/publish_hf_artifact.py artifacts/gdn'

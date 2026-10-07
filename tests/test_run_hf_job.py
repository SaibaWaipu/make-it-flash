import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_hf_job.sh"
FAKE_HF = """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2 $3" == "jobs hardware --json" ]]; then
  printf '%s\\n' '[{"name":"a100-large","cost/hour":"$2.50"}]'
elif [[ "$1 $2" == "jobs run" ]]; then
  arguments="$*"
  shift 2
  while (($#)); do
    if [[ "$1" == "--label" ]]; then
      shift
      if [[ ! "$1" =~ ^[a-zA-Z0-9_-]+=[a-zA-Z0-9_-]+$ ]]; then
        echo "invalid mock label: $1" >&2
        exit 98
      fi
    fi
    shift
  done
  printf 'MOCK_HF_JOBS_RUN %s\\n' "$arguments"
else
  echo "unexpected fake hf command: $*" >&2
  exit 99
fi
"""


def _run_runner(tmp_path, **overrides):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_hf = fake_bin / "hf"
    fake_hf.write_text(FAKE_HF, encoding="utf-8")
    fake_hf.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env.update(overrides)
    return subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, check=False)


def test_hf_job_runner_defaults_to_rate_estimated_dry_run(tmp_path):
    result = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "rate=$2.50/hour" in result.stdout
    assert "estimated_max_cost=$7.50" in result.stdout
    assert "DRY RUN ONLY" in result.stdout
    assert "MOCK_HF_JOBS_RUN" not in result.stdout


def test_hf_job_runner_refuses_over_budget_before_submission(tmp_path):
    result = _run_runner(
        tmp_path,
        MIF_LAUNCH_HF_JOB="1",
        MIF_APPROVED_BUDGET_USD="8.73",
        MIF_CONFIRMED_CUMULATIVE_SPENT_USD="0.27",
        MIF_OUTPUT_REPO="RemydreScarlet/private-pilot",
        MIF_GIT_COMMIT="deadbeef",
        HF_TIMEOUT="4h",
    )

    assert result.returncode == 3
    assert "estimated_max_cost=$10.00" in result.stdout
    assert "Cost guard refused" in result.stderr
    assert "MOCK_HF_JOBS_RUN" not in result.stdout


def test_hf_job_runner_launches_only_with_budget_and_required_pins(tmp_path):
    result = _run_runner(
        tmp_path,
        MIF_LAUNCH_HF_JOB="1",
        MIF_APPROVED_BUDGET_USD="8.73",
        MIF_CONFIRMED_CUMULATIVE_SPENT_USD="0.27",
        MIF_OUTPUT_REPO="RemydreScarlet/private-pilot",
        MIF_GIT_COMMIT="deadbeef",
    )

    assert result.returncode == 0, result.stderr
    assert "estimated_max_cost=$7.50" in result.stdout
    assert "MOCK_HF_JOBS_RUN" in result.stdout
    assert "--timeout 3h" in result.stdout
    assert "--detach" in result.stdout
    assert "bash -c" in result.stdout
    assert "apt-get update" in result.stdout


def test_hf_job_runner_requires_invoice_confirmed_prior_spend(tmp_path):
    result = _run_runner(
        tmp_path,
        MIF_LAUNCH_HF_JOB="1",
        MIF_APPROVED_BUDGET_USD="8.73",
        MIF_OUTPUT_REPO="RemydreScarlet/private-pilot",
        MIF_GIT_COMMIT="deadbeef",
    )

    assert result.returncode != 0
    assert "invoice-confirmed prior spend" in result.stderr
    assert "MOCK_HF_JOBS_RUN" not in result.stdout


def test_hf_job_runner_enforces_remaining_cumulative_cap(tmp_path):
    result = _run_runner(
        tmp_path,
        MIF_LAUNCH_HF_JOB="1",
        MIF_APPROVED_BUDGET_USD="7.50",
        MIF_CONFIRMED_CUMULATIVE_SPENT_USD="2.00",
        MIF_OUTPUT_REPO="RemydreScarlet/private-pilot",
        MIF_GIT_COMMIT="deadbeef",
    )

    assert result.returncode == 3
    assert "remaining cumulative cap $7.00" in result.stderr
    assert "MOCK_HF_JOBS_RUN" not in result.stdout

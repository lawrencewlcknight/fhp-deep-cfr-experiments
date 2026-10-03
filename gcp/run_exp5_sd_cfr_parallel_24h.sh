#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ACTION="${1:-run}"
export EVAL_LBR="${EVAL_LBR:-0}"
case "$EVAL_LBR" in 0|1) ;; *) echo "EVAL_LBR must be 0 or 1" >&2; exit 64 ;; esac
LBR_FLAG=""
if [[ "$EVAL_LBR" == "0" ]]; then LBR_FLAG="--skip-lbr"; fi
if [[ "$ACTION" == "smoke-local" ]]; then
  OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-sdcfr-exp5-smoke.XXXXXX)}"
  RUN_RAY_SD_CFR_TESTS=1 python3 -m pytest -q tests/test_exp5_sd_cfr_parallel_24h.py tests/test_sd_cfr_parallel.py -k real_ray
  python3 -m experiments.fhp.exp5_sd_cfr_parallel_24h.train --seed 0 --smoke --output-root "$OUTPUT"
  python3 -m experiments.fhp.exp5_sd_cfr_parallel_24h.evaluate smoke --source "$OUTPUT" --output "$OUTPUT/evaluation" --workers 2 $LBR_FLAG
  echo "Smoke output: $OUTPUT"
  exit 0
fi
export RUN_ID="${RUN_ID:-sdcfr5-par8-$(date -u '+%Y%m%d-%H%M%S')}"
export REPO_REF="${REPO_REF:-$(git rev-parse HEAD)}"
if [[ "$ACTION" == "run" || "$ACTION" == "evaluate-only" || "$ACTION" == "profile-only" ]]; then
  for required in deep_cfr_poker/sd_cfr_lbr.py deep_cfr_poker/sd_cfr_lbr_audit.py tests/test_sd_cfr_no_lbr.py; do
    if ! git cat-file -e "$REPO_REF:$required" 2>/dev/null; then
      echo "$required is absent from REPO_REF=$REPO_REF. Commit and push Experiment 5 with optional LBR evaluation, then refresh REPO_REF." >&2
      exit 64
    fi
  done
  for required in experiments/fhp/exp5_sd_cfr_parallel_24h/config.py experiments/fhp/exp5_sd_cfr_parallel_24h/train.py gcp/exp5_sd_cfr_parallel_24h_batch.py deep_cfr_poker/sd_cfr_parallel.py tests/test_exp5_sd_cfr_parallel_24h.py; do
    if ! git cat-file -e "$REPO_REF:$required" 2>/dev/null; then
      echo "$required is absent from REPO_REF=$REPO_REF. Commit and push Experiment 5, then refresh REPO_REF." >&2
      exit 64
    fi
  done
fi
exec python3 gcp/exp5_sd_cfr_parallel_24h_batch.py "$ACTION"

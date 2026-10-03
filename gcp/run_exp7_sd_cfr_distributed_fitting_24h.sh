#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ACTION="${1:-run}"
export EXP5_RUN_ID="${EXP5_RUN_ID:-sdcfr5-par8-20261002-102757}"
export EVAL_LBR="${EVAL_LBR:-0}"
case "$EVAL_LBR" in
  0) LBR_FLAGS=(--skip-lbr) ;;
  1) LBR_FLAGS=() ;;
  *) echo "EVAL_LBR must be 0 or 1" >&2; exit 64 ;;
esac
if [[ "$ACTION" == "smoke-local" ]]; then
  OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-sdcfr-exp7-smoke.XXXXXX)}"
  RUN_RAY_SD_CFR_TESTS=1 python3 -m pytest -q tests/test_exp7_sd_cfr_distributed_fitting.py tests/test_sd_cfr_distributed.py tests/test_exp7_comparison.py
  python3 -m experiments.fhp.exp7_sd_cfr_distributed_fitting_24h.benchmark --output "$OUTPUT/fitting_benchmark" --allow-trajectory-drift
  python3 -m experiments.fhp.exp5_sd_cfr_parallel_24h.train --seed 0 --smoke --output-root "$OUTPUT/reference_training"
  python3 -m experiments.fhp.exp7_sd_cfr_distributed_fitting_24h.train --seed 0 --smoke --output-root "$OUTPUT/training"
  python3 -m experiments.fhp.exp7_sd_cfr_distributed_fitting_24h.evaluate smoke --source "$OUTPUT/training" --reference-source "$OUTPUT/reference_training" --output "$OUTPUT/evaluation" --workers 2 "${LBR_FLAGS[@]}"
  echo "Smoke output: $OUTPUT"
  exit 0
fi
export RUN_ID="${RUN_ID:-sdcfr7-distfit-$(date -u '+%Y%m%d-%H%M%S')}"
export REPO_REF="${REPO_REF:-$(git rev-parse HEAD)}"
if [[ "$ACTION" == "run" || "$ACTION" == "evaluate-only" || "$ACTION" == "profile-only" ]]; then
  for required in experiments/fhp/exp7_sd_cfr_distributed_fitting_24h/config.py experiments/fhp/exp7_sd_cfr_distributed_fitting_24h/train.py experiments/fhp/exp7_sd_cfr_distributed_fitting_24h/benchmark.py experiments/fhp/exp7_sd_cfr_distributed_fitting_24h/comparison.py deep_cfr_poker/sd_cfr_distributed.py gcp/exp7_sd_cfr_distributed_fitting_24h_batch.py tests/test_exp7_sd_cfr_distributed_fitting.py tests/test_exp7_comparison.py; do
    if ! git cat-file -e "$REPO_REF:$required" 2>/dev/null; then
      echo "$required is absent from REPO_REF=$REPO_REF. Commit and push Experiment 7, then refresh REPO_REF." >&2
      exit 64
    fi
  done
fi
exec python3 gcp/exp7_sd_cfr_distributed_fitting_24h_batch.py "$ACTION"

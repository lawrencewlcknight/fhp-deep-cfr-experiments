#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
MODE="${1:-run}"
if [[ "$MODE" != "run" && "$MODE" != "smoke" ]]; then
  echo "Usage: bash gcp/run_exp1_sd_cfr_efficiency.sh [run|smoke]" >&2
  exit 64
fi
export RUN_ID="${RUN_ID:-fhp-sdcfr-exp1-$(date -u '+%Y%m%d-%H%M%S')}"
if [[ ! "$RUN_ID" =~ ^[a-z][a-z0-9-]{0,49}$ ]]; then
  echo "RUN_ID must be a lowercase Batch-compatible identifier (at most 50 characters)" >&2
  exit 64
fi
export REPO_REF="${REPO_REF:-$(git rev-parse HEAD)}"
if ! git rev-parse --verify "${REPO_REF}^{commit}" >/dev/null 2>&1; then
  echo "REPO_REF is not a commit in this checkout: $REPO_REF" >&2
  exit 64
fi
for required in experiments/fhp/exp1_sd_cfr_efficiency/run.py deep_cfr_poker/sd_cfr_optimised.py; do
  if ! git cat-file -e "${REPO_REF}:${required}" 2>/dev/null; then
    echo "$required is absent from REPO_REF=$REPO_REF. Commit/push this experiment first." >&2
    exit 64
  fi
done
MODULE="experiments.fhp.exp1_sd_cfr_efficiency.run"
EXPERIMENT_COMMAND="python -m ${MODULE} --smoke --output-dir outputs/${RUN_ID}/smoke"
if [[ "$MODE" == "run" ]]; then
  EXPERIMENT_COMMAND+=" && python -m ${MODULE} --output-dir outputs/${RUN_ID}/benchmark"
fi
# One VM, all arms sequential: concurrent fits would contaminate speed timings.
bash gcp/submit_batch_experiment.sh "$RUN_ID" "$EXPERIMENT_COMMAND" \
  n2-standard-8 7200 8000 30000 50 pd-balanced

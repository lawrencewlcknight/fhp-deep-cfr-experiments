#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ACTION="${1:-run}"
if [[ "$ACTION" == "smoke-local" ]]; then
  OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-sdcfr-exp3-smoke.XXXXXX)}"
  python3 -m experiments.fhp.exp3_sd_cfr_structured_24h.train --seed 0 --smoke --output-root "$OUTPUT"
  python3 -m experiments.fhp.exp3_sd_cfr_structured_24h.evaluate smoke --source "$OUTPUT" --output "$OUTPUT/evaluation" --workers 2
  echo "Smoke output: $OUTPUT"
  exit 0
fi
export RUN_ID="${RUN_ID:-sdcfr3-24h-$(date -u '+%Y%m%d-%H%M%S')}"
export REPO_REF="${REPO_REF:-$(git rev-parse HEAD)}"
if [[ "$ACTION" == "run" || "$ACTION" == "evaluate-only" ]]; then
  for required in experiments/fhp/exp3_sd_cfr_structured_24h/train.py deep_cfr_poker/fhp_features.py deep_cfr_poker/sd_cfr_structured.py gcp/exp3_sd_cfr_structured_24h_batch.py; do
    if ! git cat-file -e "$REPO_REF:$required" 2>/dev/null; then
      echo "$required is absent from REPO_REF=$REPO_REF. Commit and push Experiment 3, then refresh REPO_REF." >&2
      exit 64
    fi
  done
fi
exec python3 gcp/exp3_sd_cfr_structured_24h_batch.py "$ACTION"

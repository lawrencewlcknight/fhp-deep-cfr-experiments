#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ACTION="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$ACTION" == "smoke-local" ]]; then
  OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-sdcfr7-short.XXXXXX)}"
  exec python3 -m experiments.fhp.exp7_sd_cfr_distributed_fitting_24h.short_test --smoke --output "$OUTPUT" "$@"
fi
export RUN_ID="${RUN_ID:-sdcfr7-short-$(date -u '+%Y%m%d-%H%M%S')}"
export REPO_REF="${REPO_REF:-$(git rev-parse HEAD)}"
exec python3 gcp/exp7_sd_cfr_short_test_batch.py "$ACTION" "$@"

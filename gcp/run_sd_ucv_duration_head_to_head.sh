#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ "$ACTION" == status ]]; then
  exec python3 "$SCRIPT_DIR/sd_ucv_duration_head_to_head_batch.py" "$@"
fi
: "${REPO_REF:?Set REPO_REF to the full pushed SD-CFR evaluation commit SHA}"
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]]; then
  echo "REPO_REF must be a full pushed commit SHA" >&2; exit 2
fi
for file in experiments/fhp/retrospective_sd_ucv_duration_evaluation/run.py gcp/sd_ucv_duration_head_to_head_batch.py; do
  if ! git -C "$REPO_DIR" cat-file -e "$REPO_REF:$file"; then
    echo "REPO_REF predates this evaluator. Use the new pushed commit." >&2; exit 2
  fi
done
TEMP_DIR="$(mktemp -d /tmp/sd-ucv-duration-launch.XXXXXX)"
trap 'rm -f "$TEMP_DIR/builder.py"; rmdir "$TEMP_DIR"' EXIT
git -C "$REPO_DIR" show "$REPO_REF:gcp/sd_ucv_duration_head_to_head_batch.py" > "$TEMP_DIR/builder.py"
python3 "$TEMP_DIR/builder.py" "${@:-run}"

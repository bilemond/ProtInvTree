#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_cath_sweep.sh /path/to/cath4.2 results/sweep [extra CLI options]
if (( $# < 2 )); then
  echo "Usage: $0 DATA_DIR OUTPUT_ROOT [extra evaluation options]" >&2
  exit 2
fi
data_dir="$1"
output_root="$2"
shift 2
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# CUDA_VISIBLE_DEVICES is set by the caller. No fixed server paths or GPU IDs.
for temperature in 0.1 0.3 0.5 0.7 1.0; do
  for width in 2 4 6; do
    python "$script_dir/evaluate_cath.py" \
      --data-dir "$data_dir" --output-dir "$output_root/mcts-T${temperature}-K${width}" \
      --method mcts --samples-per-target 20 --temperature "$temperature" \
      --expand-n "$width" --pre-expand-n "$width" "$@"
  done
  python "$script_dir/evaluate_cath.py" \
    --data-dir "$data_dir" --output-dir "$output_root/esm3-T${temperature}" \
    --method esm3 --samples-per-target 20 --temperature "$temperature" "$@"
done

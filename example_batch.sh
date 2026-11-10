#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_ROOT="${1:-$PROJECT_ROOT/input}"
OUTPUT_ROOT="${2:-$PROJECT_ROOT/output/denoised_$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS=("${@:3}")
mkdir -p "$OUTPUT_ROOT" || exit 1
LOG="$OUTPUT_ROOT/denoising.log"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -u "$PROJECT_ROOT/run.py" batch "$INPUT_ROOT" "$OUTPUT_ROOT" \
    --pattern '*.vhdr' --tr 1.660 --n-slices 60 --mb-factor 3 \
    --trigger R128 --ch-regex 'ZYG|COR' --drop-last-volume \
    --science-mode matched --timing-mode initial \
    --zero-fill-gap-fraction 0 --line-cleanup off \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
PIPE_STATUS=("${PIPESTATUS[@]}")
STATUS="${PIPE_STATUS[0]}"
if [ "$STATUS" -eq 0 ] && [ "${PIPE_STATUS[1]}" -ne 0 ]; then
    STATUS="${PIPE_STATUS[1]}"
fi
exit "$STATUS"

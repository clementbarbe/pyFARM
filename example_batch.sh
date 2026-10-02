#!/usr/bin/env bash
set -euo pipefail

pyfarm-denoise batch /path/to/raw /path/to/denoised \
  --pattern '*.vhdr' \
  --tr 1.6 \
  --n-slices 54 \
  --mb-factor 3 \
  --trigger R128 \
  --ch-regex 'EXT|FLE' \
  --drop-last-volume \
  --zero-fill-gap-fraction 1.0 \
  --skip-existing

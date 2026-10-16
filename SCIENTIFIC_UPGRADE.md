# pyFARM scientific upgrade (experimental)

This branch is based on the original pyFARM 1.1.0 repository. The original
layout, CLI and BrainVision sample data are preserved.

The default configuration retains the historical arithmetic-mean template,
existing PCA and timing-search behavior. New options are opt-in:

- `--template-method mean|trimmed|median|huber`
- `--template-huber-delta 1.5`
- `--pca-mode current|disabled|conservative`
- `--timing-dtime-max-ms 250` with optional `--timing-search-step-samples`
- `--export-gap-mask` and `--gap-mask-guard-samples 16`

Example:

```bash
python -m pip install -e '.[test]'
python -m pytest farm/tests -q
pyfarm-denoise run data/me3mb3_tr1600_sl54.vhdr \
  --output-dir denoised_robust --tr 1.6 --n-slices 54 --mb-factor 3 \
  --trigger R128 --ch-regex 'EXT|FLE' \
  --template-method median --pca-mode conservative \
  --timing-dtime-max-ms 250 --export-gap-mask
```

For the documented TR=1.6 s, 54 slices, MB3 acquisition, there are 18 groups.
The original initial timing search covers only ~44.4 ms of dead time, whereas
a documented example of 81.9955 ms group duration would imply 124.081 ms dead
time. The expanded search is *not* an established physical estimate, and should
be cross-checked against sequence metadata.

Important: **these algorithm options are experimental**. Unit tests do not
establish recovery of real micro-contractions. Compare paired real acquisitions
with known EMG injections, monitor scanner-locked residual artifacts, and use
independent physiological ground-truth recordings before making scientific
claims. SPM regressors are intentionally out of scope.

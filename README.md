# pyFARM-denoise

Python pipeline for reducing MRI gradient artifacts in EMG recordings acquired
simultaneously with fMRI. It exports BrainVision signals and quantitative QC.
Envelope extraction, normalization, HRF convolution and SPM regressors belong to
a separate analysis step.

The method is inspired by [van der Meer et al. (2010), FARM](https://doi.org/10.1016/j.clinph.2009.12.035).
This implementation differs from the original algorithm; see [METHOD.md](METHOD.md).

## Install

```bash
python -m pip install -e .
```

## Process a run

```bash
pyfarm-denoise run input/participant-01/run-01.vhdr \
  --output-dir output/run-01 \
  --tr 1.660 --n-slices 60 --mb-factor 3 \
  --trigger R128 --ch-regex 'ZYG|COR' \
  --drop-last-volume --science-mode matched --timing-mode initial
```

TR is in seconds. `n-slices` is the anatomical slice count; 60 slices with MB3
means 20 acquisition groups per volume. Change these values for other sequences.
Use `--drop-last-volume` when the final trigger precedes an incomplete/manual-stop
volume. The `.vhdr`, `.eeg` and `.vmrk` files must be present together.

## Process a dataset

```bash
pyfarm-denoise batch input output/denoised \
  --pattern '*.vhdr' \
  --tr 1.660 --n-slices 60 --mb-factor 3 \
  --trigger R128 --ch-regex 'ZYG|COR' \
  --drop-last-volume --science-mode matched --timing-mode initial \
  --zero-fill-gap-fraction 0 --line-cleanup off
```

Subject subdirectories are retained. Each run receives its own figure directory.
`example_batch.sh` provides a logged batch command. For a deliberate restart,
`--skip-existing` skips any existing output header; it does **not** verify that
the old output used the same configuration. Prefer a new output directory when
comparing versions or settings.

## Output and defaults

- `*_FARM.vhdr/.eeg/.vmrk`: full-length recording, with selected channels corrected
  inside the retained scan crop. Sample positions and timing remain native.
- `*_FARM_denoise.json`: configuration, volume onsets, timing estimates, ADC-limit
  observations, correction diagnostics and quantitative QC.
- `figures/<run>/`: time plots, spectra, spectrograms and trigger-aligned views.
  `--no-figures` disables their generation.

The default `matched` science branch uses the **same 30-Hz high-pass** as the
artifact templates. No final 250-Hz low-pass, envelope or amplitude normalization
is imposed on the exported signal. The 30–250 Hz QC band is applied identically
to before/after diagnostic signals only. `--science-mode broadband` retains the
legacy unmatched subtraction for comparisons and can leave large residuals.

Timing defaults to the robust initial estimate. `local` performs a bounded
refinement and `fixed` requires `--fixed-sdur-ms` established independently for
the acquisition protocol. Unexpected trigger intervals stop processing.

Fractional shifts affect padded artifact estimates; corrections are returned to
the original grid. Templates use neighboring occurrences of the same acquisition
group, trimmed averages and repeatability gates. PCA removes only the part
supported by the estimated artifact subspace. `--pca-max-components 0` disables
this additional PCA step. Volume-gap templates are enabled; destructive zero
filling is disabled. Nonzero `--zero-fill-gap-fraction` explicitly destroys
samples and records their exact intervals.

## Optional residual-line processing

Residual peaks are not automatically removed. The default is `--line-cleanup off`.

```bash
# Report persistent narrow lines without changing samples:
pyfarm-denoise run input/participant-01/run-01.vhdr \
  --output-dir output/detection --tr 1.660 --n-slices 60 --mb-factor 3 \
  --ch-regex 'ZYG|COR' --line-cleanup detect

# Explicit experimental local sinusoidal regression:
pyfarm-denoise run input/participant-01/run-01.vhdr \
  --output-dir output/regression --tr 1.660 --n-slices 60 --mb-factor 3 \
  --ch-regex 'ZYG|COR' --line-cleanup regress --line-max-count 32
```

`notch` is also available for explicit comparisons. Detection uses neighboring
spectral amplitudes, persistence, peak width and evidence in the original signal.
For regression, supported narrow lines are fitted with sine/cosine functions in
overlapping 32-s Hann windows. Amplitude and phase can vary between windows.
`--line-regression-window-seconds` changes this duration (minimum 16 s).
A common-channel harmonic family with volume-spaced sidebands may support lines
outside the exact scanner/mains grid. Frequencies are detected from data; no
cohort-specific frequencies are hard-coded.

This stage acts on the waveform, not just the figure. It can also attenuate true
EMG near a selected frequency, especially narrow coherent activity or cross-talk
between channels. Common-channel coherence is evidence, **not proof of artifact
origin**. A more attractive FFT does not establish better physiological recovery.
Use explicit comparisons and independent injections before adopting this stage.

For paired outputs, generate FFT overlays on identical axes and a CSV report:

```bash
python scripts/compare_line_outputs.py output/line_comparison \
  --output-dir output/comparison --validation-dir output/injection_controls
```

The optional validation directory contains JSON files from `validate_preservation.py`.

An existing FARM output can be evaluated without rerunning the core:

```bash
python scripts/evaluate_lines.py output/denoised/participant-01/run-01_FARM.vhdr \
  --output-dir output/line_comparison --mode regress --max-lines 32
```

It writes a distinct BrainVision output, selection report, PSD and FFT figures.
The default evaluator mode is detection only. The source sidecar and original
recording are required. Do not apply the stage repeatedly to an already filtered
output.

## QC and validation

```bash
python scripts/qc_before_after.py output/denoised --output-dir output/qc
```

The report compares identical native sample intervals and analysis bands. It
includes duration, min/max, RMS, scanner-locked residuals, a clearly labeled
asynchronous/scanner **SNR proxy**, and ADC-limit observations. This proxy is not
a physiological SNR. No automated metric certifies signal preservation.

INT16 source limits are checked using calibrated amplitudes, including symmetric
±32767 limits. Other formats are marked unassessed. Clipped input samples cannot
be reconstructed by denoising; a single rail hit does not justify rejecting an
entire run without temporal inspection.

Independent physiological injection test (verified fixed protocol timing):

```bash
python scripts/validate_preservation.py input/participant-01/run-01.vhdr \
  --tr 1.660 --n-slices 60 --mb-factor 3 --sdur-ms 82 \
  --ch-regex 'ZYG|COR' --science-mode matched \
  --line-cleanup regress --line-max-count 32 --output output/injection.json
```

The test compares `D(raw + independent EMG) - D(raw)` with the injected signal,
with adaptive and fixed line plans reported separately. These tests do not
validate all native activity, scanner-synchronous physiology, or frequencies
below the science high-pass.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

`input/`, `output/`, `denoised/` and generated figures are ignored by Git. Keep
participant data and local reports outside the published source tree/index.

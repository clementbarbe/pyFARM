# pyFARM-denoise

`pyFARM-denoise` is a **denoising-only** EMG-fMRI pipeline.  It removes MRI
acquisition artifacts and exports a cleaned BrainVision recording.  It does
**not** compute envelopes, normalize EMG amplitudes, convolve with an HRF, or
create SPM regressors.

The scientific core retains the validated FARM-like workflow used on the
supplied dataset: trigger-anchored slice-group timing, fractional alignment,
adaptive templates, true inter-volume gap masking when explicitly enabled,
and artifact-gated global PCA residual cleanup.

## Install

```bash
python -m pip install -e .
```

## One run

```bash
pyfarm-denoise run C4X001/C4X001_R1.vhdr \
  --output-dir denoised/C4X001 \
  --tr 1.6 --n-slices 54 --mb-factor 3 \
  --trigger R128 --ch-regex 'EXT|FLE' \
  --drop-last-volume \
  --zero-fill-gap-fraction 1.0
```

The exported `*_FARM.vhdr/.eeg/.vmrk` signal is the **broadband FARM-cleaned
waveform**.  The 30–250 Hz band is used only for QC figures/metrics and is not
imposed on the exported data.

A `*_FARM_denoise.json` sidecar records the retained volume onsets, scan crop,
timing estimates, channels, and QC metrics.  The separate `emg-regressors`
tool uses this sidecar when available.

## Whole dataset

For a tree such as `raw/C4X001/C4X001_R1.vhdr`, etc.:

```bash
pyfarm-denoise batch raw denoised \
  --pattern '*.vhdr' \
  --tr 1.6 --n-slices 54 --mb-factor 3 \
  --trigger R128 --ch-regex 'EXT|FLE' \
  --drop-last-volume \
  --zero-fill-gap-fraction 1.0 \
  --skip-existing
```

Output is kept flat within each subject directory:

```text
denoised/
└── C4X001/
    ├── C4X001_R1_FARM.vhdr
    ├── C4X001_R1_FARM.eeg
    ├── C4X001_R1_FARM.vmrk
    ├── C4X001_R1_FARM_denoise.json
    ├── C4X001_R2_FARM.vhdr
    └── ...
```

Figures are written under `figures/<run_basename>/` so batch runs cannot
overwrite one another.

## Timing stability (v1.1)

`sdur` is a property of the EPI acquisition sequence. It must not jump by
tens of milliseconds merely because the subject or EMG run changed. Earlier
versions allowed a broad numerical timing search; because the FARM alignment
objective is periodic, that search could converge to harmonic/sub-harmonic
minima while still reporting `success=True`.

The default is now:

```text
--timing-mode initial
```

Timing is estimated independently on every selected EMG channel, robustly
combined across channels, and `dtime` is derived from the measured TR. No
broad global optimisation is performed.

Three modes are available:

- `initial` (recommended/default): robust run estimate; safest for fixed EPI protocols.
- `local`: optional high-frequency refinement restricted to +/-0.5 ms by default.
- `fixed`: use exactly the same protocol timing for every run. Example:

```bash
pyfarm-denoise batch raw denoised ... \
  --timing-mode fixed --fixed-sdur-ms 81.9955
```

Use `fixed` only after establishing the protocol timing from representative
runs or sequence documentation. The sidecar records both the initial and final
timing so cohort-level QC is straightforward.

## Important protocol-specific setting

`zero_fill_gap_fraction=0` is the generic safety default because masking a gap
can destroy real EMG.  On the supplied 1.6-s TR / 54-slice / MB3 acquisition,
validation showed a strong scanner-locked boundary artifact in the true
dead-time, and `1.0` produced the cleaner spectrum.  Keep this choice explicit
in analysis scripts.

## Output philosophy

Denoising and physiological modelling are deliberately independent.  This
repository produces the cleaned measurement.  Cross-run amplitude calibration,
EMG envelopes, HRF convolution, and SPM matrices are handled downstream by the
separate `emg-regressors` repository.

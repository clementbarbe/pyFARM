# pyFARM-denoise

`pyFARM-denoise` is a **denoising-only** EMG-fMRI pipeline.  It removes MRI
acquisition artifacts and exports a cleaned BrainVision recording.  It does
**not** compute envelopes, normalize EMG amplitudes, convolve with an HRF, or
create SPM regressors.

The FARM-inspired core uses trigger-anchored timing, fractional alignment of
artifact estimates, robust same-group templates, dedicated volume-gap
templates, and artifact-supported PCA. Corrections are returned to the original
time grid and subtracted from the native measurement. The scientific changes
in v1.2 require validation for each acquisition protocol.

Original article :
  Van der Meer, J. N., Tijssen, M. A. J., Bour, L. J., van Rootselaar, A. F., & Nederveen, A. J. (2010).
  **Robust EMG–fMRI artifact reduction for motion (FARM)**.
  Clinical Neurophysiology, 121(5), 766–776.
  https://doi.org/10.1016/j.clinph.2009.12.035

GitHub page of the first author :

  https://github.com/jnvandermeer

GitHub page of the MATLAB version :

  https://github.com/benoitberanger/FARM

## Install

```bash
python -m pip install -e .
```

## One run

```bash
pyfarm-denoise data/me3mb3_tr1600_sl54.vhdr \
  --output-dir denoised/C4X001 \
  --tr 1.6 --n-slices 54 --mb-factor 3 \
  --trigger R128 --ch-regex 'EXT|FLE' \
  --drop-last-volume
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
  --timing-mode fixed --fixed-sdur-ms <verified_protocol_sdur_ms>
```

Use `fixed` only after establishing the protocol timing from representative
runs or sequence documentation. The sidecar records both the initial and final
timing so cohort-level QC is straightforward.

## Signal preservation (v1.2)

Fractional shifts are applied to padded artifact estimates and inverted before
subtraction. Only the estimated correction is downsampled; the original native
science samples are retained. A zero correction therefore preserves them exactly.

Templates require both target similarity and agreement between two disjoint
averages of selected candidates (default correlation 0.8). A 10% trimmed mean and
scale bounds [0, 2] limit transient contamination and extreme scaling. Templates
for the inter-volume gap are fitted separately; `--no-gap-correction` disables
this step. These safeguards cannot distinguish true EMG that exactly repeats
with the scanner from an artifact of the same shape.

PCA defaults to at most four components with artifact correlation >=0.6. Only
the artifact-supported projection is subtracted, with correction strength
increasing continuously with artifact similarity. `--pca-max-components 0`
disables all residual PCA cleanup, including the section mean.

`--zero-fill-gap-fraction` defaults to **0**. Nonzero values destroy measured
samples and are applied only after downsampling. The sidecar records their
exact half-open sample intervals and the fraction of the scan removed. Prefer
gap-template correction when preservation matters; a cleaner spectrum alone
does not justify masking physiological activity.

The coarse timing search covers gaps up to 200 ms by default
(`--timing-max-gap-ms`). Set this bound from sequence information. Local
refinement stays within the configured neighbourhood. Missing/duplicate triggers
or unexpected intervals stop processing; `--trigger-tolerance-ms` defaults to
5 ms. The timing-only high-pass is configurable via `--timing-reference-hpf`
(default 250 Hz); verify scanner dominance after acquisition filtering.

QC includes global RMS, trigger-locked RMS and excess spectral power near
slice-group harmonics. None proves physiological preservation. Use independent
EMG injections, inspect bursts and compare with PCA disabled. The sidecar also
records applied template counts, correction RMS and timing dispersion.

Broadband export means no final analysis-band filter is imposed. It does not
restore frequencies removed by acquisition hardware, and low-frequency motion
artifacts outside the template-estimation band may remain.

See [TESTING.md](TESTING.md) for injection results, the PCA ablation and the
reproducible validation command.

## Output philosophy

Denoising and physiological modelling are deliberately independent.  This
repository produces the cleaned measurement.  Cross-run amplitude calibration,
EMG envelopes, HRF convolution, and SPM matrices are handled downstream by the
separate `emg-regressors` repository.

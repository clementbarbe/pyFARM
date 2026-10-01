# PYTHON FARM EMG-fMRI Denoising Pipeline

Python implementation of the **FARM** (fMRI Artifact Reduction for EMG) pipeline
for removing gradient artifacts from EMG signals recorded simultaneously with fMRI.

- Van der Meer et al. (2010), *Robust EMG–fMRI artifact reduction for motion (FARM)*,
  Clinical Neurophysiology 121(5), 766–776.
- MATLAB FARM implementation: https://github.com/benoitberanger/FARM

## Installation

```bash
pip install -e .
# Optional BrainVision export (.vhdr/.eeg/.vmrk):
pip install -e '.[brainvision-export]'
```

## Quick start

```bash
python run.py path/to/data.vhdr --tr 1.6 --n-slices 54 --mb-factor 3
```

Or programmatically:

```python
from farm.config import FARMConfig
from farm.workflow import run_pipeline

cfg = FARMConfig(
    vhdr_path="data/recording.vhdr",
    tr=1.6,
    n_slices=54,
    mb_factor=3,
)
results = run_pipeline(cfg)
```

## Signal-quality design

The current implementation deliberately separates two signals:

1. **Science branch** — the cropped original EMG. This is the signal ultimately
   preserved and cleaned.
2. **Artifact branch** — a high-pass-filtered copy (30 Hz by default) used to
   select and construct gradient-artifact templates. A separate 100 Hz branch
   is used for timing/reference estimation.

This prevents an aggressive preprocessing HPF from becoming an irreversible
part of the cleaned output.

The correction pipeline is:

1. Load BrainVision data and volume triggers.
2. Crop to complete fMRI volumes.
3. Build science, artifact-estimation and timing branches.
4. Select the reference channel by **trigger-locked repeatability**, not by a
   single maximum-amplitude sample.
5. Estimate and optimize slice-group timing.
6. Upsample (×10 by default; retained for maximum timing fidelity).
7. Apply fractional-sample alignment.
8. Build same-slice-group adaptive templates from nearby volumes using robust
   trimmed averaging and correlation rejection.
9. Preserve inter-volume gaps by default (`zero_fill_gap_fraction=0`).
10. Apply conservative, artifact-specific PCA residual cleanup.
11. Downsample back to the native sampling rate without imposing an extra
    250 Hz low-pass on the core FARM result.
12. Derive a separate 30–250 Hz EMG analysis branch exactly once, then export
    broadband + EMG-band arrays and build regressors.

### Important changes versus the previous version

- The final slice-group now uses its **full segment** for template correlation;
  `dtime` is not subtracted from the segment window.
- Inter-volume dead-time is **not zero-filled by default**.
- PCA is performed per slice-group by default and removes a component only if
  it both explains sufficient variance and resembles the scanner template (or
  its first/second temporal derivatives).
- The repeated PCA mean is also gated by scanner similarity and repeatability.
- PCA projection is matrix-vectorized (`U @ (U.T @ X)`) instead of one
  least-squares solve per slice.
- Template candidates use a sample-wise trimmed mean to reduce transient EMG
  leakage into the artifact estimate.
- Final QC reports both ordinary 30–250 Hz RMS reduction and **trigger-locked
  RMS reduction**, which is a more specific measure of scanner-artifact removal.
- The core FARM output is broadband; the 30–250 Hz signal is an explicit
  derived branch (`clean_emg_band`) filtered once. FFT/PSD overlays use the
  broadband signals and a Hann window, avoiding filter-shaped spectral domes.

## Quality-oriented controls

Useful CLI options:

```text
--artifact-hpf 30
--timing-hpf 100
--template-min-corr 0.15
--template-trim 0.15
--pca-artifact-corr 0.10
--pca-max-components 6
--zero-fill-gap-fraction 0.0
```

For a new acquisition protocol, compare the default output with a more
conservative PCA setting (for example `--pca-artifact-corr 0.2`) and inspect
both the residual scanner-locked waveform and known EMG bursts. A global RMS
reduction alone is not proof that physiological EMG has been preserved.

## Output

Outputs are written under `<output-dir>/<recording>_FARM/`:

- `.npz` / `.mat` — cleaned arrays and metadata, including:
  - `clean_broadband_crop`: FARM-cleaned scan interval without an imposed
    250 Hz post-filter;
  - `clean_emg_band`: one-pass 30–250 Hz analysis branch;
  - `clean_full`: full-length broadband recording with the FARM-cleaned scan
    interval inserted.
- `.vhdr/.eeg/.vmrk` — full-length BrainVision output when optional `pybv` is
  installed. If it is absent, the pipeline continues and writes NPZ/MAT.
- `regressors/` — EMG regressors at TR resolution.
- diagnostic figures, unless disabled.

## Tests

```bash
pytest -q
```


## Quality fix v2

For the supplied `me3mb3_tr1600_sl54` protocol, use the bundled `example.py`.
It restores the stable timing-reference behaviour, uses global artifact-gated
PCA, applies the final 30–250 Hz EMG band only once, and explicitly masks the
scanner-contaminated inter-volume gap. See `QUALITY_FIX_V2.md` for the A/B
diagnosis and validation details.

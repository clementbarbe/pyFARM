# PYTHON FARM EMG-fMRI Denoising Pipeline

Python implementation of the **FARM** (fMRI Artifact Reduction for EMG) pipeline
for removing gradient artifacts from EMG signals recorded simultaneously with fMRI.

- Original article :  
  Van der Meer, J. N., Tijssen, M. A. J., Bour, L. J., van Rootselaar, A. F., & Nederveen, A. J. (2010).  
  **Robust EMG–fMRI artifact reduction for motion (FARM)**.  
  Clinical Neurophysiology, 121(5), 766–776.  
  https://doi.org/10.1016/j.clinph.2009.12.035
  
- GitHub page of the first author :  
  https://github.com/jnvandermeer
  
- GitHub page of the MATLAB version :  
  https://github.com/benoitberanger/FARM

## Installation
```batch
pip install numpy scipy mne matplotlib
```
## Quick Start

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
## Pipeline Steps

    Load BrainVision data
    Detect volume triggers
    Select EMG channels
    Trim to scan boundaries
    High-pass filter (30 Hz)
    Estimate initial slice timing (sdur, dtime)
    Upsample ×10
    Optimize timing (Nelder-Mead)
    Compute slice markers
    Per-channel correction: alignment → templates → zero-fill → PCA
    Downsample + low-pass (250 Hz)
    Export (BrainVision, NPZ, MAT) + EMG regressors

## Output

All outputs are written to output/ (configurable):

    .vhdr/.eeg/.vmrk — Full-length BrainVision with all original markers preserved
    .npz / .mat — Cleaned arrays + metadata
    regressors/ — EMG regressors at TR resolution (HRF-convolved, derivatives, log)

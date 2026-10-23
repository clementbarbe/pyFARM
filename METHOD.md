# Denoising method and scope

This repository performs **artifact removal only**.  It intentionally stops
before any physiological feature extraction or statistical modelling.

Core sequence:

1. read BrainVision data and volume triggers;
2. select EMG channels;
3. crop an internal working copy to complete fMRI volumes;
4. keep a broadband science branch while deriving an HPF artifact branch;
5. estimate slice-group timing and refine `sdur`/`dtime`;
6. upsample working branches and align padded artifact segments only;
7. build adaptive same-slice-group templates from nearby volumes;
8. undo artifact-template alignment and subtract on the original time grid;
9. estimate separate templates for the inter-volume gap;
10. remove only the artifact-supported part of eligible residual PCA components;
11. downsample the correction and subtract it from the original native samples;
    apply optional explicit gap masking after downsampling;
12. replace the scan interval in the original full-length recording and export
    BrainVision.

The output waveform is not normalized and is not forced into a 30–250 Hz
analysis band.  The configured QC band is used only to quantify scanner
artifact reduction and to make comparable before/after plots.

Inter-volume gaps can contain both scanner artifacts and useful EMG. Separate
volume-gap templates are enabled by default. Zero filling is disabled by default;
if explicitly selected, every removed interval is recorded in the sidecar.
An improvement in the spectrum alone does not validate physiological preservation.

## Downstream boundary

The companion `emg-regressors` package owns all choices that are not denoising:
analysis band, envelope, baseline/noise estimation, cross-run scaling, HRF, and
SPM export.  Keeping this boundary prevents a change in statistical modelling
from silently changing the denoised measurement.


## Timing stability

Sequence timing is treated as a scanner/protocol parameter. The coarse timing
estimator is applied per channel and per volume; medians are used to reduce
contamination by voluntary EMG or motion. The default pipeline uses the
channel-consensus estimate directly. Optional refinement is performed only on
a scanner-dominated high-frequency reference and is restricted to a small
local neighbourhood. A distant minimum of the periodic alignment cost is never
accepted as a new physical sequence timing.


## Preservation safeguards (v1.2)

Template selection excludes the target occurrence, uses the same acquisition
group in neighbouring volumes, and requires correlation >=0.8 by default. Two
interleaved averages of selected candidates must also agree. Templates include
context for inverse fractional alignment; their scaling is bounded to [0, 2]
and their default average trims 10% of candidate values at each time sample.
Rounded interval endpoints avoid sample holes and overlapping ownership.

PCA is fitted per temporal section. Components must exceed the variance threshold
and have strong artifact similarity (default 0.6). Only their projection onto the
artifact-template subspace is subtracted, rather than an entire mixed component.
Correction strength increases continuously above the similarity threshold,
avoiding an abrupt full subtraction when a component barely passes.
Mean removal requires stronger similarity (0.8) and repeatability. A component
limit of zero disables this stage completely. These are empirical safeguards,
not a mathematical guarantee that all physiological content is preserved.

Timing uses a bounded coarse gap search (default 0–200 ms), normalized costs,
and robust volume/channel medians. It does not constrain gaps to half a group.
The high-pass used for timing is configurable, independent of the science branch.
The optional local optimization remains restricted to the robust estimate's
neighbourhood. Irregular triggers are rejected before correction.

The output is built as `original_native - downsample(estimated_correction)`.
Thus absence of correction preserves the original native samples, including
content near the native Nyquist frequency. No final 30–250 Hz band is imposed.
The template-estimation high-pass still limits which artifacts are modeled.

The implementation is inspired by van der Meer et al. (2010),
DOI: 10.1016/j.clinph.2009.12.035. It deliberately differs from the paper's
replacement of volume-boundary samples and final analysis-band filtering.
Independent EMG injection tests assess waveform/amplitude recovery alongside
artifact rejection; scanner-locked or RMS reduction alone is not sufficient.


BrainVision export uses pybv with explicitly rounded integer annotation samples
and durations. This avoids one-sample shifts caused by floating-point truncation
in a generic annotation-to-marker conversion. Channel data remain in their
calibrated units, and original full-recording marker positions are retained.

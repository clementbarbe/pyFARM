# Denoising method and scope

This repository performs **artifact removal only**.  It intentionally stops
before any physiological feature extraction or statistical modelling.

Core sequence:

1. read BrainVision data and volume triggers;
2. select EMG channels;
3. crop an internal working copy to complete fMRI volumes;
4. keep a broadband science branch while deriving an HPF artifact branch;
5. estimate slice-group timing and refine `sdur`/`dtime`;
6. upsample for fractional-sample alignment;
7. build adaptive same-slice-group templates from nearby volumes;
8. subtract scaled templates from the science branch;
9. optionally mask the measured inter-volume dead-time;
10. remove residual scanner-like components with global section-wise PCA;
11. downsample back to the native sampling rate;
12. replace the scan interval in the original full-length recording and export
    BrainVision.

The output waveform is not normalized and is not forced into a 30–250 Hz
analysis band.  The configured QC band is used only to quantify scanner
artifact reduction and to make comparable before/after plots.

For the supplied 1.6-s TR / 54-slice / MB3 protocol, the inter-volume gap was
empirically scanner-contaminated and `zero_fill_gap_fraction=1.0` produced a
cleaner spectrum.  That setting remains explicit rather than universal because
other protocols may contain useful EMG in the gap.

## Downstream boundary

The companion `emg-regressors` package owns all choices that are not denoising:
analysis band, envelope, baseline/noise estimation, cross-run scaling, HRF, and
SPM export.  Keeping this boundary prevents a change in statistical modelling
from silently changing the denoised measurement.

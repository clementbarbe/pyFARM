# Denoising method

The FARM-inspired core follows these stages:

1. Load calibrated BrainVision channels and exact volume markers; retain complete
   volumes and verify trigger intervals.
2. Apply the same zero-phase FIR high-pass to science and artifact branches in
   default `matched` mode.
3. Estimate acquisition-group duration on each channel, combine reliable timing
   references, and derive the inter-volume gap from the observed TR.
4. Upsample working branches. Align padded segments for artifact estimation;
   keep the science measurement on its original time grid.
5. Select correlated neighboring occurrences of the same acquisition group.
   Require agreement of disjoint candidate averages, use a trimmed mean and
   bounded scaling, then inverse-shift artifact estimates before subtraction.
6. Fit separate templates for the inter-volume gap.
7. Perform section-wise residual PCA with artifact similarity gates. Subtract
   only artifact-supported projections of eligible components and section means.
8. Downsample the correction and subtract it from the native filtered science
   branch. Optional line processing precedes explicit destructive gap masks.
9. Replace the retained interval in selected channels of the full recording,
   export BrainVision with rounded integer marker samples, and write QC/metadata.

The implementation is inspired by [van der Meer et al., 2010](https://doi.org/10.1016/j.clinph.2009.12.035).
It differs from the original boundary-sample replacement and final analysis-band
filtering. Matching science/template filtering prevents the systematic
`(I-H) × artifact` residual of an unmatched `raw - H(artifact)` subtraction.
The default high-pass attenuates physiological frequencies below its cutoff;
no claim is made that the original broadband waveform is preserved.

## Residual lines: explicit experiment

The additional stage is disabled by default. Up to 64 independent Hann-windowed
periodograms estimate a median spectrum. Local flanks provide a spectral floor.
Default candidates must exceed it by 8 dB, have FWHM ≤0.5 Hz, appear in at least
75% of sampled windows, and be supported in the original recording. Scanner/mains
compatibility supports the existing notch mode; unrelated peaks are reported.

The optional regression mode refines frequencies on the original-record Hann FFT.
A persistent common-channel carrier can additionally be supported by harmonic
structure in both original channels, or a detected harmonic family with
volume-spaced sidebands. Original cross-channel magnitude-squared coherence
must reach 0.8 at the carrier or a directly related harmonic independently
present in both original channels. Coherence is evaluated at the exact refined
frequencies using Hann-weighted projections, rather than interpolated from
neighboring FFT bins. These heuristic criteria can also be
met by physiological coherence/cross-talk and are not a significance test.

Correction confidence changes continuously near the prominence, persistence,
width, original-reference and common-channel coherence gates. This limits
discontinuities when small physiological changes move a candidate across a
selection threshold. Confidence is empirical, not a probability.

Each selected line is estimated by weighted sine/cosine regression in 32-s Hann
windows with 75% overlap. Fits overlap-add on original samples; partial edge
windows use measured data. Subtracting this estimate avoids imposing an entire
fixed IIR stopband, but still suppresses nearby physiological components within
the fit's effective bandwidth. Frequencies exactly matching a selected line
cannot be assigned to artifact versus physiology by this model.

Sliding sinusoidal regression is also used in established line-noise tools such
as [EEGLAB CleanLine](https://eeglab.org/plugins/cleanline/). The present implementation
uses Hann-weighted time-domain fits and its own evidence gates; it is **not** a
reimplementation of CleanLine's multitaper/Thomson F-test procedure.

Reports distinguish total selected-region power reduction from excess power over
a local floor. If residual excess is zero, its reduction is undefined, rather
than reported as an arbitrarily huge number of dB. All FFT comparisons use the
same time interval and QC preprocessing; diagnostic figures never replace or
rescale the exported waveform.

## Practical limits

- Acquisition clipping destroys information before correction and requires
  temporal/channel-level review. Its absence does not validate physiology.
- Template/PCA similarity and spectral coherence are empirical safeguards.
  Scanner-synchronous EMG remains intrinsically difficult to distinguish.
- Large RMS/line reductions alone do not prove useful EMG preservation.
- Independent injection tests assess a controlled family of signals, not all
  physiological activity. Adaptive and fixed-plan recovery distinguish detector
  switching from the effect of a selected filter/regression plan.
- There is no universal QC acceptance threshold or automatically validated channel.

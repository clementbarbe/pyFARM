# Timing stability fix (v1.1)

## Problem

For a fixed EPI sequence, `sdur` and `dtime` are protocol parameters and should
be essentially identical across runs. The previous implementation estimated a
stable initial timing and then allowed a broad scalar optimisation over roughly
±20% of `sdur`. The FARM alignment objective is periodic, so this search can
lock onto different harmonic/sub-harmonic minima in different runs.

`scipy` reporting `success=True` only means the numerical optimiser converged;
it does not validate the physical timing.

## Fix

1. Initial timing is estimated independently on every selected EMG channel.
2. Each channel uses robust per-volume medians rather than means.
3. The channel median defines the run timing estimate.
4. Default `timing_mode="initial"` uses that robust estimate directly.
5. Optional `timing_mode="local"` uses the scanner-dominated 250-Hz-HPF
   reference and is constrained to ±0.5 ms (configurable).
6. A local candidate is rejected if it hits the search boundary or fails to
   provide a minimum relative objective improvement.
7. `timing_mode="fixed"` lets a known protocol `sdur` be imposed on every run;
   `dtime` is always derived from the measured TR.

## Recommended cohort workflow

First run a small representative subset with the default `initial` mode and
inspect the `*_denoise.json` sidecars. If `sdur` is tightly clustered, use the
cohort median as a fixed protocol value for the production run:

```bash
pyfarm-denoise batch RAW DENOISED \
  ...your sequence arguments... \
  --timing-mode fixed \
  --fixed-sdur-ms <cohort_median_sdur_ms>
```

This ensures the same scanner sequence is denoised with the same acquisition
timing for every subject/run.


## v1.2 preservation update

The coarse search no longer limits the inter-volume gap to half a group. Its
upper bound is explicit (`timing_max_gap_seconds`, default 0.2 s). The objective
uses normalized mean deviation rather than a sum that favors shorter segments.
The configured timing reference high-pass defaults to 250 Hz and must be checked
against acquisition filtering. Per-channel volume MAD and the fraction of
estimates at the upper search bound are recorded in the sidecar. If every
channel repeatedly reaches that bound, non-fixed processing stops.

Local refinement remains bounded. The edge tolerance now accounts for duration
per group, so a low sampling rate no longer classifies the whole search interval
as an edge. Trigger labels are matched exactly (optionally including the
BrainVision prefix), and missing/duplicate triggers or unexpected intervals
stop correction instead of silently producing incorrect slice/gap assignments.

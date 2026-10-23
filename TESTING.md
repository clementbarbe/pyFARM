# Validation

## Historical split snapshot

- Unit tests: **8 passed**.
- The scientific FARM core is the validated quality-fix-v2 branch from the
  preceding development cycle; this split changes the output boundary, not the
  artifact-removal algorithm.
- A short real-data smoke test reached the broadband export stage and produced
  expected QC metrics. The current execution environment does not contain
  `pybv`, so BrainVision writing itself could not be exercised here. `pybv` is
  therefore a required package dependency in `pyproject.toml` rather than an
  optional dependency.


## v1.2 preservation changes — 2026-10-23

The complete pytest suite passes: **31 tests**. New cases cover native-sample
identity with no artifact correction, unrelated EMG, independent burst recovery
with slice and volume-gap artifacts, fractional reconstruction without sample
holes, PCA disablement, weak/mixed components, longer physical timing gaps,
exact trigger matching, missing triggers, explicit mask intervals, spectral QC,
strict JSON and BrainVision roundtrip export. Export tests check channel values
and event timing, including sample positions that floating-point truncation
previously shifted one sample earlier. No analysis-band filter is imposed on the science output.

```bash
python -m pytest -q
```

For read-only home directories, use `MNE_DONTWRITE_HOME=true`; plotting cache
can be directed to `/tmp` with `MPLCONFIGDIR`.

### Independent EMG injection into the supplied real recording

The first 24 volumes of `data/me3mb3_tr1600_sl54.vhdr` were processed at 5 kHz,
TR 1.6 s, 54 slices / MB3, with verified group duration 87.77777777777778 ms.
Both versions used interpolation factor 10 and no zero filling. The old baseline
is commit `7b6de3e8e4557f866048aaa34f1f78467cb970ab`; the new version uses
v1.2 defaults, including separate gap templates and guarded PCA.

The injection is independent 45–220 Hz band-limited noise with a burst envelope,
20 microvolt overall RMS, seeds 201–204 for the four channels. Recovery is
`D(raw + injection) - D(raw)`, where each input is processed independently, so
adaptation of template selection and PCA is included. Gain is the projection
onto the injected waveform, not a claim that every sample is preserved. Errors
are relative L2 norms; the envelope uses a 50-ms RMS window.

| Channel | Old gain | New gain | Old waveform error | New waveform error | Old envelope error | New envelope error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EXT_D | 0.753 | 0.952 | 0.556 | 0.292 | 0.239 | 0.119 |
| FLE_D | 0.800 | 0.944 | 0.525 | 0.233 | 0.246 | 0.055 |
| EXT_G | 0.904 | 0.972 | 0.439 | 0.304 | 0.129 | 0.087 |
| FLE_G | 0.862 | 0.973 | 0.934 | 0.343 | 0.665 | 0.101 |

Reproduce the current implementation:

```bash
python scripts/validate_preservation.py data/me3mb3_tr1600_sl54.vhdr \
  --tr 1.6 --n-slices 54 --mb-factor 3 \
  --sdur-ms 87.77777777777778 --n-volumes 24 \
  --output /tmp/pyfarm-preservation.json
```

Repeat with `--pca-max-components 0` for an ablation. PCA is not uniformly
beneficial: on this excerpt, EXT_D's envelope error is about 0.058 without PCA,
versus 0.119 with it; other channels change much less. Residual harmonic rejection
can improve with PCA, so inspect both artifact reduction and recovery rather
than maximizing RMS reduction alone. All of these measurements concern one
recording and one injected signal family; they are not cohort validation or
proof of full physiological recovery.


### Complete real-data run and export verification

The complete supplied recording was processed with `--drop-last-volume`, no
figures, no masking and otherwise v1.2 defaults: **486 complete volumes**, 8,748
slice-group intervals, four channels, 13 temporal PCA sections. Initial timing
agreed across channels: 87.7777778-ms groups and 20-ms inter-volume gaps.
The scanner-locked RMS reduction in the 30–250 Hz QC band was:

| Channel | Reduction factor |
| --- | ---: |
| EXT_D | 144.5 |
| FLE_D | 92.0 |
| EXT_G | 80.3 |
| FLE_G | 142.6 |

BrainVision export was re-read and checked: all 4,833,100 samples and four
channels retained, samples outside the processed scan unchanged within float32
precision, all original annotation sample positions preserved, including all
487 original volume triggers. The final trigger remains present even though
its volume was deliberately excluded from processing. No samples were masked.

This export verification revealed truncation in MNE's generic annotation
conversion. The exporter now supplies integer sample positions and rounded
durations directly to pybv, including correct handling of nonzero `first_samp`.

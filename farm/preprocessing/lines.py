"""Optional, evidence-gated removal of narrow spectral lines after FARM.

Detection uses a median of independent Hann periodograms and a local spectral
floor, not the height of individual bins in a whole-record FFT. A spectral peak
alone does not establish that a signal is non-physiological. Automatic notches
also require persistence, a narrow width, compatible scanner/mains timing and
support in the original recording. No frequencies are hard-coded to a dataset.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage, signal as sig

from farm.preprocessing.filters import apply_bandpass
from farm.preprocessing.harmonic_regression import refine_frequencies, regress_lines, coherence_at_frequencies


def _spectra(x, fs, window_seconds):
    """Sample at most 64 non-overlapping windows, bounding detection memory."""
    duration = len(x) / fs
    seconds = min(window_seconds, duration / 4)
    if seconds < 4.0:
        return None
    length = int(round(seconds * fs))
    count = len(x) // length
    blocks = np.unique(np.linspace(0, count - 1, min(count, 64)).astype(int))
    spectra = []
    for block in blocks:
        f, p = sig.periodogram(
            np.asarray(x[block*length:(block+1)*length], dtype=np.float64),
            fs=fs, window="hann", detrend="constant", scaling="density",
        )
        spectra.append(p)
    return f, np.asarray(spectra), length


def _floor(psd, df, max_width_hz):
    """Median of flanking bins, excluding the central line and its skirt."""
    guard = max(max_width_hz, 2*df)
    radius = max(3.0, 3*guard)
    offsets = np.arange(-int(np.ceil(radius/df)), int(np.ceil(radius/df))+1)
    footprint = (np.abs(offsets)*df >= guard) & (np.abs(offsets)*df <= radius)
    return np.maximum(ndimage.median_filter(psd, footprint=footprint, mode="nearest"),
                      np.finfo(np.float64).tiny)


def _association(frequency, df, sdur, tr):
    """A timing-compatible peak is a candidate, not proof of artifact origin."""
    if sdur is not None and tr is not None and sdur > 0 and tr > 0:
        group_harmonic = max(1, int(round(frequency * sdur)))
        group_distance = abs(frequency - group_harmonic/sdur)
        volume_distance = abs(frequency - round(frequency*tr)/tr)
        if (group_distance <= 0.5/tr + df and
                volume_distance <= max(0.02, df/2)):
            return "scanner"
    for base in (50.0, 60.0):
        harmonic = round(frequency/base)
        if harmonic >= 1 and abs(frequency - harmonic*base) <= max(.05, df/2):
            return f"mains_{int(base)}"
    return "unattributed"


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def _filter_lines(x, srate, selected):
    sections = []
    for candidate in selected:
        frequency, width = candidate["frequency_hz"], candidate["notch_bandwidth_hz"]
        if not 0 < frequency < srate/2 or not np.isfinite(width) or width <= 0:
            raise ValueError("Invalid frequency or width in line plan")
        b, a = sig.iirnotch(frequency, frequency/width, fs=srate)
        sections.append(sig.tf2sos(b, a))
    if not sections:
        return x.copy(), 0
    padlen = min(len(x)-1, int(np.ceil(3*srate/min(c["notch_bandwidth_hz"] for c in selected))))
    y = sig.sosfiltfilt(np.vstack(sections), x.astype(np.float64), padlen=padlen).astype(x.dtype)
    return y, padlen


def apply_line_plan(signal: np.ndarray, srate: float, report: dict) -> np.ndarray:
    """Apply an already detected plan, principally for controlled validation.

    Holding the plan fixed distinguishes filter attenuation of injected EMG
    from changes in which lines an adaptive detector selects. This does not
    validate transferring a plan between runs with different scanner timing.
    """
    x = np.asarray(signal)
    if not np.isfinite(srate) or srate <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.floating) or not np.all(np.isfinite(x)):
        raise ValueError("Line plan requires a finite floating 1-D signal")
    selected = [c for c in report["candidates"] if c["selected"]]
    if report.get("mode") == "regress":
        return regress_lines(x, srate, selected, report["parameters"].get("regression_window_seconds", 32.))[0]
    return _filter_lines(x, srate, selected)[0]


def clean_spectral_lines(
    signal: np.ndarray,
    srate: float,
    *,
    reference: np.ndarray | None = None,
    witness: np.ndarray | None = None,
    sdur: float | None = None,
    tr: float | None = None,
    mode: str = "off",
    band: tuple[float, float] = (30.0, 250.0),
    min_prominence_db: float = 8.0,
    max_width_hz: float = 0.5,
    min_persistence: float = 0.75,
    max_lines: int = 12,
    window_seconds: float = 8.0,
    regression_window_seconds: float = 32.0,
) -> tuple[np.ndarray, dict]:
    """Detect persistent lines and optionally apply zero-phase narrow notches.

    ``mode='off'`` and ``'detect'`` preserve every input sample. ``'notch'``
    filters only the selected, supported lines. A notch also attenuates genuine
    EMG at that frequency; reports and independent injections must be used to
    assess that trade-off. Original/reference evidence is required for automatic
    correction. Regression may additionally use a common-channel harmonic family
    with volume-spaced sidebands; this remains evidence, not physiological proof.
    """
    x = np.asarray(signal)
    if not np.isfinite(srate) or srate <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    if x.ndim != 1 or len(x) == 0 or not np.issubdtype(x.dtype, np.floating):
        raise ValueError("Spectral line cleanup requires a non-empty floating 1-D signal")
    if not np.all(np.isfinite(x)):
        raise ValueError("Spectral line cleanup requires finite signal samples")
    if mode not in {"off", "detect", "notch", "regress"}:
        raise ValueError("Line cleanup mode must be off, detect, notch or regress")
    if len(band) != 2 or not (0 < band[0] < band[1] < srate/2):
        raise ValueError("Line detection band must lie strictly inside Nyquist")
    if (not np.isfinite(min_prominence_db) or min_prominence_db <= 0 or
            not np.isfinite(max_width_hz) or max_width_hz <= 0 or
            not 0 < min_persistence <= 1 or max_lines < 1 or
            not np.isfinite(window_seconds) or window_seconds < 4 or
            not np.isfinite(regression_window_seconds) or regression_window_seconds < 16):
        raise ValueError("Invalid line detection thresholds, count or window length")
    if reference is not None:
        reference = np.asarray(reference)
        if reference.shape != x.shape or not np.all(np.isfinite(reference)):
            raise ValueError("Reference and signal must have equal shapes and finite samples")

    if witness is not None:
        witness = np.asarray(witness)
        if witness.shape != x.shape or not np.isfinite(witness).all():
            raise ValueError("Witness must match signal length and contain finite samples")
    report = dict(mode=mode, applied=False, applied_line_count=0, candidates=[],
                  parameters=dict(band_hz=list(band), min_prominence_db=min_prominence_db,
                                  max_width_hz=max_width_hz, min_persistence=min_persistence,
                                  max_lines=max_lines, window_seconds=window_seconds,
                                  regression_window_seconds=regression_window_seconds),
                  reference_available=reference is not None, witness_available=witness is not None)
    if mode == "off":
        report["reason"] = "disabled"
        return x.copy(), report
    result = _spectra(x, srate, window_seconds)
    if result is None:
        report["reason"] = "insufficient_duration_for_four_windows_of_at_least_four_seconds"
        return x.copy(), report
    f, windows, length = result
    df = float(f[1]-f[0])
    psd = np.median(windows, axis=0)
    floor = _floor(psd, df, max_width_hz)
    ratio_db = 10*np.log10(np.maximum(psd, np.finfo(float).tiny)/floor)
    peaks, _ = sig.find_peaks(ratio_db, height=min_prominence_db,
                              prominence=min_prominence_db/2,
                              distance=max(1, int(np.ceil(max_width_hz/df))))
    peaks = peaks[(f[peaks] >= band[0]) & (f[peaks] <= band[1])]
    excess = np.maximum(psd-floor, 0)
    # Ratio peaks without a resolved power maximum are explicitly rejected
    # below. SciPy warns for their zero widths; retain that reason in metadata.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="some peaks have.*")
        widths = sig.peak_widths(excess, peaks, rel_height=.5,
                                 wlen=max(3, int(round(6/df)) | 1))[0]*df
    reference_psd = None
    if reference is not None:
        reference_psd = np.median(_spectra(reference, srate, window_seconds)[1], axis=0)
        reference_floor = _floor(reference_psd, df, max_width_hz)
    report.update(window_samples=length, sampled_windows=len(windows),
                  frequency_resolution_hz=df, reason="no_supported_lines")
    window_floors = np.array([_floor(p, df, max_width_hz) for p in windows])
    for peak, width in zip(peaks, widths):
        # Sub-bin refinement improves centering for rates not commensurate
        # with the periodogram grid (e.g. TR=1.660 s).
        logp = np.log(np.maximum(psd[peak-1:peak+2], np.finfo(float).tiny))
        curvature = logp[0]-2*logp[1]+logp[2]
        offset = .5*(logp[0]-logp[2])/curvature if abs(curvature) > 1e-20 else 0.
        frequency = float(f[peak] + np.clip(offset, -.5, .5)*df)
        nearby = slice(max(0, peak-1), min(len(f), peak+2))
        window_ratios = np.max(windows[:, nearby]/window_floors[:, nearby], axis=1)
        persistence = float(np.mean(10*np.log10(np.maximum(window_ratios, 1e-30)) >= min_prominence_db))
        association = _association(frequency, df, sdur, tr)
        reference_db = (
            float(10*np.log10(max(reference_psd[peak], np.finfo(float).tiny)/reference_floor[peak]))
            if reference_psd is not None else None
        )
        reasons = []
        if width <= 0:
            reasons.append("not_a_resolved_power_peak")
        if width > max_width_hz:
            reasons.append("too_broad")
        if persistence < min_persistence:
            reasons.append("not_persistent")
        if association == "unattributed":
            reasons.append("no_scanner_or_mains_support")
        if reference_db is None or reference_db < 3.0:
            reasons.append("not_supported_in_original")
        report["candidates"].append(dict(
            frequency_hz=frequency, prominence_db=float(ratio_db[peak]),
            fwhm_hz=float(width), persistence=persistence, association=association,
            reference_prominence_db=reference_db, eligible=not reasons,
            rejection_reasons=reasons, selected=False, applied=False,
            # This is the single-pass -3 dB bandwidth. filtfilt squares the
            # magnitude response; effective attenuation extends beyond it.
            notch_bandwidth_hz=float(min(max_width_hz, max(2*df, width))),
        ))
    # Refine regression frequencies before evaluating harmonic/sideband support.
    # Notch/detect keep their historical selection criteria unchanged.
    if mode == "regress" and report['candidates']:
        refined = refine_frequencies(reference if reference is not None else x, srate,
                                     [c['frequency_hz'] for c in report['candidates']], df)
        for candidate, frequency in zip(report['candidates'], refined):
            candidate['coarse_frequency_hz'] = candidate['frequency_hz']
            candidate['frequency_hz'] = frequency
        if witness is not None and reference is not None and tr is not None and tr > 0:
            supported = [c for c in report['candidates'] if all(reason == 'no_scanner_or_mains_support'
                         for reason in c['rejection_reasons'])]
            tolerance = max(.01, df/8)
            anchors = [c['frequency_hz'] for c in supported if any(
                other is not c and (abs(2*c['frequency_hz']-other['frequency_hz']) <= tolerance
                                  or abs(c['frequency_hz']-2*other['frequency_hz']) <= tolerance)
                for other in supported)]
            # A carrier's harmonic may already have been removed by FARM in
            # this channel. Check its presence in BOTH original channels too.
            witness_psd = np.median(_spectra(witness, srate, window_seconds)[1], axis=0)
            witness_floor = _floor(witness_psd, df, max_width_hz)
            for candidate in supported:
                for related in (candidate['frequency_hz']/2, candidate['frequency_hz']*2):
                    if band[0] <= related <= band[1]:
                        ref_db = 10*np.log10(max(float(np.interp(related, f, reference_psd)), 1e-300)
                                           /max(float(np.interp(related, f, reference_floor)), 1e-300))
                        other_db = 10*np.log10(max(float(np.interp(related, f, witness_psd)), 1e-300)
                                             /max(float(np.interp(related, f, witness_floor)), 1e-300))
                        if min(ref_db, other_db) >= 6:
                            anchors.append(candidate['frequency_hz'])
                            candidate['original_related_harmonic_hz'] = related
                            candidate['original_related_harmonic_min_prominence_db'] = float(min(ref_db, other_db))
                            break
            frequencies = sorted(set([c['frequency_hz'] for c in supported]
                                     + [c['original_related_harmonic_hz'] for c in supported
                                        if 'original_related_harmonic_hz' in c]))
            values = coherence_at_frequencies(reference, witness, srate, length, frequencies)
            coherence = dict(zip(frequencies, values))
            for candidate in supported:
                frequency = candidate['frequency_hz']
                common = float(coherence[frequency])
                candidate['original_cross_channel_coherence'] = common
                related = candidate.get('original_related_harmonic_hz')
                family_common = max(common, float(coherence[related])) if related is not None else common
                candidate['original_harmonic_family_coherence'] = family_common
                # Common carrier plus a detected harmonic and volume-spaced
                # sidebands. This supports an interference hypothesis, not proof.
                family = any(abs((frequency-anchor)*tr-round((frequency-anchor)*tr))/tr <= tolerance
                             for anchor in anchors)
                if candidate['association'] == 'unattributed' and family_common >= .8 and family:
                    candidate['association'] = 'common_harmonic_scanner_sideband'
                    candidate['rejection_reasons'].remove('no_scanner_or_mains_support')
                    candidate['eligible'] = not candidate['rejection_reasons']
    if mode == "regress":
        for candidate in report['candidates']:
            # Continuous confidence suppresses large jumps at hard detector
            # boundaries under small changes in physiological activity.
            confidence = np.clip((candidate['prominence_db']-min_prominence_db)/8., 0., 1.)
            confidence *= np.clip((candidate['persistence']-min_persistence)/max(1-min_persistence, 1e-12), 0., 1.) if min_persistence < 1 else 1.
            confidence *= np.clip((max_width_hz-candidate['fwhm_hz'])/(max_width_hz/2), 0., 1.)
            reference_db = candidate['reference_prominence_db']
            confidence *= np.clip((reference_db-3.)/3., 0., 1.) if reference_db is not None else 0.
            if candidate['association'] == 'common_harmonic_scanner_sideband':
                confidence *= np.clip((candidate['original_harmonic_family_coherence']-.8)/.15, 0., 1.)
            candidate['regression_weight'] = float(confidence)
            if candidate['eligible'] and confidence == 0:
                candidate['eligible'] = False
                candidate['rejection_reasons'].append('zero_confidence_at_selection_boundary')
    eligible = sorted([c for c in report["candidates"] if c["eligible"]],
                      key=lambda c: c["prominence_db"], reverse=True)
    selected = eligible[:max_lines]
    for candidate in selected:
        candidate["selected"] = True
    for candidate in eligible[max_lines:]:
        candidate["rejection_reasons"].append("line_count_limit")
    report["eligible_line_count"] = len(eligible)
    report["selected_line_count"] = len(selected)
    report["summed_notch_bandwidth_hz"] = sum(c["notch_bandwidth_hz"] for c in selected)
    y = x.copy()
    if mode == "notch" and selected:
        for candidate in selected:
            candidate["applied"] = True
        # Ordinary sosfiltfilt padding is far too short for a sub-Hz notch.
        y, padlen = _filter_lines(x, srate, selected)
        report.update(applied=True, applied_line_count=len(selected), padding_samples=padlen)
    if mode == "regress" and selected:
        y, length_fit = regress_lines(x, srate, selected, regression_window_seconds)
        for candidate in selected:
            candidate["applied"] = True
        report.update(applied=True, applied_line_count=len(selected), regression_window_samples=length_fit)
    if selected:
        report["reason"] = "corrected" if report["applied"] else "detection_only"
    after = np.median(_spectra(y, srate, window_seconds)[1], axis=0) if report["applied"] else psd
    after_floor = _floor(after, df, max_width_hz)
    masks = np.zeros(len(f), dtype=bool)
    for candidate in selected:
        region = np.abs(f-candidate["frequency_hz"]) <= max(2*df, candidate["notch_bandwidth_hz"])
        masks |= region
        before_power = float(np.maximum(psd[region]-floor[region], 0).sum()*df)
        after_power = float(np.maximum(after[region]-after_floor[region], 0).sum()*df)
        candidate.update(excess_power_before=before_power, excess_power_after=after_power,
                         reduction_db=float(10*np.log10(max(float(psd[region].sum()), 1e-300)/max(float(after[region].sum()), 1e-300))),
                         excess_reduction_db=(float(10*np.log10(before_power/after_power))
                                              if before_power > 0 and after_power > 0 else None))
    inside = (f >= band[0]) & (f <= band[1])
    outside = inside & ~masks
    off_before = float(psd[outside].sum())
    envelope_length = max(1, int(round(.05*srate)))
    env_x = np.sqrt(np.maximum(ndimage.uniform_filter1d(x.astype(np.float64)**2, envelope_length), 0))
    env_y = np.sqrt(np.maximum(ndimage.uniform_filter1d(y.astype(np.float64)**2, envelope_length), 0))
    band_x = apply_bandpass(x, srate, band).astype(np.float64)
    band_y = apply_bandpass(y, srate, band).astype(np.float64)
    band_env_x = np.sqrt(np.maximum(ndimage.uniform_filter1d(band_x**2, envelope_length), 0))
    band_env_y = np.sqrt(np.maximum(ndimage.uniform_filter1d(band_y**2, envelope_length), 0))
    report["metrics"] = dict(
        rms_before=_rms(x), rms_after=_rms(y), correction_rms=_rms(x.astype(np.float64)-y),
        relative_envelope_change=float(np.linalg.norm(env_y-env_x)/max(np.linalg.norm(env_x), 1e-30)),
        band_relative_envelope_change=float(np.linalg.norm(band_env_y-band_env_x)/max(np.linalg.norm(band_env_x), 1e-30)),
        envelope_band_hz=list(band),
        off_line_power_ratio=float(after[outside].sum()/off_before) if off_before > 0 else None,
        excess_power_before=float(np.maximum(psd[masks]-floor[masks], 0).sum()*df),
        excess_power_after=float(np.maximum(after[masks]-after_floor[masks], 0).sum()*df),
    )
    return y, report


def line_cleanup_options(cfg) -> dict:
    """Shared configuration for workflow and independent injection validation."""
    return dict(mode=cfg.line_cleanup, band=cfg.line_band,
                min_prominence_db=cfg.line_min_prominence_db,
                max_width_hz=cfg.line_max_width_hz,
                min_persistence=cfg.line_min_persistence,
                max_lines=cfg.line_max_count, window_seconds=cfg.line_window_seconds,
                regression_window_seconds=cfg.line_regression_window_seconds)

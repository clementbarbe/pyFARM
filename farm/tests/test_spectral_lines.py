"""Residual-line rejection and independent physiological recovery checks."""

import json

import mne
import numpy as np
import pytest
from scipy import signal as sig

from farm.config import FARMConfig
from farm.export.brainvision import export_brainvision
from farm.preprocessing.lines import clean_spectral_lines, apply_line_plan
from farm.workflow import run_pipeline


def example(seed=4, seconds=64):
    fs = 1000.
    t = np.arange(int(seconds*fs))/fs
    rng = np.random.default_rng(seed)
    background = rng.standard_normal(len(t))*.2
    physiological = .4*np.sin(2*np.pi*120.3*t)
    contaminated = (background + physiological + np.sin(2*np.pi*170*t)).astype(np.float32)
    return contaminated, fs, t, background, physiological


def correct(x, fs, **kwargs):
    options = dict(reference=x, sdur=.1, tr=1.6, mode="notch")
    options.update(kwargs)
    return clean_spectral_lines(x, fs, **options)


def amplitude(x, fs, frequency):
    t = np.arange(len(x))/fs
    return float(2*abs(np.dot(x, np.exp(-2j*np.pi*frequency*t)))/len(x))


@pytest.mark.parametrize("mode", ["off", "detect"])
def test_detection_never_changes_samples(mode):
    x, fs, _, _, _ = example()
    y, report = correct(x, fs, mode=mode)
    np.testing.assert_array_equal(y, x)
    assert not report["applied"]
    if mode == "detect":
        assert report["selected_line_count"] >= 1


def test_persistent_line_removed_without_removing_nearby_physio_tone():
    x, fs, _, _, physiological = example()
    y, report = correct(x, fs)
    assert report["applied_line_count"] == 1
    assert amplitude(y, fs, 170) < .05*amplitude(x, fs, 170)
    gain = amplitude(y, fs, 120.3)/amplitude(x, fs, 120.3)
    assert .99 < gain < 1.01
    assert report["candidates"][0]["persistence"] >= .75
    assert np.isfinite(y).all() and y.dtype == x.dtype


def test_broadband_noise_alone_is_preserved_exactly():
    _, fs, _, noise, _ = example(seed=23)
    x = noise.astype(np.float32)
    y, report = correct(x, fs)
    np.testing.assert_array_equal(y, x)
    assert report["applied_line_count"] == 0


def test_broad_physiological_hump_is_preserved():
    _, fs, _, noise, _ = example(seed=21)
    sos = sig.butter(4, [160, 180], btype="bandpass", fs=fs, output="sos")
    x = (noise*.1 + 15*sig.sosfiltfilt(sos, noise)).astype(np.float32)
    y, report = correct(x, fs)
    np.testing.assert_array_equal(y, x)
    assert not report["applied"]


def test_transient_oscillation_is_not_removed_globally():
    _, fs, t, noise, _ = example(seed=12)
    x = (noise + 3*np.sin(2*np.pi*170*t)*(t < 8)).astype(np.float32)
    y, report = correct(x, fs)
    np.testing.assert_array_equal(y, x)


def test_unattributed_narrow_line_is_reported_but_preserved():
    _, fs, t, noise, _ = example()
    x = (noise + 2*np.sin(2*np.pi*153.2*t)).astype(np.float32)
    y, report = correct(x, fs)
    np.testing.assert_array_equal(y, x)
    assert any("no_scanner_or_mains_support" in c["rejection_reasons"] for c in report["candidates"])


def test_line_absent_in_original_is_not_automatically_removed():
    x, fs, _, noise, _ = example(seed=19)
    y, report = correct(x, fs, reference=noise)
    np.testing.assert_array_equal(y, x)
    assert any("not_supported_in_original" in c["rejection_reasons"] for c in report["candidates"])


def test_no_reference_means_detection_without_automatic_notches():
    x, fs, _, _, _ = example()
    y, report = correct(x, fs, reference=None)
    np.testing.assert_array_equal(y, x)
    assert report["selected_line_count"] == 0


def test_line_limit_does_not_delete_every_spectral_peak():
    x, fs, t, _, _ = example()
    x = x + .7*np.sin(2*np.pi*190*t)
    y, report = correct(x, fs, max_lines=1)
    assert report["eligible_line_count"] == 2
    assert report["applied_line_count"] == 1
    assert len([c for c in report["candidates"] if c["applied"]]) == 1


def test_extra_stage_preserves_independent_emg_injection():
    x, fs, t, _, _ = example()
    rng = np.random.default_rng(71)
    sos = sig.butter(4, [40, 300], btype="bandpass", fs=fs, output="sos")
    emg = sig.sosfiltfilt(sos, rng.standard_normal(len(t)))
    emg *= .1 + sum(np.exp(-.5*((t-c)/.15)**2) for c in [4.3, 15.2, 32.7, 54.1])
    emg = emg.astype(np.float32)
    baseline, _ = correct(x, fs)
    with_injection, _ = correct(x+emg, fs)
    recovered = with_injection-baseline
    gain = np.dot(recovered, emg)/np.dot(emg, emg)
    error = np.linalg.norm(recovered-emg)/np.linalg.norm(emg)
    assert .98 < gain < 1.02, gain
    assert error < .08, error


def test_fixed_plan_isolates_filter_effect_on_known_emg():
    x, fs, t, _, _ = example()
    _, plan = correct(x, fs, mode="detect")
    injected = np.random.default_rng(87).standard_normal(len(t)).astype(np.float32)*.1
    recovered = apply_line_plan(x+injected, fs, plan)-apply_line_plan(x, fs, plan)
    gain = np.dot(recovered, injected)/np.dot(injected, injected)
    assert .99 < gain < 1.01
    np.testing.assert_allclose(recovered, apply_line_plan(injected, fs, plan), atol=3e-7)


def test_slightly_low_mains_frequency_is_recognized():
    _, fs, t, noise, _ = example()
    x = (noise + np.sin(2*np.pi*49.99*t)).astype(np.float32)
    _, report = correct(x, fs, sdur=None, tr=None)
    assert report["applied_line_count"] == 1
    selected = next(c for c in report["candidates"] if c["applied"])
    assert selected["association"] == "mains_50"


def test_short_recording_and_zero_signal_are_identity():
    for x in [np.ones(1000, dtype=np.float32), np.zeros(32000, dtype=np.float32)]:
        y, report = correct(x, 1000.)
        np.testing.assert_array_equal(y, x)
        assert not report["applied"]


@pytest.mark.parametrize("options", [dict(band=(30, 600)), dict(min_persistence=0),
                                     dict(max_width_hz=-1), dict(mode="unknown"),
                                     dict(window_seconds=2), dict(min_prominence_db=float("nan"))])
def test_invalid_settings_fail_clearly(options):
    x, fs, _, _, _ = example(seconds=16)
    with pytest.raises(ValueError):
        correct(x, fs, **options)


def test_nonfinite_input_is_rejected():
    x, fs, _, _, _ = example(seconds=16)
    x[100] = np.nan
    with pytest.raises(ValueError, match="finite"):
        correct(x, fs)


def test_workflow_line_stage_preserves_crop_boundaries_and_markers(tmp_path, monkeypatch):
    fs, tr, count, prefix, suffix = 1000., .5, 40, 73, 100
    t = np.arange(int(count*tr*fs))/fs
    x = (np.random.default_rng(41).standard_normal(len(t))*.2 + np.sin(2*np.pi*150*t))*1e-5
    full = np.r_[np.ones(prefix)*1e-6, x, np.ones(suffix)*2e-6]
    raw = mne.io.RawArray(full[None, :], mne.create_info(["EXT"], fs, "emg"), verbose=False)
    times = prefix/fs + np.arange(count)*tr
    raw.set_annotations(mne.Annotations(times, np.zeros(count), ["Response/R128"]*count))
    path = export_brainvision(raw, raw.get_data(), [0], str(tmp_path), "source")
    monkeypatch.setattr("farm.workflow._estimate_interval_artifact",
                        lambda reference, *args: np.zeros_like(reference))
    cfg = FARMConfig(vhdr_path=path, tr=tr, n_slices=4, mb_factor=1,
                     timing_mode="fixed", fixed_sdur_seconds=.1,
                     figures_dir="", output_dir=str(tmp_path/"out"), line_cleanup="notch")
    result = run_pipeline(cfg)
    data = result["data_clean_full"][0]
    np.testing.assert_array_equal(data[:prefix], result["data_full_raw"][0, :prefix])
    np.testing.assert_array_equal(data[-suffix:], result["data_full_raw"][0, -suffix:])
    assert amplitude(result["data_clean_crop"][0], fs, 150) < .1*amplitude(x, fs, 150)
    metadata = json.loads(open(result["metadata_path"]).read())
    assert metadata["line_cleanup"]["channels"][0]["applied"]
    exported = mne.io.read_raw_brainvision(result["brainvision_path"], preload=False, verbose=False)
    events, _ = mne.events_from_annotations(exported, verbose=False)
    np.testing.assert_array_equal(events[:, 0], np.rint(times*fs))


def test_regression_common_harmonic_family_rejects_lines_preserves_nearby_emg():
    fs = 1000.
    t = np.arange(int(80*fs))/fs
    rng = np.random.default_rng(99)
    carrier = 3*np.sin(2*np.pi*97.3*t)+np.sin(2*np.pi*194.6*t)
    muscle = .4*np.sin(2*np.pi*97.65*t)
    witness = (carrier+rng.normal(0,.2,len(t))).astype(np.float32)
    x = (carrier+muscle+rng.normal(0,.2,len(t))).astype(np.float32)
    y, report = clean_spectral_lines(x, fs, reference=x, witness=witness,
                                     sdur=.1, tr=1.6, mode='regress')
    assert report['applied_line_count'] == 2
    assert amplitude(y, fs, 97.3) < .02*amplitude(x, fs, 97.3)
    assert amplitude(y, fs, 194.6) < .02*amplitude(x, fs, 194.6)
    assert .98 < amplitude(y, fs, 97.65)/amplitude(x, fs, 97.65) < 1.02
    assert all(c['association']=='common_harmonic_scanner_sideband'
               for c in report['candidates'] if c['selected'])
    np.testing.assert_array_equal(apply_line_plan(x, fs, report), y)


def test_regression_common_single_tone_without_harmonic_support_is_unchanged():
    x, fs, t, noise, _ = example(seconds=64)
    tone = np.sin(2*np.pi*119.7*t)
    x = (noise+tone).astype(np.float32)
    witness = (np.random.default_rng(33).normal(0,.2,len(t))+tone).astype(np.float32)
    y, report = clean_spectral_lines(x, fs, reference=x, witness=witness,
                                     sdur=.1, tr=1.6, mode='regress')
    np.testing.assert_array_equal(x, y)
    assert not report['applied']


def test_regression_known_scanner_line_tracks_slow_modulation():
    fs = 1000.
    t = np.arange(int(96*fs))/fs
    rng = np.random.default_rng(14)
    carrier = (1+.4*np.sin(2*np.pi*t/64))*np.sin(2*np.pi*170*t)
    x = (carrier+.03*rng.normal(size=len(t))).astype(np.float32)
    y, report = clean_spectral_lines(x, fs, reference=x, sdur=.1, tr=1.6, mode='regress')
    assert report['applied']
    assert np.linalg.norm(y) < .2*np.linalg.norm(x)
    assert np.isfinite(y).all()

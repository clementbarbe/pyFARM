"""Meaningful checks of paired QC, including known scanner contamination."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "simple_qc", Path(__file__).resolve().parents[2]/"scripts/qc_before_after.py")
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


def test_scanner_proxy_improves_when_known_artifact_removed():
    fs, tr, n = 1000., 1., 30
    rng = np.random.default_rng(27)
    t = np.arange(int(fs*tr))/fs
    artifact = 20e-6*np.sin(2*np.pi*72*t)
    emg = rng.normal(0, 2e-6, (n, len(t)))
    before = (emg+artifact).ravel()
    after = (emg+artifact*.01).ravel()
    onsets = np.arange(n)*len(t)
    b = qc.locked_metrics(before, fs, onsets, tr)
    a = qc.locked_metrics(after, fs, onsets, tr)
    assert b['locked_epochs'] == n
    assert b['locked_rms_uv'] > a['locked_rms_uv']*20
    assert a['snr_proxy_db'] > b['snr_proxy_db']+25
    assert abs(a['asynchronous_rms_uv']-b['asynchronous_rms_uv']) < 1e-5


def test_no_scanner_reference_is_missing_not_a_good_snr():
    result = qc.locked_metrics(np.zeros(5000), 1000, [0, 1000, 2000], 1.)
    assert result['snr_proxy_db'] is None
    short = qc.locked_metrics(np.ones(2000), 1000, [0, 1000], 1.)
    assert short['locked_rms_uv'] is None
    assert short['locked_epochs'] == 2


def test_extrema_are_actual_samples_in_microvolts():
    x = np.random.default_rng(1).normal(0, 3e-6, 10000)
    x[500] = -100e-6
    x[501] = 200e-6
    metrics = qc.channel_metrics(x, 1000., np.arange(0, 10000, 1000), 1., (30, 250))
    assert metrics['min_uv'] == pytest.approx(-100)
    assert metrics['max_uv'] == pytest.approx(200)
    assert metrics['rms_uv'] == pytest.approx(np.sqrt(np.mean(x*x))*1e6)
    with pytest.raises(ValueError, match='NaN'):
        qc.channel_metrics(np.full(1000, np.nan), 1000., [0], 1., (30, 250))


def test_mismatched_pair_is_reported_not_silently_truncated(tmp_path, monkeypatch):
    import json
    class Raw:
        info = {'sfreq': 1000.}
        def __init__(self, n):
            self.n_times = n
        def close(self):
            pass
    raws = iter([Raw(10000), Raw(9999)])
    monkeypatch.setattr(qc.mne.io, 'read_raw_brainvision', lambda *a, **k: next(raws))
    sidecar = tmp_path/'run_denoise.json'
    sidecar.write_text(json.dumps({'input_vhdr': 'raw.vhdr', 'output_vhdr': 'clean.vhdr'}))
    with pytest.raises(ValueError, match='lengths differ'):
        qc.read_pair(sidecar, (30, 250))

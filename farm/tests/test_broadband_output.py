import numpy as np

from farm.preprocessing.filters import apply_bandpass
from farm.preprocessing.resampling import upsample, downsample


def _tone_amplitude(x: np.ndarray, fs: float, f0: float) -> float:
    t = np.arange(x.shape[-1], dtype=np.float64) / fs
    ref = np.exp(-2j * np.pi * f0 * t)
    return float(2.0 * np.abs(np.dot(x.astype(np.float64), ref)) / len(x))


def test_resampling_roundtrip_does_not_impose_250_hz_lowpass():
    """Core FARM resampling must preserve native-rate broadband content."""
    fs = 5000.0
    t = np.arange(int(2.0 * fs), dtype=np.float64) / fs
    x = (0.5 * np.sin(2 * np.pi * 100.0 * t)
         + 0.4 * np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)
    data = x[None, :]
    onsets = np.array([0], dtype=np.int64)

    up, fs_up, _ = upsample(data, fs, onsets, factor=4)
    y = downsample(up, fs_up, factor=4, target_length=len(x))[0]

    a400_in = _tone_amplitude(x, fs, 400.0)
    a400_out = _tone_amplitude(y, fs, 400.0)
    assert a400_out / a400_in > 0.90


def test_analysis_band_is_an_explicit_single_derived_branch():
    fs = 5000.0
    t = np.arange(int(2.0 * fs), dtype=np.float64) / fs
    x = (np.sin(2 * np.pi * 100.0 * t)
         + np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)

    y = apply_bandpass(x, fs, (30.0, 250.0))
    a100 = _tone_amplitude(y, fs, 100.0)
    a400 = _tone_amplitude(y, fs, 400.0)

    assert a100 > 0.8
    assert a400 < 0.03


def test_multichannel_bandpass_matches_channelwise_application():
    fs = 5000.0
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 12000)).astype(np.float32)

    batch = apply_bandpass(x, fs, (30.0, 250.0))
    serial = np.vstack([apply_bandpass(row, fs, (30.0, 250.0)) for row in x])

    np.testing.assert_allclose(batch, serial, rtol=1e-6, atol=1e-7)

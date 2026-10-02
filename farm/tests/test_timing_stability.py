import numpy as np
import pytest

from farm.config import FARMConfig
from farm.timing.optimization import optimize_global_timing


def test_default_timing_mode_is_safe_initial():
    cfg = FARMConfig(vhdr_path="dummy.vhdr")
    assert cfg.timing_mode == "initial"


def test_fixed_timing_requires_sdur():
    cfg = FARMConfig(vhdr_path="dummy.vhdr", timing_mode="fixed")
    with pytest.raises(ValueError):
        cfg.validate()


def test_local_refinement_cannot_escape_physical_neighbourhood():
    # Synthetic trigger-anchored periodic gradient-like signal.
    srate = 2000.0
    tr = 1.66
    n_sg = 20
    n_vol = 24
    true_sdur = 0.0820
    n = int(round(n_vol * tr * srate)) + 100
    x = np.zeros(n, dtype=np.float64)
    vol_onsets = (np.arange(n_vol) * tr * srate).astype(np.int64)

    pulse = np.array([0.0, 1.0, -0.8, 0.4, -0.2, 0.0])
    for v0 in vol_onsets:
        for g in range(n_sg):
            idx = int(round(v0 + g * true_sdur * srate))
            if idx + len(pulse) < n:
                x[idx:idx + len(pulse)] += pulse

    init = 0.08195
    half_width = 0.0005
    sdur, dtime, result = optimize_global_timing(
        ref_signal_up=x,
        srate_up=srate,
        vol_onsets_up=vol_onsets,
        sdur_init=init,
        dtime_init=tr - n_sg * init,
        n_sg=n_sg,
        n_vol=n_vol,
        padding=4,
        window_size=10,
        refine_half_width_seconds=half_width,
        min_relative_improvement=0.0,
    )

    assert abs(sdur - init) <= half_width + 1e-12
    assert np.isclose(n_sg * sdur + dtime, tr, atol=1 / srate)
    assert hasattr(result, "accepted")

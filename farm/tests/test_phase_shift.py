"""Tests for trigger-anchored slice marker construction."""

import numpy as np

from farm.timing.optimization import compute_slice_markers


def test_slice_markers_anchor_each_volume_to_real_trigger():
    srate = 1000.0
    sdur = 0.100
    n_sg = 4

    # Volume 2 has a real trigger jitter: 1.005 s instead of exactly 1.000 s.
    volume_onsets = np.array([0, 1000, 2005], dtype=np.int64)

    onsets, errors, seg_len = compute_slice_markers(
        vol_onsets_up=volume_onsets,
        sdur=sdur,
        dtime=0.0,
        srate_up=srate,
        n_sg=n_sg,
        n_vol=3,
    )

    assert seg_len == 100
    assert np.allclose(errors, 0.0)

    # First group of each volume must equal the real trigger exactly.
    assert onsets[0] == 0
    assert onsets[4] == 1000
    assert onsets[8] == 2005

    # Second volume markers are anchored at 1000, not generated from
    # a theoretical grid started at zero.
    assert np.array_equal(onsets[4:8], [1000, 1100, 1200, 1300])
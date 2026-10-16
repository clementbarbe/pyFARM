"""Conservative native-rate mask for samples affected by zero-filled EPI gaps."""
from __future__ import annotations

import numpy as np


def gap_mask_native(onsets_up: np.ndarray, last_slice_idx: np.ndarray,
                    seg_len_up: int, n_samples_native: int, interp_factor: int,
                    gap_fraction: float, guard_samples: int = 16) -> np.ndarray:
    """Mark zero-filled gaps and surrounding guard. True means unreliable."""
    if n_samples_native < 0 or interp_factor < 1 or seg_len_up < 1:
        raise ValueError("Invalid sampling dimensions")
    if not 0 <= gap_fraction <= 1 or guard_samples < 0:
        raise ValueError("Invalid gap fraction or guard_samples")
    mask = np.zeros(int(n_samples_native), dtype=np.bool_)
    if gap_fraction == 0:
        return mask
    onsets = np.asarray(onsets_up, dtype=np.int64)
    for i in np.asarray(last_slice_idx, dtype=np.int64):
        if i < 0 or i + 1 >= len(onsets):
            continue
        start_up = int(onsets[i]) + int(seg_len_up)
        stop_up = int(onsets[i+1])
        if stop_up <= start_up:
            continue
        stop_up = start_up + int(round(gap_fraction * (stop_up - start_up)))
        if stop_up <= start_up:
            continue
        start = max(0, start_up // interp_factor - guard_samples)
        stop = min(n_samples_native, (stop_up + interp_factor - 1) // interp_factor + guard_samples)
        if stop > start:
            mask[start:stop] = True
    return mask

"""Export cleaned data as NumPy (.npz) and MATLAB (.mat) files."""

import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import savemat

logger = logging.getLogger("farm.export.arrays")


def export_npz(
    output_dir: str, basename: str, **arrays
) -> str:
    """Write a compressed ``.npz`` archive.

    All keyword arguments are stored as named arrays.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = str(out / f"{basename}.npz")
    np.savez_compressed(path, **arrays)
    logger.info("NPZ: %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
    return path


def export_mat(
    output_dir: str, basename: str, **arrays
) -> str:
    """Write a compressed ``.mat`` (v5) file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = str(out / f"{basename}.mat")
    savemat(path, arrays, do_compression=True)
    logger.info("MAT: %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
    return path
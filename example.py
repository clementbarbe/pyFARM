"""
FARM — Minimal usage example.

Run:
    python example.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

from farm.config import FARMConfig
from farm.workflow import run_pipeline


cfg = FARMConfig(
    vhdr_path="data/me3mb3_tr1600_sl54.vhdr",

    # Scanner sequence
    tr=1.6,
    n_slices=54,
    mb_factor=3,

    # 54 / 3 => 18 acquisition slice-groups:
    # 18 complete MB groups.
    trigger="R128",

    # EMG channel selection
    ch_regex=r"EXT|FLE",

    # Template candidates within +/- 50 volumes, same slice-group only
    window_size=50,
    n_candidates=12,

    # Set True only if final scanner volume is known to be incomplete.
    drop_last_volume=True,

    # This protocol shows a strong scanner-locked boundary artifact in the
    # true inter-volume dead-time.  Mask the complete measured gap, matching
    # the behaviour that gave the cleanest spectrum on the supplied dataset.
    zero_fill_gap_fraction=1.0,

    output_dir="output",
)

results = run_pipeline(cfg)

print(f"\nCleaned data shape : {results['data_clean_full'].shape}")
print(f"Optimised sdur     : {results['sdur'] * 1e3:.4f} ms")
print(f"Derived dtime      : {results['dtime'] * 1e3:.4f} ms")
print(f"Figures saved to   : {results['figures_dir']}")
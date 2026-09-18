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
    vhdr_path="data/me3mb3_tr1600_sl53.vhdr",

    # Scanner sequence
    tr=1.6,
    n_slices=53,
    mb_factor=3,

    # 53 / 3 => 18 acquisition slice-groups:
    # 17 complete MB groups + one final potentially partial group.
    trigger="R128",

    # EMG channel selection
    ch_regex=r"ZYG|COR",

    # Template candidates within +/- 50 volumes, same slice-group only
    window_size=50,
    n_candidates=12,

    # Set True only if final scanner volume is known to be incomplete.
    drop_last_volume=True,

    output_dir="output",
)

results = run_pipeline(cfg)

print(f"\nCleaned data shape : {results['data_clean_full'].shape}")
print(f"Optimised sdur     : {results['sdur'] * 1e3:.4f} ms")
print(f"Derived dtime      : {results['dtime'] * 1e3:.4f} ms")
print(f"Figures saved to   : {results['figures_dir']}")
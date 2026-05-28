"""
farm — Minimal usage example.

Adjust the parameters below to match your acquisition, then run:
    python example.py
"""

import logging, sys

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
    tr=1.6,
    n_slices=54,
    mb_factor=3,
    trigger="R128",
    ch_regex=r"EXT|FLE",
    output_dir="FARM_output",
    plot=True,
)

results = run_pipeline(cfg)
print(f"\nCleaned data shape : {results['data_clean_full'].shape}")
print(f"Optimised sdur     : {results['sdur']*1e3:.4f} ms")
print(f"Optimised dtime    : {results['dtime']*1e3:.4f} ms")
"""Compact machine-readable sidecar for downstream tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def export_metadata(output_dir: str, basename: str, payload: dict) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{basename}_denoise.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return str(path)

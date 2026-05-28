"""Shared figure-saving helper used by all visualization modules.

Every plot function receives a ``fig_dir`` argument (a ``Path`` or ``None``).
If ``fig_dir`` is ``None``, the figure is silently discarded.
In all cases ``plt.show()`` is **never** called, so the pipeline cannot
block waiting for a GUI window.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no GUI windows
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger("farm.visualization")

# Global style applied once at import time
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def savefig(fig: plt.Figure, fig_dir: Path | None, name: str) -> Path | None:
    """Save *fig* as ``<fig_dir>/<name>.png`` then close it.

    Parameters
    ----------
    fig : matplotlib Figure.
    fig_dir : Path or None.  If *None* the figure is just closed.
    name : str — file stem (without extension).  Slashes are replaced
        by underscores so the caller can use e.g. ``"ch1/step3"``.

    Returns
    -------
    Path to the saved file, or *None*.
    """
    path = None
    if fig_dir is not None:
        safe = name.replace("/", "_").replace("\\", "_")
        path = Path(fig_dir) / f"{safe}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path))
        logger.debug("Figure saved: %s", path)
    plt.close(fig)
    return path
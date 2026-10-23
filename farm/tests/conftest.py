"""Keep library configuration and plotting caches inside temporary storage."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyfarm-test-mpl"))

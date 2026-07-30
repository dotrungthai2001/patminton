"""Make the package importable when running pytest from the repo root
without installing it (the compiled CUDA library is not needed for the
CPU tests; patas loads it lazily)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

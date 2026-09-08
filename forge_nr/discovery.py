"""Validate the complete local extension and bind its own protocol package."""
from pathlib import Path
import sys


def discover_root(extension):
    extension = Path(extension).resolve()
    if not (extension / "nr_shared/contract.py").is_file():
        raise RuntimeError("NR runtime package was not found. Install the complete extension")
    return extension


def bind_shared_root(root):
    """Never silently use a different checkout's already-imported protocol."""
    root = Path(root).resolve()
    loaded = sys.modules.get("nr_shared")
    if loaded is not None and Path(loaded.__file__).resolve().parent != root / "nr_shared":
        raise RuntimeError("Another extension's NR protocol is already imported; check the installation and restart Forge")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
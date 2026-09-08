"""Execution environment stored only in this extension's runtime settings."""
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "work"
DEFAULT_NR = dict(bridge="native/nr/bin/dlss5nr_bridge.dll", runtime_dir="models/dlssnr",
                  gpu_index=0, device="cuda:0", gpu_name="", channel_order="RGBA", idle_seconds=60.)
CONFIG_PATH = ROOT / "runtime-settings.json"


def get(*keys, default=None):
    config = {}
    if CONFIG_PATH.is_file():
        if CONFIG_PATH.stat().st_size > 16384:
            raise ValueError("Runtime settings file exceeds 16 KiB")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) != {"nr"}:
            raise ValueError("Only NR runtime settings are accepted")
    value = {"nr": {**DEFAULT_NR, **config.get("nr", {})}}
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def save(patch):
    if set(patch) != {"nr"} or set(patch["nr"]) - DEFAULT_NR.keys():
        raise ValueError("Invalid runtime setting fields")
    config = {"nr": {**get("nr"), **patch["nr"]}}
    descriptor, path = tempfile.mkstemp(prefix="runtime-", suffix=".tmp", dir=ROOT)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
        os.replace(path, CONFIG_PATH)
    finally:
        Path(path).unlink(missing_ok=True)
    return config
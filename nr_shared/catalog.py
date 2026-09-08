"""Configured NR runtime identity, independent of any browser's parameter state.

Only the private control process imports this module.
File checks are CPU-only; physical alternatives come from explicit inspection.
"""
from copy import deepcopy
from pathlib import Path
import threading

from .contract import SharedError, validate_runtime


class Catalog:
    def __init__(self, root, config_provider=None, runtime_provider=None):
        from nr_runtime.manager import RuntimeFiles
        from nr_runtime import settings
        self.root = Path(root).resolve()
        self.config_provider = config_provider or settings.get
        self.files = runtime_provider or RuntimeFiles()
        self._devices = {}
        self._lock = threading.Lock()

    def config(self):
        from nr_runtime.manager import NRConfig
        try:
            raw = self.config_provider("nr", default={})
            cfg = NRConfig.model_validate(raw).model_dump()
            for key in ("bridge", "runtime_dir"):
                path = Path(cfg[key])
                cfg[key] = str((path if path.is_absolute() else self.root / path).resolve())
            return cfg
        except (OSError, ValueError, TypeError):
            raise SharedError(503, "NR 环境配置不可用；没有切换显卡或运行库") from None

    def runtime(self, device=None):
        cfg = self.config()
        if device is not None and (device != cfg["device"] or not cfg["gpu_name"].strip()):
            with self._lock:
                found = deepcopy(self._devices.get(device))
            if found is None:
                raise SharedError(409, "先显式列出物理显卡，再选择另一张卡")
            cfg.update({key: found[key] for key in ("gpu_index", "device", "gpu_name")})
        result = self.files(cfg, force=False)
        return {key: result[key] for key in ("ready", "missing", "runtime_id", "runtime")}

    def inspection(self, runtime):
        if not isinstance(runtime, dict) or set(runtime) - {"bridge", "runtime_dir", "gpu_index", "device", "gpu_name", "channel_order", "runtime_id"}:
            raise SharedError(400, "Invalid device inspection request")
        cfg = self.config()
        for key in ("bridge", "runtime_dir"):
            if runtime.get(key) != cfg[key]:
                raise SharedError(409, "Device inspection only accepts the configured local bridge")
        return {key: cfg[key] for key in ("bridge", "runtime_dir", "gpu_index", "device", "gpu_name", "channel_order")}

    def check(self, runtime):
        value = validate_runtime(runtime)
        cfg = self.config()
        for key in ("bridge", "runtime_dir", "channel_order"):
            if value[key] != cfg[key]:
                raise SharedError(409, "NR 只允许已配置的运行库/通道；本次环境快照已过期")
        # Do not infer a device from its index/name. Native inspection will verify
        # all three fields against a physical LUID before actual processing.
        cfg.update({key: value[key] for key in ("gpu_index", "device", "gpu_name")})
        actual = self.files(cfg, force=True)
        if not actual["ready"]:
            raise SharedError(503, "NR 环境未就绪：" + "; ".join(actual["missing"]))
        if actual["runtime"] != value:
            raise SharedError(409, "NR 运行库或显卡快照已过期，未执行增强")
        return value

    def remember(self, inspection):
        rows = inspection.get("devices", [])
        if not isinstance(rows, list) or len(rows) > 64:
            raise SharedError(502, "NR 物理设备枚举无效")
        devices = {}
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("device"), str)
                    or not row.get("luid") or not row.get("gpu_name") or type(row.get("gpu_index")) is not int):
                continue
            if row["device"] in devices:
                raise SharedError(409, "NR 物理设备映射不唯一")
            devices[row["device"]] = deepcopy(row)
        with self._lock:
            self._devices = devices
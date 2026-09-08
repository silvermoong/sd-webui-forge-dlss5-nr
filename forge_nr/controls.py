"""Local named parameter presets. No mutable current settings or environment store."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading

from nr_shared.contract import STAGES, validate_params


class Presets:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.path = self.directory / "named-presets.json"
        self.lock = threading.RLock()

    @staticmethod
    def _name(name):
        if (not isinstance(name, str) or not name.strip() or len(name) > 80
                or any(ord(c) < 32 or c in '/\\' for c in name)):
            raise ValueError("预设名称须为1–80字，不能包含路径分隔符/控制字符")
        return name.strip()

    def _read(self):
        if not self.path.exists():
            return {}
        if self.path.stat().st_size > 1024 * 1024:
            raise ValueError("预设文件超过1 MiB")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or len(data) > 256:
            raise ValueError("预设文件格式错误或条数超过256")
        for name, record in data.items():
            self._name(name)
            if not isinstance(record, dict) or set(record) != {"stage", "params"} or record["stage"] not in STAGES:
                raise ValueError("预设仅可存插入时机与NR参数")
            record["params"] = validate_params(record["params"])
        return data

    def _write(self, records):
        if len(records) > 256:
            raise ValueError("最多保存256条预设")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix="presets-", suffix=".tmp", delete=False) as file:
                path = Path(file.name)
                json.dump(records, file, ensure_ascii=False, allow_nan=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(path, self.path)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def names(self):
        with self.lock:
            return sorted(self._read())

    def snapshot(self):
        """Freeze one complete file read for an XYZ run, independent of later saves."""
        with self.lock:
            return deepcopy(self._read())

    def load(self, name):
        with self.lock:
            data = self._read()
            name = self._name(name)
            if name not in data:
                raise ValueError("没有该预设")
            return deepcopy(data[name])

    def save(self, name, stage, params):
        if stage not in STAGES:
            raise ValueError("未知NR插入时机")
        record = dict(stage=stage, params=validate_params(params))
        with self.lock:
            data = self._read()
            data[self._name(name)] = record
            self._write(data)

    def delete(self, name):
        with self.lock:
            data = self._read()
            name = self._name(name)
            if name not in data:
                raise ValueError("没有该预设")
            del data[name]
            self._write(data)
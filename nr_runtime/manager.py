"""Isolated runtime manager; no Forge imports, web server, GPU arbiter or UI state."""
import ctypes
import hashlib
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from . import procs, settings
from .nr_assets import NRError, NREnvironmentError
from .nr_native import NativeError


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)

class NRParams(StrictRequest):
    style: int = Field(default=1, ge=0, le=2)
    preset: int = Field(default=3, ge=0, le=3)
    intensity: float = Field(default=1., ge=0, le=2)
    tone: float = Field(default=1., ge=0, le=2)
    structure: float = Field(default=1., ge=0, le=2)
    skin: float = Field(default=-1., ge=-1, le=2)
    auto_mask: bool = False
    mix: float = Field(default=1., ge=0, le=1)
    flow: bool = True

class NRConfig(StrictRequest):
    bridge: str = Field(default="native/nr/bin/dlss5nr_bridge.dll", max_length=4096)
    runtime_dir: str = Field(default="models/dlssnr", max_length=4096)
    gpu_index: int = Field(default=0, ge=0, le=63)
    device: Annotated[str, Field(pattern=r"^cuda:(?:[0-9]|[1-5][0-9]|6[0-3])$")] = "cuda:0"
    gpu_name: str = Field(default="", max_length=256)
    channel_order: Literal["RGBA", "BGRA"] = "RGBA"
    idle_seconds: float = Field(default=60., ge=0, le=3600)

    @field_validator("bridge", "runtime_dir")
    @classmethod
    def local_environment_path(cls, value):
        if "\0" in value or value.startswith(("\\\\", "//")) or "://" in value:
            raise ValueError("运行库必须是本机文件路径")
        return value

class RuntimeFiles:
    """File-only readiness and fingerprint. Force=True bypasses all hash memoization."""
    def __init__(self):
        self._hashes = OrderedDict()
        self._exports = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _pe64(path):
        """Bounded DOS/COFF/optional-header inspection; never call LoadLibrary."""
        try:
            with path.open("rb") as stream:
                head = stream.read(64)
                if len(head) != 64 or head[:2] != b"MZ":
                    return False
                offset = struct.unpack_from("<I", head, 60)[0]
                if offset > 1024 * 1024:
                    return False
                stream.seek(offset)
                pe = stream.read(26)
            return (len(pe) == 26 and pe[:4] == b"PE\0\0"
                    and struct.unpack_from("<H", pe, 4)[0] == 0x8664
                    and struct.unpack_from("<H", pe, 22)[0] & 0x2000 != 0
                    and struct.unpack_from("<H", pe, 24)[0] == 0x20b)
        except (OSError, struct.error):
            return False

    def _bridge_exports(self, path, sha256):
        with self._lock:
            cached = self._exports.get(sha256)
        if cached is not None:
            return cached
        if path.stat().st_size > 16 * 1024 * 1024:
            return frozenset()
        from .nr_native import pe_exports
        exports = frozenset(pe_exports(path))
        with self._lock:
            self._exports[sha256] = exports
            self._exports.move_to_end(sha256)
            while len(self._exports) > 32:
                self._exports.popitem(last=False)
        return exports

    def _hash(self, path, force):
        before = path.stat()
        key = (str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
        with self._lock:
            cached = None if force else self._hashes.get(key)
        if cached is not None:
            return cached
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise NRError(409, "运行库在检查期间变化，请重试")
        with self._lock:
            self._hashes[key] = value
            self._hashes.move_to_end(key)
            while len(self._hashes) > 32:
                self._hashes.popitem(last=False)
        return value

    def __call__(self, cfg, *, force=False):
        runtime = {k: cfg[k] for k in ("bridge", "runtime_dir", "gpu_index", "device", "gpu_name", "channel_order")}
        for key in ("bridge", "runtime_dir"):
            path = Path(runtime[key])
            runtime[key] = str((path if path.is_absolute() else Path(settings.ROOT) / path).resolve())
        directory, bridge = Path(runtime["runtime_dir"]), Path(runtime["bridge"])
        files = {"bridge": bridge, "runtime": directory / "nvngx_dlssnr.dll",
                 "caller": directory / "caller" / "nvngx.dll_comfy.dll"}
        hashes, missing = {}, []
        for name, path in files.items():
            try:
                if not path.is_file() or path.stat().st_size == 0:
                    raise OSError()
                hashes[name] = self._hash(path, force)
                if not self._pe64(path):
                    missing.append(f"NR {name} 不是有效的 64 位 Windows DLL")
                if name == "bridge":
                    required = {"dlss5nr_init", "dlss5nr_process", "dlss5nr_shutdown", "dlss5nr_gpu_name",
                                "tagsystem_nr_devices_json", "tagsystem_nr_gpu_luid"}
                    if not required.issubset(self._bridge_exports(path, hashes[name])):
                        missing.append("NR bridge 缺少处理/物理设备核对接口，需要项目构建的 MIT bridge")
            except OSError:
                hashes[name] = None
                missing.append({"bridge": "缺少可用的 NR bridge",
                    "runtime": "缺少用户自行提供并获授权的 NVIDIA nvngx_dlssnr.dll",
                    "caller": "缺少 NR caller/nvngx.dll_comfy.dll"}[name])
        manifest = bridge.parent / "build.json"
        if manifest.is_file():
            try:
                if manifest.stat().st_size > 128 * 1024:
                    raise OSError()
                hashes["build"] = self._hash(manifest, force)
            except OSError:
                missing.append("NR bridge 构建清单无法读取")
        if sys.platform != "win32" or sys.maxsize <= 2 ** 32:
            missing.append("NR 需要 64 位 Windows")
        if not runtime["gpu_name"].strip():
            missing.append("尚未确认 DXGI/CUDA 物理卡映射；先列出设备并选择")
        payload = json.dumps({"runtime": runtime, "files": hashes}, sort_keys=True,
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        runtime_id = hashlib.sha256(payload).hexdigest()
        return {"ready": not missing, "missing": missing, "runtime_id": runtime_id,
                "runtime": {**runtime, "runtime_id": runtime_id}, "hashes": hashes}

@contextmanager
def _environment(cancel_event=None):
    """Use only around environment operations, never media decode/publication."""
    try:
        yield
    except Exception as error:
        if cancel_event is not None and cancel_event.is_set():
            raise NRError(409, "NR 已取消") from None
        if isinstance(error, NREnvironmentError):
            raise
        message = str(error) if isinstance(error, (NRError, NativeError)) else f"NR 环境检查失败（{type(error).__name__}）"
        raise NREnvironmentError(getattr(error, "status", 503), message) from error

def verify_device(runtime, result):
    """Expected DXGI index, CUDA ordinal and opaque adapter name must all agree."""
    rows = result.get("devices") if isinstance(result, dict) else None
    if not isinstance(rows, list) or len(rows) > 64:
        raise NREnvironmentError(503, "无法核对 NR 物理设备映射")
    matches = [row for row in rows if isinstance(row, dict)
               and type(row.get("gpu_index")) is int and row["gpu_index"] == runtime["gpu_index"]]
    if len(matches) != 1:
        raise NREnvironmentError(409, "NR DXGI 物理设备已变化，请重新选择设备")
    row = matches[0]
    if (not runtime["gpu_name"] or row.get("gpu_name") != runtime["gpu_name"]
            or row.get("device") != runtime["device"] or not row.get("luid")):
        raise NREnvironmentError(409, "NR 的 DXGI/CUDA 物理卡映射不符，已拒绝处理")
    return row

class CancelEvent(threading.Event):
    """A normal Event plus the formal Progress cancellation predicate."""
    def __init__(self, predicate=lambda: False):
        super().__init__()
        self.predicate = predicate

    def is_set(self):
        return super().is_set() or self.predicate()

    def wait(self, timeout=None):
        end = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = .05 if end is None else min(.05, end - time.monotonic())
            if remaining <= 0:
                break
            super().wait(remaining)
        return self.is_set()

class ManagedWorker:
    """One warm child, serialized commands, independent control/status channels.

    Only the IO mutex spans a run. The state mutex never spans pipes, process
    startup, hashing or waits. Late cancellation is matched to a command id.
    """
    def __init__(self, *, log_dir, spawn=None, command=None, idle_seconds=lambda: 60.,
                 idle_timeout=120., run_timeout=21_600., cancel_timeout=3.):
        self.log_dir = Path(log_dir)
        self._spawn = spawn or procs.spawn
        self._owned_job = spawn is None
        self._command = command or [sys.executable, "-u", "-B", "-m", "nr_runtime.nr_worker"]
        self.idle_seconds = idle_seconds
        self.idle_timeout, self.run_timeout, self.cancel_timeout = idle_timeout, run_timeout, cancel_timeout
        self._state_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._life_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._closed = threading.Event()
        self._proc = None
        self._queue = None
        self._active = ""
        self._cancel = None
        self._runtime_id = ""
        self._runtime = {}
        self._retire = None
        self._gpu_name = ""
        self._last_error = ""
        self._idle_at = time.monotonic()
        self._readers = []
        self._reaper = None

    def status(self):
        with self._state_lock:
            process = self._proc
            running = process is not None and process.poll() is None
            return {"running": running, "busy": bool(self._active),
                    "pid": process.pid if running else None,
                    "gpu_name": self._gpu_name if running else "", "last_error": self._last_error,
                    "runtime_id": self._runtime_id if running else "",
                    "device": self._runtime.get("device") if running else None,
                    "runtime": deepcopy(self._runtime) if running else {}}

    def invalidate(self, runtime_id):
        """Nonblocking status/config check: retire only the old process generation."""
        with self._state_lock:
            if self._proc is not None and self._runtime_id != runtime_id:
                self._retire = self._proc
                if self._cancel is not None:
                    self._cancel.set()

    def _log(self, message):
        try:
            with self._log_lock:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                path = self.log_dir / "worker.log"
                if path.exists() and path.stat().st_size >= 1024 * 1024:
                    path.replace(self.log_dir / "worker.log.1")
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(str(message).rstrip()[:2048] + "\n")
        except OSError:
            pass                         # Logging failure must not wedge pipe readers.

    @staticmethod
    def _offer(out, value):
        try:
            out.put_nowait(value)
        except queue.Full:
            # Progress is advisory. A terminal record must not be lost to a flood.
            if isinstance(value, dict) and value.get("type") == "progress":
                return
            try:
                out.get_nowait()
            except queue.Empty:
                pass
            out.put_nowait(value)

    def _reader(self, stream, out, stderr=False):
        noise = 0
        try:
            while line := stream.readline(1024 * 1024 + 1):
                if stderr:
                    self._log(line)
                    continue
                if len(line) > 1024 * 1024:
                    self._offer(out, {"type": "fatal"})
                    return
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError()
                except ValueError:
                    self._log(line)
                    noise += 1
                    if noise > 64:
                        self._offer(out, {"type": "fatal"})
                        return
                    continue
                self._offer(out, value)
        except (OSError, ValueError):
            pass
        finally:
            if not stderr:
                self._offer(out, None)

    def _writer(self, process, out, inbox, stopped):
        # Pipes can block even on write: a wedged child might never read stdin.
        # The command owner only enqueues; its timeout loop remains runnable.
        try:
            while not stopped.is_set():
                try:
                    text = inbox.get(timeout=.1)
                except queue.Empty:
                    continue
                process.stdin.write(text)
                process.stdin.flush()
        except (OSError, ValueError):
            self._offer(out, None)

    @staticmethod
    def _adopted(process):
        if sys.platform != "win32":
            return True
        # spawn() already called adopt(). Verify membership rather than adopting
        # twice (AssignProcessToJobObject may reject an existing membership).
        handle = procs.job()
        if not handle:
            return False
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_int)]
        kernel.IsProcessInJob.restype = ctypes.c_int
        inside = ctypes.c_int()
        return bool(kernel.IsProcessInJob(int(process._handle), handle, ctypes.byref(inside)) and inside.value)

    def _start(self, runtime):
        runtime = deepcopy(runtime)
        with self._life_lock:
            if self._closed.is_set():
                raise NREnvironmentError(503, "NR worker 已关闭")
            with self._state_lock:
                previous, old_runtime = self._proc, self._runtime
            identity = str(runtime.get("runtime_id") or "")
            if previous is not None and previous.poll() is None and old_runtime == runtime:
                return previous, self._queue
            if previous is not None:
                self._kill(previous)
                if previous.poll() is None:
                    raise NREnvironmentError(503, "旧 NR worker 尚未退出，不能切换运行库或物理设备")
            try:
                env = procs.launch_env()
                env.pop("CUDA_VISIBLE_DEVICES", None)
                env["CUDA_DEVICE_ORDER"] = "FASTEST_FIRST"
                process = self._spawn(self._command, cwd=settings.ROOT,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=env)
            except OSError:
                raise NREnvironmentError(503, "无法启动独立 NR worker") from None
            if self._owned_job and not self._adopted(process):
                self._kill(process)
                raise NREnvironmentError(503, "NR 子进程未绑定父进程生命周期，已停止")
            if self._closed.is_set():
                self._kill(process)
                raise NREnvironmentError(503, "NR worker 已关闭")
            out, inbox, stopped = queue.Queue(maxsize=64), queue.Queue(maxsize=8), threading.Event()
            process._nr_inbox, process._nr_stopped = inbox, stopped
            with self._state_lock:
                self._proc, self._queue = process, out
                self._retire = None
                self._runtime_id, self._gpu_name = identity, str(runtime.get("gpu_name") or "")
                self._runtime = runtime
                self._idle_at = time.monotonic()
            self._readers = [t for t in self._readers if t.is_alive()]
            for stream, stderr in ((process.stdout, False), (process.stderr, True)):
                thread = threading.Thread(target=self._reader, args=(stream, out, stderr),
                                          daemon=True, name="nr-stderr" if stderr else "nr-stdout")
                self._readers.append(thread)
                thread.start()
            writer = threading.Thread(target=self._writer, args=(process, out, inbox, stopped),
                                      daemon=True, name="nr-stdin")
            self._readers.append(writer)
            writer.start()
            if self._reaper is None:
                self._reaper = threading.Thread(target=self._reap, daemon=True, name="nr-idle")
                self._reaper.start()
            return process, out

    def _send(self, process, message):
        try:
            text = json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n"
            if len(text) > 1024 * 1024 or process.poll() is not None:
                return False
            process._nr_inbox.put_nowait(text)
            return True
        except (OSError, ValueError, AttributeError, queue.Full):
            return False

    def cancel(self, ref):
        with self._state_lock:
            if not ref or self._active != str(ref) or self._proc is None:
                return False
            process = self._proc
            self._cancel.set()
        return self._send(process, {"op": "cancel", "id": str(ref)})

    def run(self, command, progress, cancel_event):
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        started = time.monotonic()
        deadline = started + (min(20., self.run_timeout) if command.get("op") == "inspect" else self.run_timeout)
        while not self._io_lock.acquire(timeout=.05):
            if cancel_event.is_set() or self._closed.is_set():
                raise NRError(409, "NR 已取消")
            if time.monotonic() >= deadline:
                raise NREnvironmentError(504, "等待 NR worker 超时")
        process, healthy = None, False
        try:
            if cancel_event.is_set() or self._closed.is_set():
                raise NRError(409, "NR 已取消或关闭")
            with _environment(cancel_event):
                process, out = self._start(command.get("runtime") or {})
            ident = str(command["id"])
            with self._state_lock:
                self._active, self._cancel = ident, cancel_event
            if not self._send(process, command):
                raise NREnvironmentError(502, "无法发送 NR worker 命令")
            last_progress, cancel_at, stray = time.monotonic(), None, 0
            while True:
                now = time.monotonic()
                if cancel_event.is_set() or self._closed.is_set():
                    if cancel_at is None:
                        cancel_at = now
                        self._send(process, {"op": "cancel", "id": ident})
                    if now - cancel_at >= self.cancel_timeout:
                        raise NRError(409, "NR 取消超时，已停止本 worker")
                if now >= deadline or now - last_progress >= self.idle_timeout:
                    raise NREnvironmentError(504, "NR worker 长时间无进度或超过任务时限，已停止")
                try:
                    message = out.get(timeout=.05)
                except queue.Empty:
                    continue
                if message is None:
                    raise NREnvironmentError(502, "NR worker 已退出，未交回完整结果")
                if message.get("type") == "fatal":
                    raise NREnvironmentError(502, "NR worker 输出不符合 JSON-lines 协议")
                if message.get("id") != ident:
                    stray += 1
                    if stray > 128:
                        raise NREnvironmentError(502, "NR worker 返回了过多无关票据")
                    continue
                kind = message.get("type")
                if kind == "progress":
                    last_progress = now
                    ratio = message.get("ratio")
                    if type(ratio) not in (int, float) or not 0 <= ratio <= 1:
                        ratio = None
                    progress(str(message.get("message") or "")[:400], ratio)
                elif kind == "error":
                    self._log(message.get("error"))
                    if cancel_event.is_set() or message.get("canceled") is True:
                        healthy = True
                        raise NRError(409, "NR 已取消")
                    if message.get("category") == "environment" or command.get("op") == "inspect":
                        raise NREnvironmentError(502, "NR 执行环境失败；详细诊断已写入本地 worker 日志")
                    raise NRError(502, "NR 原生处理失败；详细诊断已写入本地 worker 日志")
                elif kind == "result":
                    if not isinstance(message.get("result"), dict):
                        raise NREnvironmentError(502, "NR worker 结果格式无效")
                    healthy = True
                    if cancel_event.is_set() or self._closed.is_set():
                        raise NRError(409, "NR 已取消，结果未采纳")
                    with self._state_lock:
                        self._last_error = ""
                    return message["result"]
                else:
                    raise NREnvironmentError(502, "NR worker 返回未知消息类型")
        except BaseException as error:
            canceled = isinstance(error, NREnvironmentError) and (cancel_event.is_set() or self._closed.is_set())
            with self._state_lock:
                self._last_error = "NR 已取消" if canceled else str(error) if isinstance(error, NRError) else "NR worker 调度/回调失败，已停止"
            if process is not None and not healthy:
                try:
                    self.stop(cancel=False)
                except NRError as stopping:
                    stopping.execution_unknown = process.poll() is None
                    raise
            if canceled:
                raise NRError(409, "NR 已取消") from None
            raise
        finally:
            with self._state_lock:
                self._active, self._cancel, self._idle_at = "", None, time.monotonic()
            self._io_lock.release()

    def inspect(self, runtime, *, cancel_event=None):
        return self.run({"op": "inspect", "id": uuid.uuid4().hex, "runtime": runtime},
                        lambda *_: None, cancel_event if cancel_event is not None else threading.Event())

    def _kill(self, process):
        if process.poll() is None:
            if self._send(process, {"op": "shutdown"}):
                try:
                    process.wait(timeout=.15)
                except subprocess.TimeoutExpired:
                    pass
            if sys.platform == "win32" and process.poll() is None:
                # Exact owned Popen PID only; never ports, global queues or names.
                exe = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "taskkill.exe"
                try:
                    subprocess.run([str(exe), "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True, timeout=5,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                except (OSError, subprocess.SubprocessError):
                    pass
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if process.poll() is None:
            return False                # Keep its physical resource record, and do not block on live pipes.
        stopped = getattr(process, "_nr_stopped", None)
        if stopped is not None:
            stopped.set()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        return True

    def stop(self, expected_process=None, *, cancel=True):
        with self._life_lock:
            with self._state_lock:
                if expected_process is not None and self._proc is not expected_process:
                    return False
                process = self._proc
                self._retire = process
                if cancel and self._cancel is not None:
                    self._cancel.set()
            if process is not None:
                self._kill(process)
                if process.poll() is None:
                    raise NREnvironmentError(503, "NR worker 尚未退出，仍保留实际设备占用")
                with self._state_lock:
                    self._proc, self._queue, self._retire = None, None, None
                    self._runtime, self._runtime_id, self._gpu_name = {}, "", ""
            return process is not None

    def _reap(self):
        while not self._closed.wait(.25):
            try:
                idle_seconds = self.idle_seconds()
            except (NRError, TypeError, ValueError):
                idle_seconds = 0.       # Invalid environment must not pin a warm DLL indefinitely.
            with self._state_lock:
                process = self._proc
                expired = (self._proc is not None and not self._active
                           and (self._retire is self._proc
                                or time.monotonic() - self._idle_at >= idle_seconds))
            if expired and self._io_lock.acquire(blocking=False):
                try:
                    self.stop(expected_process=process)
                except NRError as error:
                    self._log(error)
                finally:
                    self._io_lock.release()

    def close(self):
        self._closed.set()
        self.stop()
        if self._reaper is not None and self._reaper is not threading.current_thread():
            self._reaper.join(1)
        for thread in self._readers:
            if thread is not threading.current_thread():
                thread.join(.2)

"""Bounded private stdio JSON RPC. Safe to import in Forge (stdlib only)."""
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import uuid

from .contract import SharedError

MAX_BODY = 1024 * 1024
RPC_TIMEOUT = 10.


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("Nonfinite JSON number")


def decode(data):
    if len(data) > MAX_BODY:
        raise SharedError(502, "NR response exceeds 1 MiB")
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise SharedError(502, "Invalid NR JSON response") from None
    if not isinstance(value, dict):
        raise SharedError(502, "NR RPC requires a JSON object")
    return value


def encode(value):
    try:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise SharedError(400, "Invalid NR JSON request") from None
    if len(data) + 1 > MAX_BODY:
        raise SharedError(413, "NR request exceeds 1 MiB")
    return data


def _timeout(value):
    value = RPC_TIMEOUT if value is None else float(value)
    if not math.isfinite(value) or value <= 0:
        raise SharedError(504, "NR RPC deadline expired")
    return min(value, RPC_TIMEOUT)


def _unpack(envelope):
    if set(envelope) == {"result"}:
        return envelope["result"]
    if (set(envelope) == {"error", "status"} and isinstance(envelope["error"], str)
            and type(envelope["status"]) is int and 400 <= envelope["status"] <= 599):
        raise SharedError(envelope["status"], envelope["error"])
    raise SharedError(502, "Invalid NR response envelope")


def private_env():
    env = dict(os.environ)
    # Forge masks/reorders CUDA; the selected physical runtime snapshot must prevail.
    for key in tuple(env):
        if key.upper() in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES",
                           "NODEFAULTCURRENTDIRECTORYINEXEPATH"):
            env.pop(key, None)
    root = env.get("SystemRoot", r"C:\Windows")
    if not env.get("COMSPEC"):
        env["COMSPEC"] = str(Path(root) / "System32/cmd.exe")
    if ".EXE" not in env.get("PATHEXT", "").upper().split(";"):
        env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC"
    return env


class _WindowsJob:
    """An unnamed, non-inheritable owner handle. Closing it kills the complete tree."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w

        class Basic(c.Structure):
            _fields_ = [("process_time", c.c_longlong), ("job_time", c.c_longlong),
                        ("flags", w.DWORD), ("min_ws", c.c_size_t), ("max_ws", c.c_size_t),
                        ("active", w.DWORD), ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class IO(c.Structure):
            _fields_ = [(name, c.c_ulonglong) for name in ("ro", "wo", "oo", "rt", "wt", "ot")]

        class Extended(c.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", c.c_size_t),
                        ("job_memory", c.c_size_t), ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]

        self.api = c.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.api.SetInformationJobObject.restype = w.BOOL
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.AssignProcessToJobObject.restype = w.BOOL
        self.api.CloseHandle.argtypes = [w.HANDLE]
        self.api.CloseHandle.restype = w.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise SharedError(503, "Cannot create NR private process owner")
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            self.close()
            raise SharedError(503, "Cannot configure NR private process owner")

    def adopt(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise SharedError(503, "Cannot adopt NR private helper; no work was submitted")

    @staticmethod
    def resume(process):
        import ctypes as c
        from ctypes import wintypes as w
        # Popen closes the primary thread handle. Resume the suspended process by its
        # owned process handle, rather than discovering/opening threads by global IDs.
        native = c.WinDLL("ntdll")
        native.NtResumeProcess.argtypes = [w.HANDLE]
        native.NtResumeProcess.restype = c.c_long
        if native.NtResumeProcess(int(process._handle)) < 0:
            raise SharedError(503, "Cannot resume adopted NR private helper")

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class PrivateTransport:
    """One owned stdio helper; command injection is a Python-only isolated test seam."""
    mode = "private"
    instance = None

    def __init__(self, root, *, command=None):
        self.root = Path(root).resolve()
        self._lock = threading.Lock()
        self._life = threading.Lock()
        self._closed = threading.Event()
        self._requests = queue.Queue(maxsize=1)
        self._responses = queue.Queue(maxsize=2)
        self._broken = threading.Event()
        self._threads = []
        self._process = None
        self._job = _WindowsJob() if os.name == "nt" else None
        argv = command if command is not None else [str(self.root / ".venv/Scripts/python.exe"), "-u", "-B", "-m", "nr_shared.private"]
        try:
            self._process = subprocess.Popen(argv, cwd=str(self.root), env=private_env(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                close_fds=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | (0x4 if os.name == "nt" else 0),
                start_new_session=os.name != "nt")
            if self._job is not None:
                # CREATE_SUSPENDED prevents venvlauncher spawning the real interpreter
                # outside our Job in the Popen -> adopt window. Descendants inherit it.
                self._job.adopt(self._process)
                self._job.resume(self._process)
            # Adoption precedes ALL input, including hello and join.
            for name, target in (("reader", self._read), ("writer", self._write), ("stderr", self._drain)):
                thread = threading.Thread(target=target, name="nr-private-" + name, daemon=True)
                self._threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise

    def _offer(self, value):
        try:
            self._responses.put_nowait(value)
        except queue.Full:
            self._broken.set()  # Protocol flood: bounded memory and fail closed.

    def _read(self):
        try:
            # FileIO.readline reads byte-by-byte; buffered read is necessary for 1 MiB RPCs.
            import io
            with io.BufferedReader(self._process.stdout) as stream:
                while not self._closed.is_set():
                    line = stream.readline(MAX_BODY + 1)
                    if not line:
                        raise SharedError(503, "NR private helper exited")
                    if not line.endswith(b"\n"):
                        raise SharedError(502, "Invalid NR private response framing")
                    self._offer(decode(line))
        except (SharedError, OSError, ValueError):
            self._offer(SharedError(503, "NR private helper failed or closed its protocol pipe"))

    def _write(self):
        try:
            while not self._closed.is_set():
                try:
                    payload, deadline = self._requests.get(timeout=.1)
                except queue.Empty:
                    continue
                if time.monotonic() >= deadline:
                    raise SharedError(504, "NR private send deadline expired")
                data = memoryview(payload)
                while data:
                    written = self._process.stdin.write(data)
                    if not written:
                        raise OSError("closed pipe")
                    data = data[written:]
                self._process.stdin.flush()
        except (SharedError, OSError, ValueError):
            self._offer(SharedError(503, "NR private helper input failed; request is not retried"))
        finally:
            self._process.stdin.close()

    def _drain(self):
        # Native output never shares the protocol and is never echoed (could contain paths/tokens).
        try:
            while self._process.stderr.read(8192):
                pass
        except (OSError, ValueError):
            pass
        finally:
            self._process.stderr.close()

    def rpc(self, body, timeout=None):
        deadline = time.monotonic() + _timeout(timeout)
        ident = uuid.uuid4().hex
        payload = encode({"id": ident, "body": body}) + b"\n"
        if not self._lock.acquire(timeout=max(0., deadline - time.monotonic())):
            raise SharedError(504, "NR private RPC queue deadline expired; request was not sent")
        try:
            if self._closed.is_set() or self._broken.is_set():
                self.close()
                raise SharedError(503, "NR private transport is closed")
            try:
                self._requests.put((payload, deadline), timeout=max(0., deadline - time.monotonic()))
                response = self._responses.get(timeout=max(0., deadline - time.monotonic()))
            except (queue.Empty, queue.Full):
                # In-flight execution is unknown. Never reuse this stream or resubmit.
                self.close()
                raise SharedError(504, "NR private RPC timed out; owned process was stopped") from None
            if isinstance(response, SharedError):
                self.close()
                raise response
            if self._broken.is_set() or response.get("id") != ident:
                self.close()
                raise SharedError(502, "NR private response id mismatch")
            envelope = {key: value for key, value in response.items() if key != "id"}
            try:
                return _unpack(envelope)
            except SharedError as error:
                if set(envelope) != {"error", "status"}:
                    self.close()
                raise error
        finally:
            self._lock.release()

    def close(self):
        with self._life:
            self._closed.set()
            process = self._process
            if self._job is not None:
                self._job.close()
            elif process is not None and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()  # Also covers failed Job adoption (no commands sent).
                    process.wait(timeout=.5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            for thread in self._threads:
                if thread is not threading.current_thread():
                    thread.join(.05)
            # Each IO thread owns its stream. Closing a pipe while another thread
            # holds its read/write lock can block indefinitely (including .close()).
            # Kill the tree first; live daemon IO threads close their own handles.
            if process is not None and process.poll() is not None and not any(t.is_alive() for t in self._threads):
                for stream in (process.stdin, process.stdout, process.stderr):
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
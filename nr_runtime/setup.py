"""Prepare the NR runtime in Forge's model directory without loading a model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading

from .download import DownloadError, install_runtime, read_source


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(path):
    powershell = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    script = Path(__file__).resolve().parents[1] / "tools/verify_runtime.ps1"
    environment = dict(os.environ)
    environment["PSModulePath"] = str(powershell.parent / "Modules")
    result = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive",
                             "-File", str(script), str(path)], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=30, env=environment,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise ValueError("Could not verify the runtime's Authenticode signature; no DLL was loaded")
    value = json.loads(result.stdout.decode("utf-8-sig"))
    if value.get("status") != "Valid" or "NVIDIA Corporation" not in value.get("subject", ""):
        raise ValueError("A valid NVIDIA Authenticode signature is required; modified/unsigned DLLs are not accepted")
    return value


class RuntimeSetup:
    def __init__(self, root, verifier=signature, models_dir=None, bootstrapper=None):
        self.root = Path(root).resolve()
        self.verifier = verifier
        self.base_runtime_dir = ((Path(models_dir) / "DLSS-NR") if models_dir is not None else self.root / "models/dlssnr").resolve()
        self.explicit_models_dir = models_dir is not None
        self.bootstrapper = bootstrapper
        self._preparing = threading.Lock()
        self.selection_path = self.root / "runtime-selection.json"
        self.mode = self._read_mode()

    def _read_mode(self):
        if not self.selection_path.exists():
            return "auto"
        if self.selection_path.stat().st_size > 4096:
            raise DownloadError("mode_invalid", "Runtime selection file is too large")
        value = json.loads(self.selection_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"mode"} or value["mode"] not in ("auto", "manual"):
            raise DownloadError("mode_invalid", "Invalid runtime selection")
        return value["mode"]

    @property
    def runtime_dir(self):
        return self.base_runtime_dir / "manual" if self.mode == "manual" else self.base_runtime_dir

    def set_mode(self, mode):
        if mode not in ("auto", "manual"):
            raise DownloadError("mode_invalid", "Select automatic or manual runtime mode")
        if not self._preparing.acquire(blocking=False):
            raise DownloadError("preparing", "Runtime preparation is already running")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix="selection-", suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump({"mode": mode}, stream)
                os.replace(temporary, self.selection_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.mode = mode
            self.configure()
        finally:
            self._preparing.release()

    @staticmethod
    def _manual_file(path):
        if (not path.is_file() or path.resolve() != path or path.stat().st_nlink != 1
                or not 1024 <= path.stat().st_size <= 300 * 1024 ** 2):
            raise DownloadError("manual_invalid", "Select a local, non-linked Windows x64 runtime DLL")
        with path.open("rb") as stream:
            header = stream.read(64)
            if len(header) != 64 or header[:2] != b"MZ":
                raise DownloadError("manual_invalid", "The manual file is not a Windows DLL")
            offset = struct.unpack_from("<I", header, 60)[0]
            if offset > 1024 * 1024:
                raise DownloadError("manual_invalid", "The manual DLL header is invalid")
            stream.seek(offset)
            image = stream.read(26)
        if (len(image) != 26 or image[:4] != b"PE\0\0" or struct.unpack_from("<H", image, 4)[0] != 0x8664
                or not struct.unpack_from("<H", image, 22)[0] & 0x2000
                or struct.unpack_from("<H", image, 24)[0] != 0x20b):
            raise DownloadError("manual_invalid", "The manual runtime must be a Windows x64 DLL")

    def configure(self):
        from . import settings
        if self.explicit_models_dir and settings.get("nr")["runtime_dir"] != str(self.runtime_dir):
            settings.save({"nr": {"runtime_dir": str(self.runtime_dir)}})

    def _bootstrap(self):
        if self.bootstrapper is not None:
            self.bootstrapper()
            return
        spec = importlib.util.spec_from_file_location("forge_nr_installer", self.root / "install.py")
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        installer.main()

    def _caller(self):
        relative = "models/dlssnr/caller/nvngx.dll_comfy.dll"
        source = self.root / relative
        metadata = json.loads((self.root / "native-artifacts.json").read_text(encoding="utf-8"))
        expected = metadata["files"][relative]
        if not source.is_file() or digest(source) != expected:
            raise DownloadError("dependencies_missing", "The MIT caller helper is missing or damaged")
        target = self.runtime_dir / "caller/nvngx.dll_comfy.dll"
        if target.is_file():
            if digest(target) != expected:
                raise ValueError("An unrecognized caller helper was preserved; use a clean runtime directory")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.resolve() != target.parent:
            raise ValueError("Runtime destination must not be a directory link")
        descriptor, temporary = tempfile.mkstemp(prefix="caller-", suffix=".tmp", dir=target.parent)
        os.close(descriptor)
        try:
            shutil.copyfile(source, temporary)
            if digest(temporary) != expected:
                raise ValueError("MIT caller copy integrity check failed")
            os.rename(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def ensure(self, progress=None, repair=False):
        if not self._preparing.acquire(blocking=False):
            raise DownloadError("preparing", "Runtime preparation is already running")
        try:
            self.configure()
            if repair:
                try:
                    self._bootstrap()
                except Exception as error:
                    raise DownloadError("dependencies_failed", "Dependency repair did not finish") from error
            required = (".venv/Scripts/python.exe", "native/nr/bin/dlss5nr_bridge.dll",
                        "models/dlssnr/caller/nvngx.dll_comfy.dll", "native-artifacts.json")
            if not all((self.root / relative).is_file() for relative in required):
                raise DownloadError("dependencies_missing", "Required dependencies are missing; repair dependencies explicitly")
            self._caller()
            target = self.runtime_dir / "nvngx_dlssnr.dll"
            if self.mode == "manual":
                if not target.exists():
                    raise DownloadError("manual_missing", "Manual mode requires a user-supplied runtime; no download was attempted")
                self._manual_file(target)
            elif target.exists():
                self.verifier(target)
            else:
                source = read_source(self.root / "runtime-download.json")
                install_runtime(source, target, self.verifier, progress)
            return target
        finally:
            self._preparing.release()

    def status(self):
        files = {"MIT bridge": self.root / "native/nr/bin/dlss5nr_bridge.dll",
                 "MIT caller": self.runtime_dir / "caller/nvngx.dll_comfy.dll",
                 "NVIDIA runtime": self.runtime_dir / "nvngx_dlssnr.dll"}
        return "\n".join(f"{name}: {'present (not GPU-tested)' if path.is_file() else 'missing'}" for name, path in files.items())

    def select_device(self, snapshot):
        from . import settings
        required = {"gpu_index", "device", "gpu_name"}
        if (not required <= snapshot.keys() or type(snapshot["gpu_index"]) is not int
                or not isinstance(snapshot["gpu_name"], str) or not snapshot["gpu_name"].strip()):
            raise ValueError("Select an explicitly enumerated physical GPU")
        settings.save({"nr": {key: snapshot[key] for key in required}})

    def import_runtime(self, filename):
        if not self._preparing.acquire(blocking=False):
            raise DownloadError("preparing", "Runtime preparation is already running")
        try:
            return self._import_runtime(filename)
        finally:
            self._preparing.release()

    def _import_runtime(self, filename):
        if os.name != "nt":
            raise ValueError("NVIDIA NR requires 64-bit Windows")
        source = Path(str(filename).strip().strip('"'))
        if (not source.is_absolute() or str(source).startswith(("\\\\", "//"))
                or source.name.lower() != "nvngx_dlssnr.dll" or not source.is_file()
                or source.resolve() != source or not 1024 <= source.stat().st_size <= 300 * 1024 ** 2):
            raise ValueError("Select a local, non-linked file named nvngx_dlssnr.dll (maximum 300 MiB)")
        target = self.runtime_dir / "nvngx_dlssnr.dll"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.resolve() != target.parent:
            raise ValueError("Runtime destination must not be a directory link")
        before = digest(source)
        if target.is_file() and digest(target) != before:
            raise ValueError("An existing different runtime was preserved. Release NR and remove it manually before importing another version")
        descriptor, temporary = tempfile.mkstemp(prefix="runtime-import-", suffix=".dll", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            shutil.copyfile(source, temporary)
            if digest(temporary) != before or digest(source) != before:
                raise ValueError("Runtime bytes changed during import; nothing installed")
            if self.mode == "manual":
                self._manual_file(temporary)
                verification = "Manual custom file; NVIDIA signature not verified"
            else:
                verification = "NVIDIA signature: " + self.verifier(temporary)["status"]
            if not target.exists():
                os.rename(temporary, target)
            self.configure()
            return ("Runtime copy verified and installed. Original file unchanged.\n"
                    f"{verification}\nSHA-256: {before}\n"
                    "Runtime files are prepared; select the GPU to use.")
        finally:
            temporary.unlink(missing_ok=True)
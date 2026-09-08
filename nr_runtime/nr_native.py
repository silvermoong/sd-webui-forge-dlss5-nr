"""Lazy Windows MIT-bridge adapter. Status is file inspection, not DLL loading.

There is deliberately no runtime installer, Torch import or GPU probe at import.
The NVIDIA proprietary runtime must be supplied separately by its licensed user.
"""
import json
import os
import struct
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "a3de4781eef81afe80d0226d1ede0b46b3346a63"


class NativeError(RuntimeError):
    pass


def pe_exports(path):
    """Read exports from a project-owned PE, without LoadLibrary/DllMain."""
    data = Path(path).read_bytes()
    try:
        if data[:2] != b"MZ":
            return []
        pe = struct.unpack_from("<I", data, 60)[0]
        if data[pe:pe + 4] != b"PE\0\0" or struct.unpack_from("<H", data, pe + 4)[0] != 0x8664:
            return []
        count, optional = struct.unpack_from("<H", data, pe + 6)[0], pe + 24
        size = struct.unpack_from("<H", data, pe + 20)[0]
        if struct.unpack_from("<H", data, optional)[0] != 0x20b:
            return []
        sections = [struct.unpack_from("<IIII", data, optional + size + i * 40 + 8) for i in range(count)]

        def offset(rva):
            for virtual_size, virtual, raw_size, raw in sections:
                if virtual <= rva < virtual + max(virtual_size, raw_size):
                    return raw + rva - virtual
            raise ValueError("Invalid PE RVA")

        export_rva = struct.unpack_from("<I", data, optional + 112)[0]
        if not export_rva:
            return []
        directory = offset(export_rva)
        names_count = struct.unpack_from("<I", data, directory + 24)[0]
        names = offset(struct.unpack_from("<I", data, directory + 32)[0])
        if names_count > 4096:
            return []
        result = []
        for i in range(names_count):
            at = offset(struct.unpack_from("<I", data, names + i * 4)[0])
            end = data.index(b"\0", at, at + 512)
            result.append(data[at:end].decode("ascii"))
        return sorted(result)
    except (ValueError, IndexError, struct.error, UnicodeError):
        return []


def readiness(runtime=None):
    """CPU/file-only readiness for parent status; never claims hardware validation."""
    runtime = runtime or {}
    bridge = Path(runtime.get("bridge") or ROOT / "native/nr/bin/dlss5nr_bridge.dll")
    directory = Path(runtime.get("runtime_dir") or ROOT / "models/dlssnr")
    paths = {"bridge": bridge, "runtime": directory / "nvngx_dlssnr.dll",
             "caller": directory / "caller/nvngx.dll_comfy.dll"}
    missing = [f"Missing {name}: {path}" for name, path in paths.items() if not path.is_file()]
    if sys.platform != "win32" or struct.calcsize("P") != 8:
        missing.append("Native NR requires 64-bit Windows")
    build = None
    manifest = bridge.parent / "build.json"
    if manifest.is_file() and manifest.stat().st_size <= 131072:
        try:
            build = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    exports = pe_exports(bridge) if bridge.is_file() and bridge.stat().st_size <= 16777216 else []
    return dict(ready=not missing, missing=missing, hardware_verified=False,
                runtime_id=runtime.get("runtime_id", ""), build=build,
                capabilities={"file_inspection_only": True, "upstream_commit": UPSTREAM_COMMIT,
                              "hot_feature_reset": "tagsystem_nr_reset_feature" in exports,
                              "luid_enumeration": "tagsystem_nr_devices_json" in exports},
                warnings=["Runtime presence is not NR/GPU/license validation; no device has been initialized"])


def canonical_gpu_name(value):
    """Whitespace/case canonicalization only: never remove SKU/product identity."""
    return " ".join(str(value).split()).casefold()


def _load_library(path):
    import ctypes
    if sys.platform != "win32" or struct.calcsize("P") != 8:
        raise NativeError("Native NR requires 64-bit Windows")
    try:
        return ctypes.WinDLL(str(path), winmode=0x00000100 | 0x00001000)
    except OSError as exc:
        raise NativeError(f"Could not load the project MIT bridge: {exc}") from exc


def _function(library, name, args, result, *, optional=False):
    function = getattr(library, name, None)
    if function is None:
        if optional:
            return None
        raise NativeError(f"Native bridge is missing export {name}")
    function.argtypes = args
    function.restype = result
    return function


def _enumerate(library):
    import ctypes as c
    function = _function(library, "tagsystem_nr_devices_json", [c.c_char_p, c.c_int, c.c_char_p, c.c_int], c.c_int)
    output, error = c.create_string_buffer(65536), c.create_string_buffer(4096)
    if not function(output, len(output), error, len(error)):
        raise NativeError(error.value.decode("utf-8", "replace") or "Device enumeration failed")
    try:
        result = json.loads(output.value.decode("utf-8"))
        if not isinstance(result, dict) or not isinstance(result.get("devices"), list):
            raise ValueError("Expected a devices list")
        for device in result["devices"]:
            if not isinstance(device, dict) or not {"gpu_index", "gpu_name", "device", "luid", "total_memory"} <= device.keys():
                raise ValueError("Incomplete physical adapter record")
        result.setdefault("warnings", [])
        return result
    except (ValueError, UnicodeError) as exc:
        raise NativeError(f"Invalid native device enumeration: {exc}") from exc


def _mapping_environment():
    return tuple(os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"))


def inspect_devices(runtime=None, *, library_loader=None):
    """Explicit DXGI/CUDA-LUID enumeration, with no NR/model/context initialization.

    This function is NOT called by status/import. cuInit enumerates the driver;
    it is still an explicit hardware action and belongs in the isolated worker.
    library_loader is a Python-only test seam, not a command/env fake mode.
    """
    from .nr_media import local_path
    runtime = runtime or {}
    bridge = Path(runtime.get("bridge") or ROOT / "native/nr/bin/dlss5nr_bridge.dll")
    if not bridge.is_file():
        raise NativeError(f"Missing MIT device enumeration bridge: {bridge}")
    bridge = local_path(bridge)
    library = (library_loader or _load_library)(bridge)
    if not hasattr(library, "tagsystem_nr_devices_json"):
        enumerator = ROOT / "native/nr/bin/dlss5nr_bridge.dll"
        if bridge == enumerator or not enumerator.is_file():
            raise NativeError("Bridge cannot enumerate physical LUIDs; build the project MIT wrapper")
        library = (library_loader or _load_library)(enumerator)
    result = _enumerate(library)
    visible, order = _mapping_environment()
    if visible is not None or order not in (None, "", "FASTEST_FIRST"):
        for device in result["devices"]:
            device["device"] = ""
        result["warnings"].append("CUDA environment changes prevent verified default cuda:N mapping; parent must launch a clean-environment inspection")
    return result


class NativeDriver:
    """One lazy, serial DLL/device session, independent of ComfyUI and Torch.

    begin(runtime, params, source_key=..., temporal=...) establishes a *complete*
    request. process consumes one contiguous HxWx3 float32 frame in SDR [0,1].
    Every request's first process must use reset=True. Model floats/flow/source
    changes recreate the feature (or conservatively shutdown/init an old bridge).
    The outer mix is deliberately not a native model parameter.
    """
    def __init__(self, *, library_loader=None):
        self._loader = library_loader or _load_library
        self._injected_loader = library_loader is not None
        self._library = None
        self._initialized = False
        self._runtime_key = None
        self._model_key = None
        self._source_key = None
        self._first = True
        self._directories = []
        self._lock = threading.RLock()
        self._info = {}

    def _bind(self):
        import ctypes as c
        library = self._library
        self._init = _function(library, "dlss5nr_init", [c.c_int, c.c_wchar_p, c.c_char_p, c.c_int], c.c_int)
        self._process = _function(library, "dlss5nr_process",
                                  [c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_int, c.c_int,
                                   c.c_int, c.c_int, c.c_float, c.c_float, c.c_float, c.c_float,
                                   c.c_int, c.c_int, c.c_int, c.c_char_p, c.c_int], c.c_int)
        self._shutdown = _function(library, "dlss5nr_shutdown", [], None)
        self._gpu_name = _function(library, "dlss5nr_gpu_name", [], c.c_char_p)
        self._reset = _function(library, "tagsystem_nr_reset_feature", [], None, optional=True)
        self._gpu_luid = _function(library, "tagsystem_nr_gpu_luid", [], c.c_char_p, optional=True)

    def _initialize(self, runtime, expected):
        import ctypes as c
        error = c.create_string_buffer(4096)
        if not self._init(runtime["gpu_index"], runtime["runtime_dir"], error, len(error)):
            # The upstream init unwinds failed initialization itself.
            raise NativeError(error.value.decode("utf-8", "replace") or "NR initialization failed")
        self._initialized = True
        name = (self._gpu_name() or b"").decode("utf-8", "replace")
        if canonical_gpu_name(name) != canonical_gpu_name(runtime["gpu_name"]):
            self.close()
            raise NativeError("Initialized GPU name differs from the explicitly requested GPU")
        if self._gpu_luid:
            luid = (self._gpu_luid() or b"").decode("ascii", "replace")
            if not luid or luid.lower() != expected["luid"].lower():
                self.close()
                raise NativeError("Initialized adapter LUID differs from the physically verified CUDA device")

    def begin(self, runtime, params, *, source_key, temporal):
        from .nr_media import local_path
        from .nr_worker import validate_params
        params = validate_params(params)
        if type(temporal) is not bool:
            raise ValueError("temporal must be boolean")
        if runtime.get("channel_order") not in ("RGBA", "BGRA"):
            raise ValueError("Fixed RGBA/BGRA channel_order is required; auto guessing is not supported")
        status = readiness(runtime)
        if not status["ready"]:
            raise NativeError("; ".join(status["missing"]))
        runtime = dict(runtime)
        runtime["bridge"] = str(local_path(runtime["bridge"]))
        runtime["runtime_dir"] = str(local_path(runtime["runtime_dir"], directory=True))
        native_key = tuple(params[key] for key in ("style", "preset", "intensity", "tone", "structure", "skin", "auto_mask", "flow")) + (temporal,)
        runtime_key = tuple(runtime.get(key) for key in ("bridge", "runtime_dir", "gpu_index", "device", "gpu_name", "channel_order", "runtime_id"))
        environment = _mapping_environment()
        if environment[0] is not None or environment[1] not in (None, "", "FASTEST_FIRST"):
            raise NativeError("CUDA environment is overridden; default physical device mapping is not verified")
        with self._lock:
            if self._runtime_key != runtime_key or not self._initialized:
                self.close()
                if not self._injected_loader and hasattr(os, "add_dll_directory"):
                    for directory in (Path(runtime["bridge"]).parent, Path(runtime["runtime_dir"]), Path(runtime["runtime_dir"]) / "caller"):
                        self._directories.append(os.add_dll_directory(str(directory)))
                try:
                    self._library = self._loader(Path(runtime["bridge"]))
                    self._bind()
                    if hasattr(self._library, "tagsystem_nr_devices_json"):
                        inspection = _enumerate(self._library)
                    else:
                        inspection = inspect_devices(runtime, library_loader=self._loader)
                    matches = [d for d in inspection["devices"] if d["gpu_index"] == runtime.get("gpu_index")]
                    if len(matches) != 1:
                        raise NativeError("Requested NVIDIA DXGI adapter was not uniquely enumerated")
                    expected = matches[0]
                    if not expected["device"] or expected["device"] != runtime.get("device") or not expected["luid"]:
                        raise NativeError("Requested adapter is not the physically verified CUDA device (LUID mismatch/unmapped)")
                    if not runtime.get("gpu_name") or canonical_gpu_name(expected["gpu_name"]) != canonical_gpu_name(runtime["gpu_name"]):
                        raise NativeError("Requested GPU name differs from physical adapter inspection")
                    self._initialize(runtime, expected)
                    self._expected = expected
                    self._runtime = runtime
                    self._runtime_key = runtime_key
                    self._info = dict(gpu_name=expected["gpu_name"], device=expected["device"], luid=expected["luid"],
                                      hardware_verified=True, warnings=list(inspection.get("warnings", [])),
                                      capabilities=dict(hot_feature_reset=bool(self._reset),
                                                        physical_luid_after_init=bool(self._gpu_luid),
                                                        channel_order=runtime["channel_order"], abi="RGB float32 / int32 flags"))
                    if not self._reset:
                        self._info["warnings"].append("Legacy bridge: model/source changes use conservative shutdown/init, not hot parameter updates")
                except BaseException:
                    self.close()
                    raise
            elif native_key != self._model_key or source_key != self._source_key:
                if self._reset:
                    self._reset()
                else:
                    self._shutdown()
                    self._initialized = False
                    self._initialize(runtime, self._expected)
            self._model_key = native_key
            self._source_key = source_key
            self._temporal = temporal
            self._first = True
            return {**self._info, "warnings": list(self._info["warnings"])}

    def process(self, rgb, *, params, reset, temporal):
        import ctypes as c
        import numpy as np
        from .nr_worker import validate_params
        params = validate_params(params)
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[2] != 3
                or not rgb.flags.c_contiguous or not np.isfinite(rgb).all() or rgb.size == 0
                or min(rgb.shape[:2]) < 1 or max(rgb.shape[:2]) > 16384 or rgb.min() < 0 or rgb.max() > 1):
            raise ValueError("Native input must be contiguous finite float32 HxWx3 SDR RGB in [0,1]")
        if type(reset) is not bool or type(temporal) is not bool:
            raise ValueError("reset and temporal must be boolean")
        key = tuple(params[name] for name in ("style", "preset", "intensity", "tone", "structure", "skin", "auto_mask", "flow")) + (temporal,)
        with self._lock:
            if not self._initialized or key != self._model_key:
                raise NativeError("Call begin with the complete request before processing frames or changing model parameters")
            if self._first and not reset:
                raise NativeError("First frame of every new request must explicitly reset NR/OFA history")
            height, width = rgb.shape[:2]
            output = np.empty_like(rgb)
            error = c.create_string_buffer(4096)
            ok = self._process(rgb.ctypes.data_as(c.POINTER(c.c_float)), output.ctypes.data_as(c.POINTER(c.c_float)),
                               width, height, params["style"], params["preset"], params["intensity"], params["tone"],
                               params["structure"], params["skin"], int(params["auto_mask"]), int(reset), int(temporal),
                               error, len(error))
            if not ok:
                message = error.value.decode("utf-8", "replace") or "NR EvaluateFeature failed"
                self.close()
                raise NativeError(message)
            if not np.isfinite(output).all():
                self.close()
                raise NativeError("NR returned nonfinite pixels")
            self._first = False
            if self._runtime["channel_order"] == "BGRA":
                output = output[..., ::-1]
            return np.ascontiguousarray(np.clip(output, 0, 1), dtype=np.float32)

    def close(self):
        with self._lock:
            try:
                if self._initialized:
                    self._shutdown()
            finally:
                self._initialized = False
                self._library = None
                self._runtime_key = self._model_key = self._source_key = None
                self._first = True
                for directory in self._directories:
                    directory.close()
                self._directories.clear()
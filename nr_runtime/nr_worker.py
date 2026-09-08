"""Independent NR JSON-lines worker; no implicit GPU work at module import.

NativeSession(native=...) and serve(session=..., inspector=...) are the Python
test seams. The executable has no fake renderer flag or environment override.
"""
import copy
import json
import math
import re
import threading

from . import nr_media
from .nr_media import inspect_media, check_cancel, MediaError
from .nr_native import NativeError, readiness

DEFAULT_PARAMS = dict(style=1, preset=3, intensity=1., tone=1., structure=1., skin=-1.,
                      auto_mask=False, mix=1., flow=True)


def validate_params(value):
    """Deterministic defaults, finite typed numbers, no ignored parameters."""
    if not isinstance(value, dict) or set(value) - DEFAULT_PARAMS.keys():
        raise ValueError("Invalid/unknown NR params")
    params = {**DEFAULT_PARAMS, **value}
    for name in ("auto_mask", "flow"):
        if type(params[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    for name, low, high in (("style", 0, 2), ("preset", 0, 3), ("intensity", 0, 2),
                            ("tone", 0, 2), ("structure", 0, 2), ("skin", -1, 2), ("mix", 0, 1)):
        number = params[name]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{name} must be finite and within {low}..{high}")
        if name in ("style", "preset") and type(number) is not int:
            raise ValueError(f"{name} must be an integer")
    return params


def validate_command(command):
    if not isinstance(command, dict) or command.get("op") != "run":
        raise ValueError("Expected op:run")
    if not isinstance(command.get("id"), str) or not 0 < len(command["id"]) <= 128:
        raise ValueError("A nonempty request id of at most 128 characters is required")
    allowed = {"op", "id", "source_path", "output_dir", "source", "params", "start", "end",
               "preview_kind", "max_edge", "runtime"}
    if set(command) - allowed:
        raise ValueError("Unknown NR run fields")
    result = copy.deepcopy(command)
    result["params"] = validate_params(command.get("params", {}))
    for key in ("start", "end"):
        value = command.get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a nonnegative finite number")
        result[key] = value
    if result["end"] and result["end"] <= result["start"]:
        raise ValueError("Selection end must be after start")
    preview = result.setdefault("preview_kind", "")
    if preview not in ("", "still", "clip"):
        raise ValueError("preview_kind must be empty, still or clip")
    edge = result.setdefault("max_edge", 0)
    if type(edge) is not int or not 0 <= edge <= 16384 or (not preview and edge):
        raise ValueError("max_edge is a preview-only integer; final output requires 0")
    if not isinstance(result.get("source"), dict) or not isinstance(result.get("runtime"), dict):
        raise ValueError("source and runtime snapshots are required")
    runtime = result["runtime"]
    if runtime.get("channel_order") not in ("RGBA", "BGRA"):
        raise ValueError("runtime.channel_order must explicitly be RGBA or BGRA")
    if type(runtime.get("gpu_index")) is not int or runtime["gpu_index"] < 0:
        raise ValueError("Explicit NVIDIA adapter gpu_index is required")
    if not isinstance(runtime.get("device"), str) or not re.fullmatch(r"cuda:\d+", runtime["device"]):
        raise ValueError("Explicit physically verified cuda:N device is required")
    if not isinstance(runtime.get("gpu_name"), str) or not runtime["gpu_name"].strip():
        raise ValueError("Expected GPU name is required")
    result["source_path"] = str(nr_media.local_path(result.get("source_path")))
    output = nr_media.local_path(result.get("output_dir"), directory=True)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output_dir must be a fresh empty directory")
    result["output_dir"] = str(output)
    return result


class NativeSession:
    """Serial request driver. A missing native dependency is never pass-through."""
    def __init__(self, native=None):
        self.native = native

    def _begin(self, command, *, temporal):
        if self.native is None:
            status = readiness(command["runtime"])
            if not status["ready"]:
                raise NativeError("; ".join(status["missing"]))
            from .nr_native import NativeDriver
            self.native = NativeDriver()
        source = command["source"]
        key = (source.get("id"), source.get("sha256"), command["source_path"])
        return self.native.begin(command["runtime"], command["params"], source_key=key, temporal=temporal)

    def _frame(self, original, params, cancel_event, *, reset, temporal):
        import numpy as np
        check_cancel(cancel_event)
        neural = self.native.process(original.copy(), params=params, reset=reset, temporal=temporal)
        check_cancel(cancel_event)
        if not isinstance(neural, np.ndarray) or neural.dtype != np.float32 or neural.shape != original.shape or not np.isfinite(neural).all():
            raise NativeError("Native output must be finite float32 RGB with unchanged dimensions")
        neural = np.clip(neural, 0, 1)
        mixed = original + (neural - original) * np.float32(params["mix"])
        return neural, mixed

    def run(self, command, emit, cancel_event):
        from pathlib import Path
        command = validate_command(command)
        check_cancel(cancel_event)
        if self.native is None:
            # Dependency failure is a cheap, useful answer even without PyAV or
            # numpy. Do not decode the entire video before discovering no NR.
            status = readiness(command["runtime"])
            if not status["ready"]:
                raise NativeError("; ".join(status["missing"]))

        def progress(message, ratio):
            check_cancel(cancel_event)
            emit(dict(id=command["id"], type="progress", message=message, ratio=ratio))

        facts = nr_media.source_facts(command["source"])
        if facts is None:
            facts = inspect_media(command["source_path"], cancel_event=cancel_event, progress=progress)
        if facts["kind"] != "image":
            return nr_media.process_video(command, facts=facts,
                                           native_begin=lambda **kw: self._begin(command, **kw),
                                           process_frame=lambda rgb, **kw: self._frame(rgb, command["params"], cancel_event, **kw),
                                           progress=progress, cancel_event=cancel_event)
        if command["start"] or command["end"] or command["preview_kind"] == "clip":
            raise ValueError("Still images have no clip/time selection")
        image = nr_media.read_image(command["source_path"], max_edge=command["max_edge"])
        original = image["rgb"]
        check_cancel(cancel_event)
        native_info = self._begin(command, temporal=False) or {}
        progress("Processing SDR still image", 0.)
        neural, mixed = self._frame(original, command["params"], cancel_event, reset=True, temporal=False)
        height, width = original.shape[:2]
        preview = bool(command["preview_kind"])
        approximate = (width, height) != (facts["width"], facts["height"])
        names = ["output.png", "original.png", "neural.png"] if preview else ["output.png"]
        result = dict(name="output.png", kind="image", width=width, height=height, duration=0., frames=1,
                      fps=None, files=names, warnings=[*image["warnings"], *native_info.get("warnings", [])],
                      parent=command["source"], params=command["params"], runtime_id=command["runtime"].get("runtime_id", ""),
                      selection=dict(start=0., end=0.), preview=preview, approximate=approximate,
                      note="缩小预览（近似，不裁切）" if approximate else "原尺寸；首帧清空历史",
                      color_space="srgb", mix_space="display-encoded sRGB float32",
                      native=native_info)
        if preview:
            result.update(original_name="original.png", neural_name="neural.png")
        output = Path(command["output_dir"])
        try:
            nr_media.publish_images(output, result, mixed=mixed, original=original, neural=neural,
                                     alpha=image["alpha"], cancel_event=cancel_event)
            progress("Verified SDR PNG output", 1.)
            check_cancel(cancel_event)
            return result
        except BaseException:
            for path in (output / name for name in names):
                path.unlink(missing_ok=True)
            raise

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def serve(incoming, outgoing, *, session=None, inspector=None):
    """Blocking JSON-lines service: one executor, one independent stdin reader.

    There is at most one waiting request, replaced latest-only. The parent owns
    preemption: a newer run does NOT cancel the active request by itself. EOF is
    shutdown, so a disappearing parent cannot leave its GPU session working.
    The only synthetic entry points are these explicit Python arguments.
    """
    session = session if session is not None else NativeSession()
    condition = threading.Condition()
    output_lock = threading.Lock()
    state = dict(active=None, pending=None, stopping=False)

    def send(event):
        text = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        with output_lock:
            outgoing.write(text)
            outgoing.flush()

    def error(request_id, message, canceled=False, *, category=None):
        event = dict(id=request_id, type="error", error=str(message), canceled=bool(canceled))
        if category is not None and not canceled:
            event["category"] = category
        send(event)

    def stop():
        with condition:
            state["stopping"] = True
            if state["active"]:
                state["active"][1].set()
            if state["pending"]:
                error(state["pending"]["id"], "Worker stopped before request started", True)
                state["pending"] = None
            condition.notify_all()

    def read_commands():
        try:
            while True:
                line = incoming.readline(1048577)
                if not line:
                    break
                command = None
                try:
                    if len(line) > 1048576:
                        newline = b"\n" if isinstance(line, bytes) else "\n"
                        while line and not line.endswith(newline):
                            line = incoming.readline(1048577)
                        raise ValueError("NR command exceeds the 1 MiB line limit")
                    command = json.loads(line, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
                    if not isinstance(command, dict):
                        raise ValueError("Expected a JSON object")
                    op = command.get("op")
                    if op == "shutdown":
                        break
                    request_id = command.get("id")
                    if not isinstance(request_id, str) or not 0 < len(request_id) <= 128:
                        raise ValueError("A nonempty request id of at most 128 characters is required")
                    with condition:
                        if state["stopping"]:
                            break
                        if op == "cancel":
                            if state["active"] and state["active"][0] == request_id:
                                state["active"][1].set()
                            if state["pending"] and state["pending"]["id"] == request_id:
                                state["pending"] = None
                                error(request_id, "Canceled before request started", True)
                            continue
                        if op not in ("run", "inspect"):
                            raise ValueError("Unknown worker operation")
                        if (state["active"] and state["active"][0] == request_id) or (
                                state["pending"] and state["pending"]["id"] == request_id):
                            raise ValueError("Duplicate in-flight request id")
                        if state["pending"]:
                            error(state["pending"]["id"], "Replaced by newer waiting request", True)
                        state["pending"] = command
                        condition.notify_all()
                except (ValueError, TypeError) as exc:
                    request_id = command.get("id") if isinstance(command, dict) else None
                    error(request_id, exc)
        finally:
            stop()

    reader = threading.Thread(target=read_commands, name="nr-stdin", daemon=True)
    reader.start()
    try:
        while True:
            with condition:
                condition.wait_for(lambda: state["pending"] is not None or state["stopping"])
                if state["stopping"]:
                    break
                command = state["pending"]
                state["pending"] = None
                cancel = threading.Event()
                state["active"] = (command["id"], cancel)
            try:
                if command["op"] == "inspect":
                    if inspector is None:
                        from .nr_native import inspect_devices
                        result = inspect_devices(command.get("runtime", {}))
                    else:
                        result = inspector(command.get("runtime", {}))
                else:
                    result = session.run(command, send, cancel)
                # Serialize publication with cancel/shutdown processing. A late
                # cancel after this terminal event belongs to the parent's ticket.
                with condition:
                    check_cancel(cancel)
                    send(dict(id=command["id"], type="result", result=result))
            except Exception as exc:
                error(command["id"], exc, cancel.is_set() or isinstance(exc, nr_media.Cancelled),
                      category="environment" if isinstance(exc, NativeError) or command["op"] == "inspect" else None)
            finally:
                with condition:
                    state["active"] = None
                    condition.notify_all()
    finally:
        stop()
        session.close()


def main(*, session=None, inspector=None):
    import os
    import sys
    if len(sys.argv) != 1:
        print("NR worker accepts JSON-lines on stdin only; no renderer command-line flags", file=sys.stderr)
        return 2
    # Windows numpy/OpenBLAS first-load can block while a stdin reader is alive
    # (also reproduced with raw kernel32 ReadFile). Warm *CPU-only* media modules
    # before starting that thread. Never import/load GPU/NR/setup here. Missing
    # PyAV stays a clean per-video dependency error, not an executable failure.
    import importlib.util
    import numpy  # noqa: F401
    from PIL import Image, ImageCms  # noqa: F401
    if importlib.util.find_spec("av") is not None:
        import av  # noqa: F401
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    # Native printf/driver diagnostics must never corrupt the protocol pipe.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1) as protocol_out:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        serve(sys.stdin, protocol_out, session=session, inspector=inspector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
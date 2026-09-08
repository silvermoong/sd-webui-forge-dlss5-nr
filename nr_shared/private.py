"""Owned stdio controller entry point. Core imports occur only inside this process."""
import contextlib
import os
from pathlib import Path
import sys
import uuid

from .contract import SharedError
from .transports import MAX_BODY, decode, encode


def protocol_output():
    """Preserve the protocol fd; redirect Python, CRT and Win32 native stdout to stderr."""
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.set_inheritable(protocol.fileno(), False)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
        api.SetStdHandle.restype = wintypes.BOOL
        if not api.SetStdHandle(wintypes.DWORD(-11), msvcrt.get_osfhandle(sys.stderr.fileno())):
            protocol.close()
            raise SharedError(503, "Cannot isolate NR protocol output")
    sys.stdout = sys.stderr
    return protocol


def serve(host, incoming, outgoing):
    """Python test seam: a Host fixture can run without importing core or any GPU library."""
    try:
        while True:
            line = incoming.readline(MAX_BODY + 1)
            if not line:
                return
            ident = None
            try:
                if not line.endswith(b"\n"):
                    raise SharedError(413, "NR input line exceeds 1 MiB or is incomplete")
                envelope = decode(line)
                ident = envelope.get("id")
                if (set(envelope) != {"id", "body"} or not isinstance(ident, str)
                        or not 0 < len(ident) <= 128 or not isinstance(envelope["body"], dict)):
                    raise SharedError(400, "Invalid NR private RPC envelope")
                result = host.dispatch(envelope["body"])
                response = {"id": ident, "result": result}
            except SharedError as error:
                response = {"id": ident, "error": str(error), "status": error.status}
            except Exception:
                response = {"id": ident, "error": "NR private control failed", "status": 500}
            try:
                payload = encode(response) + b"\n"
            except SharedError:
                payload = encode({"id": ident, "error": "NR result exceeds protocol limits", "status": 502}) + b"\n"
            data = memoryview(payload)
            while data:
                written = outgoing.write(data)
                if not written:
                    raise BrokenPipeError("NR protocol output closed")
                data = data[written:]
            outgoing.flush()
            if len(line) > MAX_BODY or not line.endswith(b"\n"):
                return  # Never parse the tail of an oversized line as a new command.
    finally:
        host.close()


@contextlib.contextmanager
def _private_gate(ref, event, wait=30):
    # Private Host already serializes the worker. No tenants, GPU arbiter or coordinator.
    if event.is_set():
        raise SharedError(409, "NR canceled")
    yield


def main():
    with protocol_output() as outgoing:
        # Complete CPU DLL imports BEFORE any blocking stdin reader is started.
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
        from nr_runtime.manager import ManagedWorker
        from nr_runtime import settings
        from .catalog import Catalog
        from .host import Host

        root = Path.cwd().resolve()
        worker = ManagedWorker(log_dir=root / "work/_gen/nr-private" / uuid.uuid4().hex / "logs",
                               idle_seconds=lambda: 60)
        try:
            host = Host(root, worker, catalog=Catalog(root, config_provider=settings.get), gate=_private_gate)
        except BaseException:
            worker.close()
            raise
        serve(host, sys.stdin.buffer, outgoing)


if __name__ == "__main__":
    main()
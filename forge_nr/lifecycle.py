"""Client ownership and a read-only capabilities route, independent of FastAPI versions."""
from contextlib import contextmanager
from pathlib import Path
import threading

from nr_shared.contract import ARG_KEYS, PROTOCOL, SCRIPT_TITLE
from . import VERSION


class RepairBlocked(RuntimeError):
    code = "repair_blocked"


class Service:
    def __init__(self, root, *, client_factory=None):
        self.root = Path(root).resolve()
        self.factory = client_factory
        self.client = None
        self.started = False
        self.error = "Private NR controller has not started yet"
        self.lock = threading.RLock()
        self._operations = 0
        self._repairing = False

    def start(self):
        with self.lock:
            if self._repairing:
                raise RepairBlocked("Dependency repair is already running")
            if self.started:
                return
            try:
                factory = self.factory
                if factory is None:
                    from nr_shared.client import Client
                    factory = Client
                self.client = factory(root=self.root, label="forge")
                self.client.start()
                self.started = True
                self.error = ""
            except Exception as exc:
                self.error = "Private NR controller startup failed: " + str(exc)
                if self.client is not None:
                    self.client.close()
                    self.client = None
                raise

    def _get(self):
        with self.lock:
            if self._repairing:
                raise RepairBlocked("Dependency repair is running; no request was submitted")
            if not self.started or self.client is None:
                raise RuntimeError(self.error or "NR extension is unloaded")
            return self.client

    def runtime(self, device=None):
        return self._get().runtime(device=device)

    @contextmanager
    def _operation(self):
        with self.lock:
            client = self._get()
            self._operations += 1
        try:
            yield client
        finally:
            with self.lock:
                self._operations -= 1

    def inspect(self, runtime, cancel_event=None):
        with self._operation() as client:
            return client.inspect(runtime, cancel_event=cancel_event)

    def run(self, command, progress, cancel_event):
        with self._operation() as client:
            return client.run(command, progress=progress, cancel_event=cancel_event)

    def repair(self, action):
        with self.lock:
            if self._repairing or self._operations:
                raise RepairBlocked("An NR request or dependency repair is active")
            client = self.client
            if client is not None:
                try:
                    status = client.status()
                except Exception as error:
                    raise RepairBlocked("Cannot confirm that the private NR controller is idle") from error
                if not isinstance(status, dict) or status.get("busy") is not False:
                    raise RepairBlocked("The private NR controller is busy or its state is unknown")
            self._repairing = True
            self.client = None
            self.started = False
            self.error = "Dependency repair is running"
        try:
            if client is not None:
                client.close()
            return action()
        finally:
            with self.lock:
                self._repairing = False
                self.error = "Private NR controller needs to be started after dependency repair"

    def status(self):
        return self._get().status()

    def connection_info(self):
        with self.lock:
            client = self.client
            return dict(connected=self.started and client is not None,
                        mode=getattr(client, "mode", None), instance=getattr(client, "instance", None))

    def stop(self):
        return self._get().stop()

    def close(self):
        with self.lock:
            client, self.client = self.client, None
            self.started = False
            self.error = "NR extension is unloaded; its private controller is closed"
        if client is not None:
            client.close()


def app_started(app, service, *, ready_hook):
    """Register once across reloads; the endpoint reports current hook availability."""
    def capabilities():
        return dict(protocol=PROTOCOL, script=SCRIPT_TITLE, arg_keys=list(ARG_KEYS),
                    ready_hook=bool(ready_hook()), version=VERSION, **service.connection_info())

    if not hasattr(app, "_forge_nr_capabilities"):
        if any(getattr(route, "path", None) == "/forge-nr/capabilities" for route in app.routes):
            raise RuntimeError("另一个扩展已占用 NR capabilities 路由")

        def endpoint():
            return app._forge_nr_capabilities()

        app.add_api_route("/forge-nr/capabilities", endpoint, methods=["GET"])
    app._forge_nr_capabilities = capabilities
    service.start()
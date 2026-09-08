"""Synchronous NR client: select once, explicit ownership, never retry submission."""
from pathlib import Path
import re
import threading
import time
import uuid

from .contract import PROTOCOL, SharedError
from .transports import PrivateTransport


class Client:
    HEARTBEAT = 5.
    POLL_INTERVAL = .1
    WORK_RPC_TIMEOUT = 1.
    CANCEL_WAIT = 3.  # Leaves room for one 1s work RPC and bounded private tree cleanup (<5s).

    def __init__(self, root, label="forge", transport=None):
        self.root = Path(root).resolve()
        self.execution_root = self.root
        self.label = label
        self.transport = transport
        self.mode = None
        self.instance = None
        self._client = None
        self._start_lock = threading.Lock()
        self._closed = threading.Event()
        self._heartbeat_thread = None
        self._failure = None
        self._attempted = False

    @staticmethod
    def _hello(transport):
        hello = transport.rpc({"op": "hello"})
        if (not isinstance(hello, dict) or type(hello.get("protocol")) is not int
                or hello["protocol"] != PROTOCOL or not isinstance(hello.get("instance"), str)
                or not hello["instance"] or hello.get("mode") != "private"
                or hello["mode"] != transport.mode
                or (getattr(transport, "instance", None) is not None and hello["instance"] != transport.instance)):
            raise SharedError(409, "NR bridge protocol/instance/mode mismatch")
        return hello

    def start(self):
        with self._start_lock:
            self._check()
            if self._client is not None:
                return self
            if self._attempted:
                raise SharedError(503, "NR client startup already failed; create a new client explicitly")
            self._attempted = True
            try:
                if self.transport is None:
                    self.transport = PrivateTransport(self.root)
                hello = self._hello(self.transport)
                # No fallback after hello; failed join may already have been accepted.
                self.mode, self.instance = hello["mode"], hello["instance"]
                joined = self.transport.rpc({"op": "join", "label": self.label})
                if (not isinstance(joined, dict) or joined.get("instance") != self.instance
                        or joined.get("mode") != self.mode or not isinstance(joined.get("client"), str)
                        or not re.fullmatch(r"[a-f0-9]{64}", joined["client"])):
                    raise SharedError(409, "Invalid NR client registration")
                self._client = joined["client"]
                self._heartbeat_thread = threading.Thread(target=self._heartbeat, name="nr-client-heartbeat", daemon=True)
                self._heartbeat_thread.start()
                return self
            except BaseException as error:
                self._failure = error if isinstance(error, SharedError) else SharedError(503, "NR client startup failed")
                if self.transport is not None:
                    self.transport.close()
                raise self._failure from None

    def _check(self):
        if self._closed.is_set():
            raise SharedError(503, "NR client is closed")
        if self._failure is not None:
            raise self._failure

    def _heartbeat(self):
        while not self._closed.wait(self.HEARTBEAT):
            try:
                value = self._rpc("heartbeat", timeout=2.)
                if not isinstance(value, dict) or value.get("ok") is not True:
                    raise SharedError(502, "Invalid NR heartbeat response")
            except Exception as error:
                self._failure = error if isinstance(error, SharedError) else SharedError(503, "NR heartbeat failed")
                return

    def _rpc(self, op, *, timeout=None, **fields):
        self._check()
        transport = self.transport
        if transport is None:
            raise SharedError(503, "NR client is closed")
        try:
            return transport.rpc({"op": op, "client": self._client, **fields}, timeout=timeout)
        except SharedError as error:
            if error.status >= 500 or error.status in (401, 403):
                self._failure = error
            raise

    def status(self):
        self.start()
        value = self._rpc("status")
        if not isinstance(value, dict) or value.get("mode") != self.mode or value.get("instance") != self.instance:
            raise SharedError(409, "NR status instance/mode changed")
        return value

    def runtime(self, device=None):
        self.start()
        return self._rpc("runtime", device=device)

    def inspect(self, runtime, cancel_event=None):
        return self.run({"op": "inspect", "id": uuid.uuid4().hex, "runtime": runtime},
                        lambda _: None, cancel_event)

    @staticmethod
    def _ticket(value, ticket=None):
        if (not isinstance(value, dict) or not isinstance(value.get("ticket"), str)
                or not re.fullmatch(r"[a-f0-9]{32}", value["ticket"])
                or (ticket is not None and value["ticket"] != ticket)
                or value.get("status") not in ("pending", "running", "done", "error", "canceled")):
            raise SharedError(502, "Invalid NR ticket response")
        return value

    def _cancel(self, ticket, transport):
        """True only after this ticket is terminal, never merely after cancel/leave ACK."""
        deadline = time.monotonic() + self.CANCEL_WAIT
        body = {"op": "cancel", "client": self._client, "ticket": ticket}
        try:
            # Bypass a remembered heartbeat error: still attempt this exact ticket, never release others.
            value = transport.rpc(body, timeout=min(1., deadline - time.monotonic()))
            while time.monotonic() < deadline:
                value = self._ticket(value, ticket)
                if value["status"] in ("done", "error", "canceled"):
                    return value.get("execution_unknown") is not True
                body["op"] = "poll"
                # A closed client must not turn the wait into a busy-spin.
                threading.Event().wait(min(self.POLL_INTERVAL, max(0., deadline - time.monotonic())))
                value = transport.rpc(body, timeout=min(.5, deadline - time.monotonic()))
        except Exception:
            pass
        return False

    def run(self, command, progress, cancel_event):
        """Run once. On ambiguous failure SharedError.execution_unknown is True.

        The caller MUST retain and not reuse its input/output directory in that
        case. request_id and ticket (possibly None after lost submit) identify the
        operation. close()/leave acknowledgement does NOT clear this obligation.
        False means the ticket was observed terminal, not simply canceled locally.
        Without a ticket the current Host protocol offers no completion lookup;
        retain conservatively, never resubmit or release the global worker.
        """
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        if cancel_event.is_set():
            raise SharedError(409, "NR canceled before submission")
        self.start()
        if cancel_event.is_set():
            raise SharedError(409, "NR canceled before submission")
        ticket = None
        terminal = False
        transport = self.transport  # Retain the exact connection across concurrent close().
        submitted = False
        try:
            self._check()
            submitted = True  # A lost submit reply can leave an executing request without a ticket.
            value = self._ticket(self._rpc("submit", command=command, timeout=self.WORK_RPC_TIMEOUT))
            ticket = value["ticket"]
            while True:
                self._check()
                if cancel_event.is_set():
                    raise SharedError(409, "NR canceled")
                value = self._ticket(value, ticket)
                state = value["status"]
                terminal = state in ("done", "error", "canceled")
                if terminal:
                    if value.get("execution_unknown") is True:
                        terminal = False
                        raise SharedError(503, "NR worker未确认停止；结果不能采纳")
                    if state == "done":
                        if "result" not in value:
                            raise SharedError(502, "NR completed without a result")
                        return value["result"]
                    status = value.get("error_status", 409 if state == "canceled" else 500)
                    raise SharedError(status if type(status) is int and 400 <= status <= 599 else 500,
                                      value.get("error") or "NR processing failed")
                progress({"message": value.get("progress", ""), "ratio": value.get("ratio")})
                if cancel_event.wait(self.POLL_INTERVAL):
                    raise SharedError(409, "NR canceled")
                value = self._rpc("poll", ticket=ticket, timeout=self.WORK_RPC_TIMEOUT)
        except BaseException as error:
            if ticket is not None and not terminal:
                terminal = self._cancel(ticket, transport)
            unknown = submitted and not terminal
            # Caller owns the input/output directory. Unknown execution forbids its
            # deletion or reuse even after close/leave (leases cancel asynchronously).
            # Keep callback exception compatibility when terminal was confirmed.
            if isinstance(error, SharedError) or unknown:
                failure = SharedError(getattr(error, "status", 502), str(error))
                failure.execution_unknown = unknown
                failure.request_id = command.get("id") if isinstance(command, dict) else None
                failure.ticket = ticket
                if unknown:
                    failure.args = (str(error) + "; NR execution unconfirmed: retain input/output directory",)
                raise failure from error
            raise

    def stop(self):
        self.start()
        return self._rpc("release")

    def close(self):
        self._closed.set()
        with self._start_lock:
            if self.transport is None:
                return
            transport, self.transport = self.transport, None
            if self._client is not None:
                try:
                    transport.rpc({"op": "leave", "client": self._client}, timeout=1.)
                except Exception:
                    pass
            transport.close()
        if self._heartbeat_thread is not None and self._heartbeat_thread is not threading.current_thread():
            self._heartbeat_thread.join(.1)
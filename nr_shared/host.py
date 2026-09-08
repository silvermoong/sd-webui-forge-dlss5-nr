"""Owned NR worker with scoped receipts, served only over private stdio."""
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re
import secrets
import threading
import time
import uuid

from .contract import PROTOCOL, SharedError, validate_params, validate_runtime

TERMINAL = frozenset(("done", "error", "canceled"))


class Host:
    def __init__(self, root, worker, *, catalog=None, gate=None,
                 client_ttl=30., receipt_ttl=300., clock=time.monotonic):
        from .catalog import Catalog
        self.root = Path(root).resolve()
        self.temp_root = self.root / "tmp"
        self.worker = worker
        self.catalog = catalog or Catalog(root)
        self.gate = gate or (lambda *a, **kw: nullcontext())
        self.mode = "private"
        self.instance = uuid.uuid4().hex
        self.client_ttl, self.receipt_ttl, self.clock = client_ttl, receipt_ttl, clock
        self._cv = threading.Condition()
        self._clients, self._records = {}, OrderedDict()
        self._closed = False
        self._active = None
        self._thread = None
        self._monitor = None
        self._unconfirmed = False

    def _start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="nr-private-host")
            self._thread.start()
            self._monitor = threading.Thread(target=self._leases, daemon=True, name="nr-private-leases")
            self._monitor.start()

    def _leases(self):
        # The executor can be inside native evaluation; expiry must stay live.
        with self._cv:
            while not self._closed:
                self._cv.wait(.25)
                self._prune()

    def _prune(self):
        now = self.clock()
        expired = {client for client, row in self._clients.items() if row["seen"] + self.client_ttl < now}
        for client in expired:
            self._leave(client)
        for ticket, row in list(self._records.items()):
            if row["status"] in TERMINAL and row["at"] + self.receipt_ttl < now:
                del self._records[ticket]

    def _leave(self, client):
        self._clients.pop(client, None)
        for row in self._records.values():
            if row["client"] == client and row["status"] not in TERMINAL:
                row["event"].set()
                if row["status"] == "pending":
                    row.update(status="canceled", error="NR 客户端已离开", error_status=409)
        self._cv.notify_all()

    @staticmethod
    def _view(row):
        return deepcopy({key: row[key] for key in
            ("ticket", "status", "progress", "ratio", "result", "error", "error_status", "execution_unknown") if key in row})

    def _command(self, raw):
        if not isinstance(raw, dict):
            raise SharedError(400, "NR command 必须是对象")
        command = deepcopy(raw)
        operation = command.get("op")
        if (operation not in ("inspect", "run") or not isinstance(command.get("id"), str)
                or not 1 <= len(command["id"]) <= 128):
            raise SharedError(400, "NR 操作或请求身份无效")
        if operation == "inspect":
            if set(command) != {"op", "id", "runtime"} or not isinstance(command.get("runtime"), dict):
                raise SharedError(400, "未知 NR 枚举字段")
            return command
        command["runtime"] = validate_runtime(command.get("runtime"))
        required = {"op", "id", "source_path", "output_dir", "source", "params", "runtime",
                    "start", "end", "preview_kind", "max_edge"}
        if set(command) != required:
            raise SharedError(400, "NR 插件必须提交完整单图请求")
        if (command["start"] != 0 or command["end"] != 0 or command["preview_kind"] != ""
                or type(command["max_edge"]) is not int or command["max_edge"] != 0):
            raise SharedError(400, "此接口只执行原尺寸单图，不能提交预览或视频")
        command["params"] = validate_params(command["params"])
        source = command["source"]
        if (not isinstance(source, dict) or source.get("kind", "image") != "image"
                or not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])):
            raise SharedError(400, "NR 单图必须提供明确 SHA-256")
        self._paths(command)
        return command

    def _paths(self, command):
        """A request may only touch its own fixed plugin input/output names."""
        try:
            source, output = Path(command["source_path"]), Path(command["output_dir"])
            if (not source.is_absolute() or not output.is_absolute()
                    or ".." in source.parts or ".." in output.parts
                    or source.name != "input.png" or output.name != "output"
                    or source.parent != output.parent or source.parent.parent != self.temp_root
                    or not re.fullmatch(r"nr-[a-zA-Z0-9_-]{1,90}", source.parent.name)
                    or source.resolve(strict=True) != source or output.resolve(strict=True) != output
                    or not source.is_file() or not output.is_dir() or source.stat().st_nlink != 1
                    or source.stat().st_size > 128 * 1024 ** 2):
                raise ValueError()
            return source, output
        except (KeyError, TypeError, OSError, ValueError, RuntimeError):
            raise SharedError(400, "NR 仅允许本插件单次临时目录内的 input.png/output，拒绝越界或链接") from None

    def _source(self, command, event):
        source, output = self._paths(command)
        if any(output.iterdir()):
            raise SharedError(409, "NR 输出目录不为空，拒绝覆盖")
        before = source.stat()
        hasher = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if event.is_set():
                    raise SharedError(409, "NR 已取消")
                hasher.update(chunk)
        after = source.stat()
        stamp = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stamp(before) != stamp(after) or hasher.hexdigest() != command["source"]["sha256"]:
            raise SharedError(409, "NR 源图身份/字节在提交后变化")
        return stamp(after)

    def dispatch(self, body):
        try:
            return self._dispatch(body)
        except (ValueError, TypeError, KeyError, RecursionError):
            raise SharedError(400, "NR请求参数无效或不完整") from None

    def _dispatch(self, body):
        if not isinstance(body, dict):
            raise SharedError(400, "NR RPC必须是对象")
        op = body.get("op")
        fields = {"hello": {"op"}, "join": {"op", "label"}, "heartbeat": {"op", "client"},
            "status": {"op", "client"}, "runtime": {"op", "client", "device"},
            "submit": {"op", "client", "command"}, "poll": {"op", "client", "ticket"},
            "cancel": {"op", "client", "ticket"}, "release": {"op", "client"}, "leave": {"op", "client"}}
        if op not in fields or set(body) != fields[op]:
            raise SharedError(400, "未知 NR RPC 操作/字段")
        with self._cv:
            if self._closed:
                raise SharedError(503, "NR 提供方已关闭")
            self._prune()
            if op == "hello":
                return dict(protocol=PROTOCOL, instance=self.instance, mode=self.mode)
            if op == "join":
                label = body["label"]
                if not isinstance(label, str) or not 1 <= len(label) <= 80 or len(self._clients) >= 16:
                    raise SharedError(429, "NR 客户端名称无效或连接过多")
                client = secrets.token_hex(32)
                self._clients[client] = {"label": label, "seen": self.clock()}
                self._start()
                return dict(client=client, instance=self.instance, mode=self.mode)
            client = body.get("client")
            if not isinstance(client, str) or client not in self._clients:
                raise SharedError(403, "NR 客户端未登记或租约已过期")
            self._clients[client]["seen"] = self.clock()
            if op == "heartbeat":
                return {"ok": True}
            if op == "leave":
                self._leave(client)
                return {"closed": True}
            if op in ("poll", "cancel"):
                row = self._records.get(body["ticket"]) if isinstance(body["ticket"], str) else None
                if row is None or row["client"] != client:
                    raise SharedError(404, "没有属于本客户端的NR票据")
                if op == "cancel" and row["status"] not in TERMINAL:
                    row["event"].set()
                    if row["status"] == "pending":
                        row.update(status="canceled", error="NR 已取消", error_status=409)
                return self._view(row)
            if op == "submit":
                if self._unconfirmed:
                    raise SharedError(503, "NR上次执行未确认停止，拒绝启动另一请求；请检查提供方进程")
                command = self._command(body["command"])
                if any(row["client"] == client and row["command"]["id"] == command["id"]
                       for row in self._records.values()):
                    raise SharedError(409, "NR 请求身份已使用，禁止不明接收状态下重复执行")
                if sum(row["status"] not in TERMINAL for row in self._records.values()) >= 16:
                    raise SharedError(429, "NR 执行等待过多，请稍后提交")
                while len(self._records) >= 64:
                    oldest = next((ticket for ticket, row in self._records.items() if row["status"] in TERMINAL), None)
                    if oldest is None:
                        raise SharedError(429, "NR 收据暂满")
                    del self._records[oldest]
                ticket = uuid.uuid4().hex
                row = dict(ticket=ticket, client=client, command=command, event=threading.Event(),
                           at=self.clock(), status="pending", progress="等待NR执行器", ratio=None)
                self._records[ticket] = row
                self._cv.notify_all()
                return self._view(row)
            if op == "release" and any(row["status"] not in TERMINAL for row in self._records.values()):
                raise SharedError(409, "NR 正在处理或等待；先取消自己的任务，不能释放他人的任务")
        # No state mutex around IO, filesystem hashing or process lifetime calls.
        if op == "runtime":
            return self.catalog.runtime(body["device"])
        if op == "status":
            value = self.worker.status()
            with self._cv:
                active = self._active is not None
                clients = len(self._clients)
            return {**value, "busy": bool(value.get("busy") or active), "worker_pid": value.get("pid"),
                    "instance": self.instance, "mode": self.mode, "clients": clients}
        if op == "release":
            with self.gate("release", threading.Event(), wait=0):
                with self._cv:
                    if any(row["status"] not in TERMINAL for row in self._records.values()):
                        raise SharedError(409, "NR 已有新任务，未释放")
                return {"released": bool(self.worker.stop())}
        raise SharedError(400, "NR 操作未实现")

    def _execute(self, row):
        event, ticket, command = row["event"], row["ticket"], deepcopy(row["command"])
        command["id"] = ticket

        def progress(message, ratio=None):
            with self._cv:
                row["progress"] = str(message)[:400]
                row["ratio"] = ratio if type(ratio) in (float, int) and math.isfinite(ratio) and 0 <= ratio <= 1 else None

        with self.gate(ticket, event, wait=30):
            if event.is_set():
                raise SharedError(409, "NR 已取消")
            if command["op"] == "inspect":
                runtime = self.catalog.inspection(command["runtime"])
                inspection = self.worker.inspect(runtime, cancel_event=event)
                self.catalog.remember(inspection)
                return inspection
            self.catalog.check(command["runtime"])
            inspection = self.worker.inspect(command["runtime"], cancel_event=event)
            self.catalog.remember(inspection)
            from nr_runtime.manager import verify_device
            verify_device(command["runtime"], inspection)
            self.catalog.check(command["runtime"])
            stamp = self._source(command, event)
            result = self.worker.run(command, progress, event)
            if event.is_set():
                raise SharedError(409, "NR 已取消，结果未采纳")
            self.catalog.check(command["runtime"])
            source, _ = self._paths(command)
            after = source.stat()
            if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise SharedError(409, "NR 源图在处理期间变化，结果未采纳")
            if not isinstance(result, dict):
                raise SharedError(502, "NR 未交回完整结果")
            return {**result, "instance": self.instance, "worker_pid": self.worker.status().get("pid")}

    def _loop(self):
        while True:
            with self._cv:
                self._prune()
                if self._closed:
                    return
                row = next((r for r in self._records.values() if r["status"] == "pending"), None)
                if row is None:
                    self._cv.wait(.5)
                    continue
                self._active = row["ticket"]
                row["status"] = "running"
            try:
                result = self._execute(row)
                with self._cv:
                    if row["event"].is_set():
                        row.update(status="canceled", error="NR 已取消", error_status=409)
                    else:
                        row.update(status="done", result=result, ratio=1.)
            except Exception as error:
                with self._cv:
                    canceled = row["event"].is_set()
                    status = getattr(error, "status", 502)
                    if getattr(error, "execution_unknown", False):
                        self._unconfirmed = True
                        for other in self._records.values():
                            if other["status"] == "pending":
                                other["event"].set()
                                other.update(status="canceled", error="前一NR执行未确认停止，未启动本请求", error_status=409)
                    row.update(status="canceled" if canceled else "error", error_status=409 if canceled else status,
                               error="NR 已取消" if canceled else str(error)[:1000],
                               execution_unknown=bool(getattr(error, "execution_unknown", False)))
            finally:
                with self._cv:
                    self._active = None
                    row["at"] = self.clock()
                    self._cv.notify_all()

    def close(self):
        with self._cv:
            if self._closed:
                return
            self._closed = True
            for client in list(self._clients):
                self._leave(client)
            self._cv.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(5)
        if self._monitor is not None and self._monitor is not threading.current_thread():
            self._monitor.join(1)
        self.worker.close()
"""Private startup and process ownership checks; no network or GPU."""
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nr_shared.client import Client
from nr_shared.contract import PROTOCOL, SharedError


class Transport:
    def __init__(self, mode="private", failure=None):
        self.mode = mode
        self.instance = "cpu-only"
        self.failure = failure
        self.calls = []
        self.closed = False

    def rpc(self, body, timeout=None):
        self.calls.append(body["op"])
        if self.failure is not None:
            raise self.failure
        if body["op"] == "hello":
            return dict(protocol=PROTOCOL, instance=self.instance, mode=self.mode)
        if body["op"] == "join":
            return dict(client="a" * 64, instance=self.instance, mode=self.mode)
        return dict(ok=True)

    def close(self):
        self.closed = True


class ClientTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows private process ownership")
    def test_real_private_controller_starts_without_a_model_and_closes(self):
        from nr_shared.transports import PrivateTransport
        entry = """import sys
def guard(event, args):
    if event in ('socket.connect', 'socket.bind'):
        raise AssertionError('No network in private startup')
    if event == 'ctypes.dlopen' and any(part in str(args[0]).lower() for part in ('nvcuda', 'nvml', 'dlss', 'd3d12')):
        raise AssertionError('No GPU libraries in private startup')
sys.addaudithook(guard)
from nr_shared.private import main
main()
"""
        transport = PrivateTransport(ROOT, command=[sys.executable, "-u", "-B", "-c", entry])
        process = transport._process
        client = Client(ROOT, transport=transport)
        try:
            client.start()
            self.assertEqual(client.mode, "private")
            self.assertFalse(client.status()["running"])
            runtime = client.runtime()
            self.assertFalse(runtime["ready"])
            self.assertTrue(runtime["missing"])
            self.assertFalse(client.status()["pid"])
            self.assertIsNone(process.poll())
        finally:
            client.close()
        self.assertIsNotNone(process.poll())

    def test_only_private_transport_is_available(self):
        from nr_shared import transports
        self.assertFalse(hasattr(transports, "HttpTransport"))
        self.assertFalse(hasattr(transports, "Offline"))
        self.assertFalse((ROOT / "nr_shared/bridge.py").exists())

    def test_host_always_owns_its_worker_and_extension_temp_directory(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from nr_shared.host import Host
        worker = SimpleNamespace(close=Mock())
        host = Host(ROOT, worker, catalog=object())
        self.assertEqual(host.mode, "private")
        self.assertEqual(host.temp_root, ROOT / "tmp")
        host.close()
        host.close()
        worker.close.assert_called_once_with()

    def test_starts_only_its_private_transport_without_discovery_or_network(self):
        transport = Transport()
        with patch("nr_shared.client.PrivateTransport", return_value=transport) as factory, \
                patch.object(Path, "open", side_effect=AssertionError("No discovery files")), \
                patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
            client = Client(ROOT)
            try:
                self.assertIs(client.start(), client)
                factory.assert_called_once_with(ROOT)
                self.assertEqual(client.mode, "private")
                self.assertEqual(client.execution_root, ROOT)
                self.assertEqual(transport.calls, ["hello", "join"])
            finally:
                client.close()
        self.assertTrue(transport.closed)

    def test_rejects_nonprivate_handshake_before_join(self):
        transport = Transport(mode="borrowed")
        with self.assertRaises(SharedError):
            Client(ROOT, transport=transport).start()
        self.assertEqual(transport.calls, ["hello"])
        self.assertTrue(transport.closed)

    def test_failed_private_start_does_not_select_another_provider(self):
        transport = Transport(failure=SharedError(503, "Private startup failed"))
        with patch("nr_shared.client.PrivateTransport", return_value=transport) as factory:
            client = Client(ROOT)
            try:
                for attempt in range(2):
                    with self.subTest(attempt=attempt), self.assertRaisesRegex(SharedError, "Private startup failed"):
                        client.start()
                factory.assert_called_once_with(ROOT)
                self.assertEqual(transport.calls, ["hello"])
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
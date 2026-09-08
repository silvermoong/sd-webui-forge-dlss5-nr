"""Dependency maintenance only closes an idle, owned NR controller."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forge_nr.lifecycle import RepairBlocked, Service


class LifecycleTests(unittest.TestCase):
    def service(self):
        client = SimpleNamespace(start=Mock(), close=Mock(), status=Mock(return_value={"busy": False}), run=Mock(), inspect=Mock())
        service = Service(ROOT, client_factory=lambda **kwargs: client)
        service.start()
        return service, client

    def test_idle_controller_closes_before_repair_and_restarts_only_afterward(self):
        service, client = self.service()
        def repaired():
            client.close.assert_called_once_with()
            self.assertFalse(service.connection_info()["connected"])
            with self.assertRaises(RepairBlocked):
                service.run({}, lambda value: None, None)
            with self.assertRaises(RepairBlocked):
                service.start()
            return "repaired"
        self.assertEqual(service.repair(repaired), "repaired")
        client.run.assert_not_called()
        service.start()
        self.assertTrue(service.connection_info()["connected"])
        self.assertEqual(client.start.call_count, 2)

    def test_active_run_and_inspection_block_repair_without_closing_anything(self):
        for operation in ("run", "inspect"):
            with self.subTest(operation=operation):
                service, client = self.service()
                repair = Mock()
                def active(*args, **kwargs):
                    with self.assertRaises(RepairBlocked):
                        service.repair(repair)
                    client.close.assert_not_called()
                getattr(client, operation).side_effect = active
                if operation == "run":
                    service.run({}, lambda value: None, None)
                else:
                    service.inspect({})
                repair.assert_not_called()
                service.repair(repair)
                repair.assert_called_once_with()

    def test_busy_or_unknown_remote_state_blocks_dependency_installation(self):
        for status in ({"busy": True}, {}, None, RuntimeError("status unavailable")):
            with self.subTest(status=status):
                service, client = self.service()
                if isinstance(status, Exception):
                    client.status.side_effect = status
                else:
                    client.status.return_value = status
                repair = Mock()
                with self.assertRaises(RepairBlocked):
                    service.repair(repair)
                repair.assert_not_called()
                client.close.assert_not_called()
                self.assertTrue(service.connection_info()["connected"])

    def test_failed_repair_releases_the_gate_for_an_explicit_retry(self):
        service, client = self.service()
        with self.assertRaisesRegex(RuntimeError, "installer failed"):
            service.repair(Mock(side_effect=RuntimeError("installer failed")))
        self.assertFalse(service.connection_info()["connected"])
        second = Mock(return_value="fixed")
        self.assertEqual(service.repair(second), "fixed")
        second.assert_called_once_with()
        self.assertEqual(client.close.call_count, 1)


if __name__ == "__main__":
    unittest.main()
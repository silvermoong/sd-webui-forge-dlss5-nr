"""User-managed runtime selection; no GPU, network or executable fixture code."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nr_runtime import settings
from nr_runtime.download import DownloadError
from nr_runtime.setup import RuntimeSetup


def manual_dll(marker=0):
    data = bytearray(2048)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<H", data, 132, 0x8664)
    struct.pack_into("<H", data, 150, 0x2000)
    struct.pack_into("<H", data, 152, 0x20b)
    data[-1] = marker
    return bytes(data)


class ManualRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.extension = self.root / "extension"
        self.models = self.root / "forge-models"
        self.enterContext(patch.object(settings, "ROOT", self.extension))
        self.enterContext(patch.object(settings, "CONFIG_PATH", self.extension / "runtime-settings.json"))
        helper = "models/dlssnr/caller/nvngx.dll_comfy.dll"
        for relative in (helper, ".venv/Scripts/python.exe", "native/nr/bin/dlss5nr_bridge.dll"):
            path = self.extension / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"CPU fixture")
        (self.extension / "native-artifacts.json").write_text(json.dumps({"files": {helper: hashlib.sha256(b"CPU fixture").hexdigest()}}))
        (self.extension / "runtime-download.json").write_text(json.dumps({"url": "https://example.invalid/original", "member": None}))
        self.verifier = Mock(side_effect=ValueError("Not NVIDIA-signed"))
        self.setup = RuntimeSetup(self.extension, verifier=self.verifier, models_dir=self.models)

    def source(self, marker=0):
        path = self.root / f"selected-{marker}/nvngx_dlssnr.dll"
        path.parent.mkdir()
        path.write_bytes(manual_dll(marker))
        return path

    def test_manual_missing_file_never_falls_back_to_a_download(self):
        self.setup.set_mode("manual")
        with patch("nr_runtime.setup.install_runtime") as download:
            with self.assertRaises(DownloadError) as caught:
                self.setup.ensure()
        self.assertEqual(caught.exception.code, "manual_missing")
        download.assert_not_called()
        self.verifier.assert_not_called()
        self.assertEqual(self.setup.runtime_dir, self.models / "DLSS-NR/manual")

    def test_manual_import_survives_reload_without_signature_gate_or_overwrite(self):
        source = self.source()
        original = source.read_bytes()
        self.setup.set_mode("manual")
        result = self.setup.import_runtime(source)
        self.assertIn("NVIDIA signature not verified", result)
        reloaded = RuntimeSetup(self.extension, verifier=self.verifier, models_dir=self.models)
        with patch("nr_runtime.setup.install_runtime") as download:
            target = reloaded.ensure()
        self.assertEqual(reloaded.mode, "manual")
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(source.read_bytes(), original)
        self.verifier.assert_not_called()
        download.assert_not_called()
        self.assertEqual(settings.get("nr")["runtime_dir"], str(target.parent))

    def test_automatic_mode_keeps_signature_checks_and_its_separate_file(self):
        original_dir = self.models / "DLSS-NR"
        original_dir.mkdir(parents=True)
        original = original_dir / "nvngx_dlssnr.dll"
        original.write_bytes(manual_dll(1))
        with self.assertRaisesRegex(ValueError, "Not NVIDIA-signed"):
            self.setup.ensure()
        self.setup.set_mode("manual")
        self.setup.import_runtime(self.source(2))
        manual = self.setup.ensure()
        self.setup.set_mode("auto")
        self.verifier.side_effect = None
        with patch("nr_runtime.setup.install_runtime") as download:
            self.assertEqual(self.setup.ensure(), original)
        download.assert_not_called()
        self.assertEqual(manual.read_bytes(), manual_dll(2))
        self.assertEqual(original.read_bytes(), manual_dll(1))
        self.verifier.assert_called_with(original)

    def test_manual_files_can_be_installed_directly_and_are_preserved(self):
        self.setup.set_mode("manual")
        self.setup.runtime_dir.mkdir(parents=True)
        target = self.setup.runtime_dir / "nvngx_dlssnr.dll"
        target.write_bytes(manual_dll(3))
        with patch("nr_runtime.setup.install_runtime") as download:
            self.assertEqual(self.setup.ensure(), target)
        download.assert_not_called()
        with self.assertRaisesRegex(ValueError, "existing different runtime"):
            self.setup.import_runtime(self.source(4))
        self.assertEqual(target.read_bytes(), manual_dll(3))

    def test_invalid_manual_binary_never_becomes_installed(self):
        self.setup.set_mode("manual")
        source = self.source()
        source.write_bytes(b"not executable" * 200)
        with self.assertRaises(DownloadError) as caught:
            self.setup.import_runtime(source)
        self.assertEqual(caught.exception.code, "manual_invalid")
        self.assertFalse((self.setup.runtime_dir / "nvngx_dlssnr.dll").exists())
        self.assertFalse(list(self.setup.runtime_dir.glob("runtime-import-*")))

    def test_unknown_mode_and_switch_during_preparation_fail_closed(self):
        with self.assertRaises(DownloadError):
            self.setup.set_mode("guess-gpu")
        with self.setup._preparing:
            with self.assertRaises(DownloadError):
                self.setup.set_mode("manual")
        self.assertEqual(self.setup.mode, "auto")
        self.assertFalse(self.setup.selection_path.exists())


if __name__ == "__main__":
    unittest.main()
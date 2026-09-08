"""No GPU or proprietary binaries; fixtures are CPU-only inputs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nr_shared.catalog import Catalog
from nr_shared.contract import SharedError
from nr_runtime import settings
from nr_runtime.manager import NRConfig
from nr_runtime.setup import RuntimeSetup, signature
from nr_runtime.download import DownloadError
from forge_nr.i18n import CATALOGS, preparation_message, text


class RuntimeTests(unittest.TestCase):
    def test_project_and_plugin_names_are_dlss5_nr(self):
        from configparser import ConfigParser
        from forge_nr import VERSION
        from forge_nr.xyz import ENABLED_LABEL, PRESET_LABEL
        from nr_shared.contract import SCRIPT_TITLE
        metadata = ConfigParser()
        metadata.read(ROOT / "metadata.ini", encoding="utf-8")
        bundle = json.loads((ROOT / "PUBLIC_BUNDLE.json").read_text(encoding="utf-8"))
        self.assertEqual(bundle["project"], "sd-webui-forge-dlss5-nr")
        self.assertRegex(VERSION, r"^\d+\.\d+\.\d+$")
        self.assertEqual(bundle["version"], VERSION)
        self.assertEqual(metadata["Extension"]["Name"], "DLSS5 NR for Forge")
        self.assertEqual(SCRIPT_TITLE, "DLSS5 NR")
        self.assertEqual(ENABLED_LABEL, "[DLSS5 NR] Enabled")
        self.assertEqual(PRESET_LABEL, "[DLSS5 NR] Preset")
        for language in ("en", "zh"):
            self.assertIn(SCRIPT_TITLE, text("prepared", language))
        release_prefix = "https://github.com/silvermoong/sd-webui-forge-dlss5-nr/releases/download/"
        artifacts = json.loads((ROOT / "native-artifacts.json").read_text(encoding="utf-8"))
        self.assertEqual(artifacts["version"], VERSION)
        self.assertEqual(artifacts["url"], f"{release_prefix}v{VERSION}/nr-bridge-windows-x64.zip")
        self.assertIn(release_prefix, (ROOT / "install.py").read_text(encoding="utf-8"))
        for filename in ("README.md", "README.zh-CN.md", "MODEL_SETUP.md", "MODEL_SETUP.zh-CN.md", "docs/usage.zh-CN.md"):
            document = (ROOT / filename).read_text(encoding="utf-8")
            self.assertTrue(document.startswith("# DLSS5 NR"), filename)
            self.assertNotIn("sd-webui-forge-dlss-nr", document)
            self.assertNotIn("DLSS NR", document)
        for filename in ("README.md", "README.zh-CN.md"):
            document = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("https://github.com/silvermoong/sd-webui-forge-dlss5-nr.git", document)

    def test_missing_dependencies_do_not_start_an_installer_during_prepare(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            bootstrapper = Mock()
            setup = RuntimeSetup(directory, bootstrapper=bootstrapper)
            with self.assertRaises(DownloadError) as caught:
                setup.ensure()
            self.assertEqual(caught.exception.code, "dependencies_missing")
            bootstrapper.assert_not_called()

    def test_dependency_repair_is_explicit_and_keeps_command_out_of_summary(self):
        import subprocess
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            failure = subprocess.CalledProcessError(1, ["python", "-m", "pip", "install", "fixture"])
            bootstrapper = Mock(side_effect=failure)
            setup = RuntimeSetup(directory, bootstrapper=bootstrapper)
            with self.assertRaises(DownloadError) as caught:
                setup.ensure(repair=True)
            bootstrapper.assert_called_once()
            self.assertEqual(caught.exception.code, "dependencies_failed")
            self.assertNotIn("pip", str(caught.exception))
            self.assertIs(caught.exception.__cause__, failure)

    @unittest.skipUnless(os.name == "nt", "Windows Authenticode verifier")
    def test_signature_verifier_ignores_inherited_foreign_modules(self):
        powershell = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            modules = Path(directory)
            foreign = modules / "Microsoft.PowerShell.Security"
            foreign.mkdir()
            (foreign / "Microsoft.PowerShell.Security.psm1").write_text(
                "function Get-AuthenticodeSignature { throw 'Foreign security module was invoked' }\n"
                "Export-ModuleMember -Function Get-AuthenticodeSignature\n", encoding="utf-8")
            with patch.dict(os.environ, {"PSModulePath": str(modules)}):
                with self.assertRaisesRegex(ValueError, "valid NVIDIA Authenticode signature is required"):
                    signature(powershell)

    def test_complete_bilingual_catalog(self):
        self.assertEqual(set(CATALOGS["en"]), set(CATALOGS["zh"]))
        self.assertEqual(text("stage"), "Insertion point")
        self.assertEqual(text("stage", "zh"), "插入时机")

    def test_recovery_instructions_match_the_actual_button_labels(self):
        for language, suffix in (("en", ""), ("zh", ".zh-CN")):
            for basename in ("README", "MODEL_SETUP"):
                document = (ROOT / f"{basename}{suffix}.md").read_text(encoding="utf-8")
                with self.subTest(language=language, document=basename):
                    for key in ("prepare", "manual_setup", "repair_dependencies"):
                        self.assertIn(text(key, language), document)

    def test_documented_source_and_modes_match_the_runtime_configuration(self):
        from nr_runtime.download import read_source
        source = read_source(ROOT / "runtime-download.json")
        for language, suffix in (("en", ""), ("zh", ".zh-CN")):
            for basename in ("README", "MODEL_SETUP"):
                document = (ROOT / f"{basename}{suffix}.md").read_text(encoding="utf-8")
                with self.subTest(language=language, document=basename):
                    self.assertIn(text("runtime_auto", language), document)
                    self.assertIn(text("runtime_manual", language), document)
                    self.assertIn("DLSS-NR/manual", document)
                    if basename == "MODEL_SETUP":
                        self.assertIn(source["url"], document)
                        self.assertIn(source["member"], document)

    def test_preparation_messages_select_the_matching_recovery_action(self):
        from urllib.error import HTTPError, URLError
        errors = [(DownloadError("dependencies_missing", "raw command"), "修复依赖", "Repair dependencies"),
                  (DownloadError("dependencies_failed", "pip install raw command"), "依赖修复未完成", "Dependency repair failed"),
                  (URLError("raw command"), "重试运行库", "Retry runtime"),
                  (HTTPError("https://example.invalid", 404, "raw command", {}, None), "维护者", "maintainer")]
        for error, chinese, english in errors:
            with self.subTest(error=type(error).__name__):
                self.assertIn(chinese, preparation_message(error, "zh"))
                self.assertIn(english, preparation_message(error, "en"))
                self.assertNotIn("raw command", preparation_message(error, "zh"))
                self.assertNotIn("raw command", preparation_message(error, "en"))

    def test_initial_device_enumeration_does_not_require_a_model_or_guessed_gpu(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            cfg = NRConfig().model_dump()
            cfg["bridge"], cfg["runtime_dir"] = str(root / "bridge.dll"), str(root / "models")

            def runtime_files(config, force=False):
                runtime = {key: config[key] for key in ("bridge", "runtime_dir", "gpu_index", "device", "gpu_name", "channel_order")}
                runtime["runtime_id"] = "a" * 64
                return dict(ready=bool(config["gpu_name"]), missing=[], runtime_id="a" * 64, runtime=runtime)

            catalog = Catalog(root, config_provider=lambda *a, **kw: cfg, runtime_provider=runtime_files)
            initial = catalog.runtime()
            self.assertFalse(initial["ready"])
            self.assertEqual(initial["runtime"]["gpu_name"], "")
            self.assertEqual(catalog.inspection(initial["runtime"])["bridge"], cfg["bridge"])
            catalog.remember({"devices": [{"gpu_index": 1, "device": "cuda:0", "gpu_name": "CPU fixture", "luid": "fixture"}]})
            selected = catalog.runtime("cuda:0")
            self.assertTrue(selected["ready"])
            self.assertEqual(selected["runtime"]["gpu_index"], 1)
            self.assertEqual(selected["runtime"]["gpu_name"], "CPU fixture")
            with self.assertRaises(SharedError):
                catalog.inspection({**initial["runtime"], "bridge": str(root / "other.dll")})

    @unittest.skipUnless(os.name == "nt", "Windows runtime import path")
    def test_runtime_import_requires_signature_and_matching_immutable_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            source = root / "licensed-source/nvngx_dlssnr.dll"
            source.parent.mkdir()
            source.write_bytes(b"CPU TEST ONLY " * 128)
            original = source.read_bytes()
            installer = RuntimeSetup(root / "extension", verifier=lambda path: {"status": "Valid", "subject": "NVIDIA Corporation (TEST ONLY)"})
            with patch.object(installer, "verifier", side_effect=ValueError("Invalid NVIDIA signature")):
                with self.assertRaisesRegex(ValueError, "Invalid NVIDIA signature"):
                    installer.import_runtime(source)
            self.assertFalse((installer.root / "models/dlssnr/nvngx_dlssnr.dll").exists())
            result = installer.import_runtime(source)
            self.assertIn("Original file unchanged", result)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual((installer.root / "models/dlssnr/nvngx_dlssnr.dll").read_bytes(), original)
            self.assertFalse(list(installer.root.rglob("runtime-import-*")))
            revised = original + b"ANOTHER SIGNED VERSION"
            source.write_bytes(revised)
            with self.assertRaisesRegex(ValueError, "existing different runtime was preserved"):
                installer.import_runtime(source)
            self.assertEqual((installer.root / "models/dlssnr/nvngx_dlssnr.dll").read_bytes(), original)
            another = RuntimeSetup(root / "another-extension", verifier=installer.verifier)
            another.import_runtime(source)
            self.assertEqual((another.root / "models/dlssnr/nvngx_dlssnr.dll").read_bytes(), revised)
            self.assertEqual(source.read_bytes(), revised)

    def test_private_device_choice_is_explicit_environment_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            with patch.object(settings, "ROOT", root), patch.object(settings, "CONFIG_PATH", root / "runtime-settings.json"):
                before = settings.get("nr")
                RuntimeSetup(root).select_device({"gpu_index": 2, "device": "cuda:1", "gpu_name": "CPU fixture"})
                saved = settings.get("nr")
                self.assertEqual(saved["device"], "cuda:1")
                self.assertEqual(saved["bridge"], before["bridge"])
                self.assertEqual(saved["runtime_dir"], before["runtime_dir"])
                self.assertNotIn("params", json.loads((root / "runtime-settings.json").read_text()))

    def test_models_directory_is_supplied_by_forge_not_the_extension_name(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            models = root / "custom-forge-models"
            with patch.object(settings, "ROOT", root), patch.object(settings, "CONFIG_PATH", root / "runtime-settings.json"):
                setup = RuntimeSetup(root / "arbitrary-extension-name", models_dir=models)
                setup.configure()
                self.assertEqual(setup.runtime_dir, models / "DLSS-NR")
                self.assertEqual(settings.get("nr")["runtime_dir"], str(models / "DLSS-NR"))
                self.assertFalse((models / "DLSS-NR/nvngx_dlssnr.dll").exists())

    def test_preparation_installs_under_forge_models_and_reuses_existing_runtime(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            extension, models = root / "extension", root / "forge/models"
            for name in (".venv/Scripts/python.exe", "native/nr/bin/dlss5nr_bridge.dll", "models/dlssnr/caller/nvngx.dll_comfy.dll"):
                target = extension / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"CPU HELPER FIXTURE")
            helper = "models/dlssnr/caller/nvngx.dll_comfy.dll"
            (extension / "native-artifacts.json").write_text(json.dumps({"files": {helper: hashlib.sha256(b"CPU HELPER FIXTURE").hexdigest()}}))
            (extension / "runtime-download.json").write_text(json.dumps(dict(url="https://example.invalid/runtime", member=None)))
            with patch.object(settings, "ROOT", extension), patch.object(settings, "CONFIG_PATH", extension / "runtime-settings.json"):
                setup = RuntimeSetup(extension, verifier=Mock(), models_dir=models)
                def downloaded(source, target, verifier, progress):
                    target.write_bytes(b"CPU RUNTIME FIXTURE")
                    verifier(target)
                with patch("nr_runtime.setup.install_runtime", side_effect=downloaded) as download:
                    first = setup.ensure()
                    second = setup.ensure()
                self.assertEqual(first, models / "DLSS-NR/nvngx_dlssnr.dll")
                self.assertEqual(second, first)
                download.assert_called_once()
                self.assertEqual((models / "DLSS-NR/caller/nvngx.dll_comfy.dll").read_bytes(), b"CPU HELPER FIXTURE")
                self.assertFalse((extension / "models/dlssnr/nvngx_dlssnr.dll").exists())

    def test_package_is_self_contained(self):
        self.assertNotIn("core", sys.modules)
        self.assertNotIn("server", sys.modules)
        self.assertFalse((ROOT / "config.json").exists())

    def test_no_external_product_in_executable_code_or_user_guides(self):
        paths = [path for folder in ("forge_nr", "scripts", "nr_runtime", "nr_shared")
                 for path in (ROOT / folder).glob("*.py")]
        paths.extend(ROOT / name for name in ("README.md", "README.zh-CN.md", "MODEL_SETUP.md", "MODEL_SETUP.zh-CN.md", "metadata.ini"))
        for path in paths:
            with self.subTest(path=path.name):
                content = path.read_text(encoding="utf-8").lower()
                if path.name in ("manager.py", "nr_native.py"):
                    for symbol in ("tagsystem_nr_devices_json", "tagsystem_nr_gpu_luid", "tagsystem_nr_reset_feature"):
                        content = content.replace(symbol, "")
                self.assertNotIn("tagsystem", content)

    def test_runtime_file_checks_preserve_installed_native_bridge_abi(self):
        from nr_runtime.manager import RuntimeFiles
        bridge = ROOT / "native/nr/bin/dlss5nr_bridge.dll"
        if not bridge.is_file():
            self.skipTest("Optional installed MIT native bridge")
        config = settings.get("nr")
        result = RuntimeFiles()(config, force=True)
        self.assertFalse(any("物理设备核对接口" in message for message in result["missing"]), result["missing"])


if __name__ == "__main__":
    unittest.main()
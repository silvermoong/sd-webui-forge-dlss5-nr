"""CPU-only runtime download checks, with explicit synthetic NVIDIA verification."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nr_runtime.download import DownloadError, install_runtime, read_source


class Response(io.BytesIO):
    def __init__(self, payload, length=None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}

    def geturl(self):
        return "https://downloads.example.invalid/runtime"


class DownloadTests(unittest.TestCase):
    def test_configured_download_selects_only_the_original_runtime(self):
        source = read_source(ROOT / "runtime-download.json")
        self.assertEqual(source["url"], "https://github.com/SAOG0721/Magpie/releases/download/v0.6.6-experimental/DLSSNR-DLL-Options-310.8.0.0.zip")
        self.assertEqual(source["member"], "NVIDIA-Original/nvngx_dlssnr.dll")
        self.assertEqual(set(source), {"url", "member"})

    def test_direct_download_and_alternative_signed_bytes_have_no_hash_lock(self):
        for payload in (b"CPU FIXTURE " * 128, b"OTHER SIGNED BUILD " * 128):
            with self.subTest(size=len(payload)), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                target = Path(directory) / "nvngx_dlssnr.dll"
                verifier = Mock()
                progress = Mock()
                install_runtime(dict(url="https://downloads.example.invalid/runtime", member=None), target,
                                verifier, progress, opener=lambda *args, **kwargs: Response(payload))
                self.assertEqual(target.read_bytes(), payload)
                verifier.assert_called_once()
                self.assertFalse(list(target.parent.glob("download-*")))
                progress.assert_called()

    def test_failed_signature_and_interrupted_download_publish_nothing(self):
        for wrong_length in (False, True):
            with self.subTest(wrong_length=wrong_length), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                target = Path(directory) / "nvngx_dlssnr.dll"
                payload = b"CPU FIXTURE " * 128
                verifier = Mock(side_effect=ValueError("Invalid signature"))
                with self.assertRaises((ValueError, DownloadError)):
                    install_runtime(dict(url="https://downloads.example.invalid/runtime", member=None), target, verifier,
                                    opener=lambda *args, **kwargs: Response(payload, len(payload) + int(wrong_length)))
                self.assertFalse(target.exists())
                self.assertFalse(list(target.parent.glob("download-*")))

    def test_existing_file_is_not_overwritten_or_downloaded(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory) / "nvngx_dlssnr.dll"
            target.write_bytes(b"preserve")
            opener = Mock()
            with self.assertRaises(DownloadError):
                install_runtime(dict(url="https://downloads.example.invalid/runtime", member=None), target, Mock(), opener=opener)
            opener.assert_not_called()
            self.assertEqual(target.read_bytes(), b"preserve")

    def test_archive_extracts_only_declared_runtime(self):
        payload = b"CPU FIXTURE " * 128
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("package/nvngx_dlssnr.dll", payload)
            bundle.writestr("../../ignored.txt", b"never extracted")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory) / "nvngx_dlssnr.dll"
            install_runtime(dict(url="https://downloads.example.invalid/runtime.zip", member="package/nvngx_dlssnr.dll"),
                            target, Mock(), opener=lambda *args, **kwargs: Response(archive.getvalue()))
            self.assertEqual(list(target.parent.iterdir()), [target])
            self.assertEqual(target.read_bytes(), payload)

    def test_missing_source_and_unsafe_urls_fail_before_network(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "runtime-download.json"
            for url in (None, "http://example.invalid/file", "https://user:secret@example.invalid/file"):
                path.write_text(json.dumps(dict(url=url, member=None)), encoding="utf-8")
                with self.subTest(url=url), self.assertRaises(DownloadError):
                    read_source(path)


if __name__ == "__main__":
    unittest.main()
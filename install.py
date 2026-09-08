"""Forge installer: isolated small Python environment plus our MIT-only bridge.

No NVIDIA model/runtime download and no packages installed into Forge's environment.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_native():
    metadata = json.loads((ROOT / "native-artifacts.json").read_text(encoding="utf-8"))
    if all((ROOT / path).is_file() and digest(ROOT / path) == value for path, value in metadata["files"].items()):
        return
    url = metadata["url"]
    if not url.startswith("https://github.com/silvermoong/sd-webui-forge-dlss5-nr/releases/download/"):
        raise ValueError("Unexpected bridge release source")
    with tempfile.TemporaryDirectory(prefix="nr-native-install-", dir=ROOT) as temporary:
        archive = Path(temporary) / "bridge.zip"
        request = urllib.request.Request(url, headers={"User-Agent": "sd-webui-forge-dlss5-nr-installer"})
        with urllib.request.urlopen(request, timeout=45) as response, archive.open("wb") as output:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > metadata["size"]:
                    raise ValueError("Bridge archive exceeds the pinned size")
                output.write(chunk)
        if received != metadata["size"] or digest(archive) != metadata["sha256"]:
            raise ValueError("Bridge archive integrity check failed")
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or set(names) != set(metadata["files"]):
                raise ValueError("Bridge archive has unexpected files")
            for name, expected in metadata["files"].items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
                    raise ValueError("Unsafe bridge archive path")
                data = bundle.read(name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Bridge file integrity check failed")
                destination = ROOT / name
                if destination.parent.resolve() != destination.parent:
                    raise ValueError("Native installation must not follow directory links")
                if destination.exists() and digest(destination) != expected:
                    raise ValueError("An unrecognized local bridge was preserved; remove it manually before reinstalling")
                destination.parent.mkdir(parents=True, exist_ok=True)
                descriptor, stage = tempfile.mkstemp(dir=destination.parent, prefix="native-", suffix=".tmp")
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(data)
                    os.replace(stage, destination)
                finally:
                    Path(stage).unlink(missing_ok=True)


def main():
    if sys.platform != "win32" or sys.maxsize <= 2 ** 32 or sys.version_info < (3, 12):
        raise RuntimeError("DLSS5 NR requires Forge-neo on 64-bit Windows with Python 3.12 or newer")
    environment = ROOT / ".venv"
    if environment.exists() and environment.resolve() != environment:
        raise ValueError("The extension environment must be a physical directory, not a junction")
    python = environment / "Scripts/python.exe"
    if not python.is_file():
        venv.EnvBuilder(with_pip=True, symlinks=False).create(environment)
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
                    "-r", str(ROOT / "requirements-runtime.txt")], check=True, timeout=300,
                   env=dict(os.environ), stdin=subprocess.DEVNULL)
    install_native()
    print("DLSS5 NR dependencies are installed. Open DLSS5 NR in txt2img to finish automatic runtime preparation.")


if __name__ == "__main__":
    main()
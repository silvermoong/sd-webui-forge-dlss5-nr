"""Bounded HTTPS download of a publisher-configured runtime, without a hash allowlist."""
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from urllib.parse import urlsplit
import urllib.request
import zipfile

MAX_DOWNLOAD = 512 * 1024 ** 2
MAX_RUNTIME = 300 * 1024 ** 2


class DownloadError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def source_url(value):
    if not isinstance(value, str) or not value.strip():
        raise DownloadError("source_missing", "The runtime download source has not been configured in this development build")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise DownloadError("source_invalid", "Runtime downloads require a public HTTPS URL without credentials")
    return value


class SecureRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, url):
        source_url(url)
        return super().redirect_request(request, file, code, message, headers, url)


def read_source(path):
    if not path.is_file() or path.stat().st_size > 16384:
        raise DownloadError("source_missing", "The runtime download source has not been configured in this development build")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"url", "member"}:
        raise DownloadError("source_invalid", "Invalid runtime download manifest")
    source_url(value["url"])
    member = value["member"]
    if member is not None:
        if not isinstance(member, str):
            raise DownloadError("source_invalid", "Invalid runtime archive member")
        relative = PurePosixPath(member)
        if (relative.is_absolute() or ".." in relative.parts
                or "\\" in member or ":" in member or relative.name.lower() != "nvngx_dlssnr.dll"):
            raise DownloadError("source_invalid", "Invalid runtime archive member")
    return value


def install_runtime(source, target, verifier, progress=None, *, opener=None):
    url = source_url(source["url"])
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.resolve() != target.parent:
        raise DownloadError("path_invalid", "Runtime directory must not change through a link")
    if target.exists():
        raise DownloadError("exists", "An existing runtime was preserved")
    opener = opener or urllib.request.build_opener(SecureRedirect()).open
    deadline = time.monotonic() + 300
    with tempfile.TemporaryDirectory(prefix="download-", dir=target.parent) as directory:
        archive = Path(directory) / "payload"
        request = urllib.request.Request(url, headers={"User-Agent": "forge-dlss5-nr", "Accept-Encoding": "identity"})
        with opener(request, timeout=30) as response, archive.open("wb") as output:
            source_url(response.geturl())
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > MAX_DOWNLOAD):
                raise DownloadError("size_invalid", "Invalid runtime download size")
            expected = int(length) if length is not None else None
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > MAX_DOWNLOAD or time.monotonic() > deadline:
                    raise DownloadError("limit", "Runtime download exceeded its size or time limit")
                output.write(chunk)
                if progress is not None:
                    progress(received, expected)
            if not received or (expected is not None and received != expected):
                raise DownloadError("incomplete", "Runtime download was incomplete; retry preparation")
        staged = Path(directory) / "nvngx_dlssnr.dll"
        member = source.get("member")
        if member is None:
            if archive.stat().st_size > MAX_RUNTIME:
                raise DownloadError("size_invalid", "Runtime file is too large")
            archive.rename(staged)
        else:
            with zipfile.ZipFile(archive) as bundle:
                matches = [entry for entry in bundle.infolist() if entry.filename == member]
                if len(matches) != 1 or matches[0].is_dir() or not 1024 <= matches[0].file_size <= MAX_RUNTIME:
                    raise DownloadError("archive_invalid", "The archive does not contain one valid runtime file")
                with bundle.open(matches[0]) as input_file, staged.open("wb") as output:
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)
        if staged.stat().st_size < 1024:
            raise DownloadError("size_invalid", "Downloaded runtime file is too small")
        verifier(staged)
        if target.exists():
            raise DownloadError("exists", "Another process installed a runtime; the existing file was preserved")
        os.rename(staged, target)
    return target
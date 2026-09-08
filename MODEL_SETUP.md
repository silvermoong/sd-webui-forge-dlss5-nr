# DLSS5 NR Runtime Preparation

## Normal Use

With **Automatic original** selected, opening the UI prepares a missing runtime from the configured source and shows download progress.
End users do not need to search for the original-runtime download, enter a file path or check a permission box.
After a download failure, use **Retry runtime and detect GPUs**, then choose the NVIDIA adapter by model name.
For missing dependencies, open **Advanced: diagnostics and repair** and choose **Repair dependencies and continue**.

Files are stored in **DLSS-NR inside Forge's model directory**:

```text
models/
    DLSS-NR/
        nvngx_dlssnr.dll
        caller/
            nvngx.dll_comfy.dll
```

Forge's custom model directory is respected. The first file is the NVIDIA runtime; the second is the MIT caller helper.
The private Python environment and native bridge stay in the extension directory. Forge's Torch installation is not replaced.

## Download Source

The configured source was recovered from the existing acquisition record:
[Magpie v0.6.6-experimental](https://github.com/SAOG0721/Magpie/releases/tag/v0.6.6-experimental),
asset [DLSSNR-DLL-Options-310.8.0.0.zip](https://github.com/SAOG0721/Magpie/releases/download/v0.6.6-experimental/DLSSNR-DLL-Options-310.8.0.0.zip).

The archive contains original and community alternatives. Automatic mode downloads the archive but extracts **only**
`NVIDIA-Original/nvngx_dlssnr.dll`, without extracting or executing the community member.
The original was downloaded again, passed NVIDIA Authenticode verification and matched the previously used original byte for byte.
This is a **third-party distribution source**, not an NVIDIA-hosted download or an endorsement. This project does not rehost that DLL.
The extension installer obtains this project's MIT helpers from its GitHub release separately.

## Manual / Community Mode

Select **Manual custom / community**, then use **Advanced: diagnostics and repair -> Import runtime copy** to choose your DLL.
You can also place `nvngx_dlssnr.dll` directly in the displayed directory, normally `models/DLSS-NR/manual`, and retry preparation.
The helper is prepared alongside it; you do not need to replace the automatic-original file.

Manual mode is saved across page reloads and Forge restarts. It **never downloads a missing runtime**, never replaces it with the original,
and does not claim a valid NVIDIA signature. It checks local Windows x64 DLL headers and copying consistency only.
An executable file can still be unsafe or incompatible despite passing those checks. RTX 40/community support has not been run or certified here.

Changing modes or importing requires the private controller to be idle. The source file is not modified.
An existing different target is kept; stop using the runtime before manually replacing that local file.
Switching back to Automatic original restores its separate directory without deleting the manual copy.

## Validation And Retry

- Runtime retries do not run the dependency installer. Only the explicit repair action reinstalls dependencies.
- Dependency repair refuses active or unknown worker states and closes only the extension's idle private controller.
- Short, localized recovery messages appear at the top. Full errors are kept in the Advanced diagnostic box.
- Automatic downloads are staged and published only after completion and NVIDIA signature verification. Interrupted downloads leave no completed-looking runtime.
- There is no fixed NVIDIA runtime hash or version allowlist. SHA-256 identifies local bytes and checks copy integrity.
- A valid existing runtime is reused, not repeatedly downloaded or automatically replaced.
- File readiness is not GPU/driver/API compatibility. The first real generation reports compatibility failures explicitly; the original image is not reported as an enhanced success.
- Do not disable security software, patch drivers or delete the model directory to work around errors.

## Publisher Configuration

The maintainer configures `runtime-download.json`; it is not an end-user setting.
`url` is a stable public HTTPS file URL. `member` is null for a direct DLL, or the exact `nvngx_dlssnr.dll` member name inside a ZIP.
The current URL points directly to the third-party asset above, with `member` set to `NVIDIA-Original/nvngx_dlssnr.dll`.
The source record and a valid publisher signature do not establish redistribution rights. No NVIDIA runtime has been uploaded to this project's repository or Hugging Face.

Pinned MIT bridge checksums cover this project's helpers only, not allowed NVIDIA versions. The MIT license does not grant rights in NVIDIA software.

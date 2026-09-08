# DLSS5 NR for Forge

[简体中文](README.zh-CN.md) | [Runtime / model setup](MODEL_SETUP.md)

An **experimental, unofficial** NVIDIA Neural Rendering extension for **Forge-neo on Windows**.
Enhance generated still images before or after Hires. fix, with an always-visible header switch,
named presets and X/Y/Z comparisons. The panel follows Forge's interface language automatically;
English and Simplified Chinese are supported, with no separate language selector.

![English interface in a CPU-only UI fixture](docs/ui-en.png)

**Automatic runtime source:** the third-party
[Magpie v0.6.6-experimental release](https://github.com/SAOG0721/Magpie/releases/tag/v0.6.6-experimental).
The downloader fetches its archive and extracts only `NVIDIA-Original/nvngx_dlssnr.dll`, then verifies the NVIDIA signature.
Manual custom/community runtimes are supported separately. This project's releases contain only the MIT bridge and its notices, not the NVIDIA runtime.
See [runtime preparation](MODEL_SETUP.md) for the source and manual installation options.
This is not NVIDIA Super Resolution, Ray Reconstruction or a general DLSS SDK installer.

## Features

- Header checkbox: enable/disable NR while the panel is collapsed. Off leaves native generation untouched.
- Before hires: first pass -> NR -> upscaler -> hires sampling. After hires: both passes -> NR.
  Without Hires. fix, only the first position is available. Both positions run before ADetailer.
- Style/preset indices, intensity, tone, structure, skin, automatic mask and a separate output blend.
- Named parameter presets; `[DLSS5 NR] Enabled` and `[DLSS5 NR] Preset` axes in **X/Y/Z plot**.
  Presets do not implicitly enable NR or select a GPU. Grid presets are frozen before the grid begins.
- Explicit physical GPU selection. NR never unloads other applications' models, guesses another GPU or silently reduces quality.
- A private process isolates native runtime failures and exits with Forge, including abrupt Forge exits on Windows.
- Automatic-original and manual-custom runtime modes use separate directories. Your manual file is never automatically downloaded or replaced.

## Requirements

- 64-bit Windows; Forge-neo with Python **3.12 or 3.13**. Other WebUI forks are not claimed compatible.
- Compatible NVIDIA GPU/driver and NR runtime. Tested: **RTX 5060 Ti, driver 610.62,
  unmodified NVIDIA 310.8.0.0 runtime**. RTX 40/community runtime compatibility has not been tested here; use the explicit manual mode for your own compatible file.
- A network connection for the small private Python dependencies, this project's MIT bridge and the automatic runtime download.
- Visual Studio is **not** needed for the prebuilt bridge. It is needed only to rebuild the native source yourself.

The tested still-image GPU allocation was about 0.5 GiB at 832x1216 and 0.9 GiB at 2048x2048.
These are observations, not minimum-memory guarantees. Full cards can still fail or become slow.

## Install

Repository URL (no GitHub access token required):

```text
https://github.com/silvermoong/sd-webui-forge-dlss5-nr.git
```

1. Check that normal generation works in **Forge-neo** and that its startup log shows Python **3.12 or 3.13**.
2. Open **Extensions -> Install from URL**, paste the repository URL above, leave other fields at their defaults,
  and click **Install**. Dependency preparation starts at this point.
3. Open **Installed -> Apply and restart UI** after active jobs finish. A browser refresh is not this operation.
4. Return to **txt2img** and expand **DLSS5 NR**. **Automatic original** prepares a missing runtime and shows download progress.
  For an interrupted download, use **Retry runtime and detect GPUs**. Missing dependencies require
  **Advanced: diagnostics and repair -> Repair dependencies and continue**; ordinary retries do not install Python packages.
5. Click **List GPUs** and select your NVIDIA adapter by model name. The runtime retry button already lists devices.
  Continue once **Ready to generate with ...** appears. No model is loaded by listing devices.

The runtime is stored in **`DLSS-NR` inside Forge's model directory**, normally `models/DLSS-NR`.
Forge's custom model directory is respected. The plugin keeps its own Python dependencies and never replaces Forge's Torch installation.
The **Installed** message alone is not proof that dependencies or the NR runtime are ready.

## Manual / Community Runtime

1. In **Runtime source**, choose **Manual custom / community**. This disables automatic runtime downloading, including after restarting Forge.
2. Under **Advanced: diagnostics and repair**, select your `nvngx_dlssnr.dll` and click **Import runtime copy**.
  Alternatively, place it in the displayed directory, normally `models/DLSS-NR/manual`, and click **Retry runtime and detect GPUs**.
3. Select the intended GPU and test one image. Switching back to **Automatic original** preserves your manual file.

Manual files are checked for local Windows x64 DLL format and copy integrity, not NVIDIA signature or safety.
Use only code you trust. RTX 40 compatibility depends on your chosen community runtime and driver; an import is not a compatibility test.
Switching modes and importing files require an idle private controller. Existing different files are preserved, not silently overwritten.

## First Test

Use a model and prompt that already generate correctly. Set **512x512, batch size 1, batch count 1**, temporarily disable Hires. fix
and ADetailer, keep the default NR parameters, enable **DLSS5 NR** in its header and click the usual **Generate** button.

Success means a normal output image whose generation parameters contain a `DLSS5 NR` receipt with `status: done`.
File readiness is not GPU/runtime compatibility validation. On failure, disabling DLSS5 NR restores normal generation;
follow the recovery action shown in the preparation message instead of deleting models or reinstalling Forge.
An unavailable release download is a publisher/source problem, not a reason to reinstall Forge.

Dependency repair closes only this extension's confirmed-idle controller before reinstalling its dependencies.
It refuses to run during an NR request or while the controller's state is unknown. Short messages stay at the top;
original errors are available under **Diagnostic details** in the Advanced section and clear after a successful preparation.

## Language, Presets And X/Y/Z

The panel reads Forge's `localization` setting when Forge builds its UI. Change the language in Forge's settings
and reload its UI to apply it everywhere. `zh_CN` and `zh-Hans` use Simplified Chinese; English and unrecognized
locales use English. The plugin has no independent language setting. Model parameters and presets are unchanged.
Save presets in the panel. For a 2x2 comparison, choose `Enabled` (`Off, On`) on X and two named `Preset` values on Y.
With only a preset axis, enable NR in the main panel first. Hires-off normalizes a saved after-hires preset to before-hires.

The model's numeric style/internal-preset controls are experimental indices, not calibrated quality levels.
The optical-flow flag is retained for parameter compatibility but is not used by still-image processing.
NR can change material, lighting and identity details: inspect results; it is not a fidelity guarantee.

## API And Compatibility

The always-on script name is `DLSS5 NR`. Its 12 arguments are:
`enabled, stage, style, preset, intensity, tone, structure, skin, auto_mask, mix, flow, runtime`.
`stage` is `before_hr` or `after_hr`; `runtime` is a complete explicit snapshot, not a preset name.
`GET /forge-nr/capabilities` reports protocol and hook readiness without loading a model.

Tested Forge-neo 2.24 commit `231c0a11038c400a315f1532fb80dd09d67e10f6`: Anima/Qwen VAE, no-hires,
pixel-hires before/after, same-model latent-hires, batching and ADetailer. Cross-checkpoint/VAE/refiner
latent-before-hires is rejected. Video and img2img entry points are not exposed by this extension.

On an uncertain submission/termination, the extension preserves that request's temporary directory and reports it.
Do not reuse it while the provider may still be writing. A failed or missing NR receipt is not treated as success.

## Development And Licensing

`python -m unittest discover -s tests` runs CPU checks. Optional native Forge CPU checks:
set `FORGE_ROOT` and run `python tests/test_plugin.py --native-torch --native-gradio` using Forge's Python.
Tests do not make a claim that every GPU/driver or runtime combination has been validated.

Native bridge rebuild: `cmake -S native -B build -A x64`, then `cmake --build build --config Release`.
The source is pinned to the MIT-licensed [ComfyUI-DLSS5-NR](https://github.com/lisitskyaa/ComfyUI-DLSS5-NR)
commit `a3de4781eef81afe80d0226d1ede0b46b3346a63`, with a local physical-adapter identity wrapper.
See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [source manifest](SOURCE_MANIFEST.json).

NVIDIA software, trademarks, drivers and models are not licensed by this repository's MIT license.
This project is not affiliated with, endorsed by or supported by NVIDIA or the Forge/ComfyUI maintainers.
Do not post proprietary DLLs, tokens, private prompts or personal files in issues.
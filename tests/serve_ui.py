"""Real Forge Gradio components, synthetic NR only, listening on 7875."""
import argparse
import ast
import asyncio
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = str(ROOT / "work/ui-uploads")


def guard(event, args):
    if event in ("socket.connect", "socket.bind"):
        address = args[1]
        if isinstance(address, tuple) and address[:2] != ("127.0.0.1", 7875):
            raise AssertionError("UI fixture cannot access other services")
    if event == "subprocess.Popen":
        raise AssertionError("UI fixture cannot start a worker")
    if event == "ctypes.dlopen" and any(name in str(args[0]).lower() for name in ("nvcuda", "nvml", "dlss", "d3d12")):
        raise AssertionError("UI fixture cannot load GPU libraries")


def definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def main(check, forge_root=None):
    forge = Path(forge_root or os.environ["FORGE_ROOT"]).resolve()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sys.addaudithook(guard)
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI
    from forge_nr.adapter import ForgeAdapter
    from forge_nr.direct import DirectProcessor
    from forge_nr.ui import build_direct_tab, build_ui
    from forge_nr.controls import Presets
    from nr_shared.contract import ARG_KEYS, STAGES, active_passes, default_passes, from_script_args
    from nr_runtime.download import DownloadError

    namespace = dict(gr=gr, wraps=wraps, inspect=inspect, warnings=warnings,
                     GradioDeprecationWarning=DeprecationWarning, __name__=__name__)
    definitions(forge / "modules/gradio_extensions.py", {"EventWrapper", "repair"}, namespace)
    namespace["repair"](gr.Checkbox)
    with patch("gradio.component_meta.create_or_modify_pyi", return_value=None):
        definitions(forge / "modules/ui_components.py", {"InputAccordionImpl", "InputAccordion"}, namespace)
    runtime = dict(bridge="fixture-bridge", runtime_dir="fixture-runtime", gpu_index=0,
                   device="cuda:1", gpu_name="Synthetic GPU (no NR)", channel_order="RGBA", runtime_id="a" * 64)

    class Service:
        def __init__(self, initial=False, setup=None):
            self.initial = initial
            self.setup = setup
            self.direct_mode = "normal"

        def start(self):
            return None

        def repair(self, action):
            return action()

        def runtime(self, device=None):
            selected = {**runtime, "runtime_dir": str(self.setup.runtime_dir),
                        "runtime_id": ("b" if self.setup.mode == "manual" else "a") * 64}
            if self.initial and not device:
                return dict(ready=False, runtime={**selected, "gpu_name": ""},
                            missing=["尚未确认 DXGI/CUDA 物理卡映射；先列出设备并选择"])
            self.initial = False
            return dict(ready=True, runtime=selected, runtime_id=selected["runtime_id"], max_passes=3)

        def status(self):
            return dict(mode="private", running=False, busy=False, pid=None, instance="cpu-fixture")

        def inspect(self, *args):
            return dict(devices=[{**runtime, "luid": "synthetic-luid"}])

        def stop(self):
            return dict(released=True)

        def run(self, command, progress, cancel_event):
            import numpy as np
            from PIL import Image
            if self.direct_mode == "failure":
                raise RuntimeError("Synthetic NR failure")
            if self.direct_mode == "wait":
                progress(dict(ratio=0, message="Waiting for scoped cancellation"))
                if not cancel_event.wait(15):
                    raise RuntimeError("Synthetic cancellation was not requested")
            if cancel_event.is_set():
                raise RuntimeError("Synthetic NR canceled")
            with Image.open(command["source_path"]) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            for params in command.get("passes", [command["params"]]):
                enhanced = np.where(pixels >= 128, 224, 16)
                pixels = pixels * (1 - params["mix"]) + enhanced * params["mix"]
            Image.fromarray(np.rint(pixels).astype(np.uint8)).save(Path(command["output_dir"]) / "output.png")
            progress(dict(ratio=1, message="Synthetic NR complete"))
            selected = command["runtime"]
            native = dict(hardware_verified=True, device=selected["device"], gpu_name=selected["gpu_name"])
            result = dict(name="output.png", kind="image", width=command["source"]["width"], height=command["source"]["height"],
                          params=command["params"], runtime_id=selected["runtime_id"], native=native,
                          instance="cpu-fixture", worker_pid=73, preview=False, approximate=False)
            if "passes" in command:
                result.update(passes=command["passes"], completed_passes=len(command["passes"]),
                              pass_results=[dict(params=params, native=native) for params in command["passes"]])
            return result

    head = """<script>
function gradioApp() { return document; }
function updateInput(element) { element.dispatchEvent(new Event('input', {bubbles: true})); }
function onUiLoaded(callback) {
  const check = () => {
    if (!document.querySelector('#forge_nr-checkbox input') || !document.querySelector('#forge_nr .label-wrap')) return;
        if ((location.pathname.startsWith('/i2i-') || location.pathname.startsWith('/direct-')) && (!document.querySelector('#forge_nr_img2img-checkbox input') || !document.querySelector('#forge_nr_img2img .label-wrap'))) return;
    observer.disconnect(); callback();
  };
  const observer = new MutationObserver(check);
  observer.observe(document.documentElement, {childList: true, subtree: true}); check();
}
""" + (forge / "javascript/inputAccordion.js").read_text(encoding="utf-8") + "\n" + (ROOT / "javascript/forge_nr_presets.js").read_text(encoding="utf-8") + "</script>"
    script_path = ROOT / "scripts/forge_nr_script.py"
    script = next(node for node in ast.parse(script_path.read_text(encoding="utf-8")).body
                  if isinstance(node, ast.ClassDef) and node.name == "Script")
    method = next(node for node in script.body if isinstance(node, ast.FunctionDef) and node.name == "ui")
    (ROOT / "work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ui-fixture-", dir=ROOT / "work") as directory:
        app = FastAPI()

        def page(locale, scenario="ready", dual=False, direct=False):
            calls = []
            repairs = 0
            automatic_preparations = 0
            manual_imported = False
            dependencies_ready = scenario != "repair"
            def ensure(repair=False, progress=None):
                nonlocal repairs, dependencies_ready, automatic_preparations
                calls.append(repair)
                if not dependencies_ready:
                    if not repair:
                        raise DownloadError("dependencies_missing", f"Synthetic missing dependencies (attempt {len(calls)})")
                    repairs += 1
                    if repairs == 1:
                        try:
                            raise subprocess.CalledProcessError(1, ["python", "-m", "pip", "install", "CPU-fixture"])
                        except subprocess.CalledProcessError as error:
                            raise DownloadError("dependencies_failed", "Synthetic dependency repair failed") from error
                    dependencies_ready = True
                if setup.mode == "manual":
                    if not manual_imported:
                        raise DownloadError("manual_missing", "Synthetic manual file is missing")
                    return
                automatic_preparations += 1
                if scenario == "network":
                    if repair:
                        raise AssertionError("A runtime retry must not reinstall dependencies")
                    if len(calls) == 1:
                        raise URLError("Synthetic download interruption")
                if progress is not None:
                    progress(10, 10)
            def select_mode(mode):
                setup.mode = mode
                setup.runtime_dir = Path("C:/Forge/models/DLSS-NR/manual" if mode == "manual" else "C:/Forge/models/DLSS-NR")
            def import_file(filename):
                nonlocal manual_imported
                if setup.mode != "manual" or Path(filename).name != "nvngx_dlssnr.dll":
                    raise ValueError("This fixture only accepts the synthetic manual upload")
                if b"CPU_FIXTURE_ONLY" not in Path(filename).read_bytes():
                    raise ValueError("The fixture refuses real runtime files")
                manual_imported = True
            setup = SimpleNamespace(root=Path(directory) / (locale + scenario + ("-direct" if direct else "-dual" if dual else "")), ensure=ensure,
                                    import_runtime=import_file, select_device=lambda value: None,
                                    mode="auto", runtime_dir=Path("C:/Forge/models/DLSS-NR"), set_mode=select_mode)
            presets = Presets(setup.root / "data")
            presets.save("legacy-fixture", "after_hr", {"mix": .25})
            if dual:
                pages = default_passes()
                for index, record in enumerate(pages):
                    record.update(enabled=True, stage="before_hr" if index == 1 else "after_hr")
                    record["params"]["mix"] = (.31, .52, .73)[index]
                presets.save_passes("hires-fixture", pages)
            service = Service(scenario != "ready", setup)
            if direct:
                processor = DirectProcessor(ForgeAdapter(None, None, None, None, service, setup.root / "requests"))
                direct_block = build_direct_tab(gr, service, presets, processor, localization=locale, setup=setup)
                app.add_event_handler("shutdown", processor.close)
            with gr.Blocks(analytics_enabled=False, head=head, css=".input-accordion-checkbox {margin-right: 8px !important;}") as block:
                hr = gr.Checkbox(False, label="Hires. fix", elem_id="fixture_hr")
                entry = dict(gr=gr, shared=SimpleNamespace(opts=SimpleNamespace(localization=locale)),
                             build_ui=build_ui, service=service, presets=presets,
                             setup=setup, InputAccordion=namespace["InputAccordion"])
                exec(compile(ast.Module(body=[method], type_ignores=[]), str(script_path), "exec"), entry)

                def panel(is_img2img):
                    controls = entry["ui"](SimpleNamespace(hr=hr), is_img2img)
                    assert len(controls) == len(ARG_KEYS) == 35
                    suffix = "_img2img" if is_img2img else ""
                    capture = gr.Button("Capture request (CPU fixture)", elem_id="fixture_capture" + suffix)
                    result = gr.Textbox(label="Submitted arguments", elem_id="fixture_request" + suffix)
                    sequence = 0

                    def submitted(*values):
                        nonlocal sequence
                        args, hires = (values, False) if is_img2img else (values[:-1], values[-1])
                        spec = from_script_args(args, hires=hires)
                        sequence += 1
                        return json.dumps(dict(args=args, synthetic=True, sequence=sequence, preparation_calls=list(calls),
                                               spec=spec, execution_order=[record for stage in STAGES for record in active_passes(spec, stage)],
                                               runtime_mode=setup.mode, manual_imported=manual_imported,
                                               automatic_preparations=automatic_preparations, saved_presets=presets.snapshot()))

                    capture.click(submitted, inputs=[*controls, *([] if is_img2img else [hr])], outputs=[result], api_name=False)

                if dual:
                    with gr.Tabs(elem_id="fixture_modes"):
                        with gr.Tab("txt2img"):
                            panel(False)
                        with gr.Tab("img2img"):
                            panel(True)
                        if direct:
                            with gr.Tab("DLSS5 NR", elem_id="fixture_direct_tab"):
                                direct_block.render()
                                with gr.Accordion("CPU fixture", open=False):
                                    mode = gr.Radio(["normal", "wait", "failure"], value="normal", label="Synthetic worker mode",
                                                    elem_id="fixture_direct_mode")
                                    mode_state = gr.Textbox("normal", label="Active synthetic mode", interactive=False,
                                                           elem_id="fixture_direct_mode_state")

                                    def change_direct_mode(value):
                                        service.direct_mode = value
                                        return value

                                    mode.change(change_direct_mode, inputs=[mode], outputs=[mode_state], queue=False, api_name=False)
                else:
                    panel(False)
            return block

        for route, locale, scenario, dual, direct in (("/en", "None", "ready", False, False), ("/zh", "zh_CN", "ready", False, False),
                                                      ("/new", "None", "network", False, False), ("/repair", "zh_CN", "repair", False, False),
                                                      ("/i2i-en", "None", "ready", True, False), ("/i2i-zh", "zh_CN", "ready", True, False),
                                                      ("/direct-en", "None", "ready", True, True), ("/direct-zh", "zh_CN", "ready", True, True)):
            block = page(locale, scenario, dual, direct)
            config = block.get_config_file()
            assert not any(component["props"].get("elem_id") == "forge_nr_language" for component in config["components"])
            assert all(len(item["outputs"]) == len(set(item["outputs"])) for item in config["dependencies"])
            element_ids = [item["props"]["elem_id"] for item in config["components"] if item["props"].get("elem_id")]
            assert len(element_ids) == len(set(element_ids))
            app = gr.mount_gradio_app(app, block, path=route)
        if check:
            print("CPU_UI_CONFIG_OK: actual Forge entry; 35 inputs per mode; txt2img/img2img/direct; English/Chinese")
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind(("127.0.0.1", 7875))
                listener.listen(128)
                print("CPU NR FORGE FIXTURE http://127.0.0.1:7875/en/ http://127.0.0.1:7875/zh/", flush=True)
                loop.run_until_complete(uvicorn.Server(uvicorn.Config(app, log_level="warning")).serve(sockets=[listener]))
    loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--forge-root")
    arguments = parser.parse_args()
    main(arguments.check, arguments.forge_root)
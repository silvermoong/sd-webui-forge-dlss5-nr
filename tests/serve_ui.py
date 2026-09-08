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


def main(check):
    forge = Path(os.environ["FORGE_ROOT"]).resolve()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sys.addaudithook(guard)
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI
    from forge_nr.ui import build_ui
    from forge_nr.controls import Presets
    from nr_shared.contract import from_script_args
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
            return dict(ready=True, runtime=selected, runtime_id=selected["runtime_id"])

        def status(self):
            return dict(mode="private", running=False, busy=False, pid=None, instance="cpu-fixture")

        def inspect(self, *args):
            return dict(devices=[{**runtime, "luid": "synthetic-luid"}])

        def stop(self):
            return dict(released=True)

    head = """<script>
function gradioApp() { return document; }
function updateInput(element) { element.dispatchEvent(new Event('input', {bubbles: true})); }
function onUiLoaded(callback) {
  const check = () => {
    if (!document.querySelector('#forge_nr-checkbox input') || !document.querySelector('#forge_nr .label-wrap')) return;
    observer.disconnect(); callback();
  };
  const observer = new MutationObserver(check);
  observer.observe(document.documentElement, {childList: true, subtree: true}); check();
}
""" + (forge / "javascript/inputAccordion.js").read_text(encoding="utf-8") + "</script>"
    script_path = ROOT / "scripts/forge_nr_script.py"
    script = next(node for node in ast.parse(script_path.read_text(encoding="utf-8")).body
                  if isinstance(node, ast.ClassDef) and node.name == "Script")
    method = next(node for node in script.body if isinstance(node, ast.FunctionDef) and node.name == "ui")
    (ROOT / "work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ui-fixture-", dir=ROOT / "work") as directory:
        app = FastAPI()

        def page(locale, scenario="ready"):
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
            setup = SimpleNamespace(root=Path(directory) / (locale + scenario), ensure=ensure,
                                    import_runtime=import_file, select_device=lambda value: None,
                                    mode="auto", runtime_dir=Path("C:/Forge/models/DLSS-NR"), set_mode=select_mode)
            with gr.Blocks(analytics_enabled=False, head=head, css=".input-accordion-checkbox {margin-right: 8px !important;}") as block:
                hr = gr.Checkbox(False, label="Hires. fix", elem_id="fixture_hr")
                entry = dict(gr=gr, shared=SimpleNamespace(opts=SimpleNamespace(localization=locale)),
                             build_ui=build_ui, service=Service(scenario != "ready", setup), presets=Presets(setup.root / "data"),
                             setup=setup, InputAccordion=namespace["InputAccordion"])
                exec(compile(ast.Module(body=[method], type_ignores=[]), str(script_path), "exec"), entry)
                controls = entry["ui"](SimpleNamespace(hr=hr), False)
                capture = gr.Button("Capture request (CPU fixture)", elem_id="fixture_capture")
                result = gr.Textbox(label="Submitted arguments", elem_id="fixture_request")
                sequence = 0

                def submitted(*values):
                    nonlocal sequence
                    args, hires = values[:-1], values[-1]
                    from_script_args(args, hires=hires)
                    sequence += 1
                    return json.dumps(dict(args=args, synthetic=True, sequence=sequence, preparation_calls=list(calls),
                                           runtime_mode=setup.mode, manual_imported=manual_imported,
                                           automatic_preparations=automatic_preparations))

                capture.click(submitted, inputs=[*controls, hr], outputs=[result], api_name=False)
            return block

        for route, locale, scenario in (("/en", "None", "ready"), ("/zh", "zh_CN", "ready"),
                                        ("/new", "None", "network"), ("/repair", "zh_CN", "repair")):
            block = page(locale, scenario)
            config = block.get_config_file()
            assert not any(component["props"].get("elem_id") == "forge_nr_language" for component in config["components"])
            assert all(len(item["outputs"]) == len(set(item["outputs"])) for item in config["dependencies"])
            app = gr.mount_gradio_app(app, block, path=route)
        if check:
            print("CPU_UI_CONFIG_OK: actual Forge entry; English/Chinese; no language control")
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
    main(parser.parse_args().check)
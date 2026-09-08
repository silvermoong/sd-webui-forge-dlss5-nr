"""Forge-neo alwayson entry point with an owned private NR controller."""
from pathlib import Path
import sys

EXTENSION = Path(__file__).resolve().parents[1]
if str(EXTENSION) not in sys.path:
    sys.path.insert(0, str(EXTENSION))

from forge_nr.discovery import bind_shared_root, discover_root

ROOT = discover_root(EXTENSION)
bind_shared_root(ROOT)

import gradio as gr
import torch
from modules import devices, processing, script_callbacks, scripts, shared
from modules.paths_internal import models_path
from modules.ui_components import InputAccordion

from nr_shared.contract import SCRIPT_TITLE
from forge_nr.adapter import ForgeAdapter
from forge_nr.controls import Presets
from forge_nr.hooks import install
from forge_nr.lifecycle import Service, app_started
from forge_nr.ui import build_ui
from forge_nr.xyz import register as register_xyz

from nr_runtime.setup import RuntimeSetup

setup = RuntimeSetup(ROOT, models_dir=models_path)
service = Service(ROOT)
installation = install(processing, ForgeAdapter(processing, shared, devices, torch, service, ROOT / "tmp"))
presets = Presets(EXTENSION / "data")
xyz_registration = None


class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.hr = None
        # InputAccordion is implemented by this actual hidden Checkbox, not its div.
        self.on_after_component(self._hires_component, elem_id="txt2img_hr-checkbox")

    def _hires_component(self, event):
        self.hr = event.component

    def title(self):
        return SCRIPT_TITLE

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        if is_img2img:
            return []
        return build_ui(gr, service, presets, input_accordion=InputAccordion, hr=self.hr,
            block=gr.context.Context.root_block, localization=shared.opts.localization, setup=setup)


def started(demo, app):
    try:
        setup.configure()
        app_started(app, service, ready_hook=installation.ready)
    except Exception as exc:
        # The visible environment panel and enabled sample path will both refuse.
        print("NR app_started:", exc)


def before_ui():
    global xyz_registration
    xyz_registration = register_xyz(scripts.scripts_data, presets)


def cleanup():
    if xyz_registration is not None:
        xyz_registration.close()
    installation.close()
    service.close()


script_callbacks.on_before_ui(before_ui)
script_callbacks.on_app_started(started)
script_callbacks.on_before_reload(cleanup)
script_callbacks.on_script_unloaded(cleanup)
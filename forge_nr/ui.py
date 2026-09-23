"""Independent txt2img/img2img controls returning exactly contract.ARG_KEYS."""
from base64 import b64encode
import json
from pathlib import Path

from nr_shared.contract import ARG_KEYS, DEFAULT_PARAMS, MAX_PASSES, PARAM_KEYS, PASS_KEYS, SCRIPT_TITLE, validate_passes, validate_runtime
from .controls import preset_passes
from .i18n import forge_language, preparation_message, text

STAGES = ("before_hr", "after_hr")


def build_ui(gr, service, presets, *, input_accordion, hr=None, block=None, localization=None, setup=None,
             is_img2img=False, direct=False):
    language = forge_language(localization)
    elem_prefix = "forge_nr_direct" if direct else "forge_nr_img2img" if is_img2img else "forge_nr"
    if is_img2img or direct:
        hr = None
    controls = {}
    pages = []
    copy_buttons = []

    def component(kind, key, translation=None, property="label", **kwargs):
        if translation:
            kwargs[property] = text(translation, language)
        item = getattr(gr, kind)(elem_id=elem_prefix + "_" + key, **kwargs)
        controls[key] = item
        return item

    def note(key):
        if not direct:
            return gr.Markdown(text(key, language))

    def group(key, **kwargs):
        return gr.Accordion(text(key, language), **kwargs)

    def stage_choices(hires):
        return [(text("direct_stage" if direct else "after_img2img" if is_img2img else key, language), key)
                for key in STAGES[:2 if hires and not is_img2img else 1]]

    def stage_update(hires, stage):
        return gr.update(label=text("stage", language), choices=stage_choices(hires),
                         value=stage if hires else "before_hr")

    def preset_catalog():
        return {name: preset_passes(record) for name, record in presets.snapshot().items()}

    def update_presets(**kwargs):
        records = preset_catalog()
        return gr.update(choices=sorted(records), **kwargs), json.dumps(records, ensure_ascii=False)

    def confirm_action(key):
        message = json.dumps(text(key, language, name="{name}"), ensure_ascii=False)
        return f"""(title, confirmed, ...values) => {{
            const root = (typeof gradioApp === 'function' && gradioApp()) || document;
            const records = JSON.parse(root.querySelector('#{elem_prefix}_preset_catalog textarea')?.value || '{{}}');
            const name = (title || '').trim();
            return [title, Object.hasOwn(records, name) && window.confirm({message}.replace('{{name}}', name)), ...values];
        }}"""

    container = gr.Column(elem_id=elem_prefix) if direct else input_accordion(False, label=SCRIPT_TITLE, elem_id=elem_prefix)
    with container as enabled:
        controls["enabled"] = gr.State(True) if direct else enabled
        gr.HTML("""<style>
.forge-nr-preset-toolbar { display: grid !important; grid-template-columns: minmax(0, 1fr) repeat(2, 32px); gap: 4px !important; align-items: center; }
.forge-nr-preset-toolbar > .form, .forge-nr-preset-toolbar [id$="_preset_list"] { min-width: 0 !important; }
.forge-nr-preset-toolbar .wrap-inner { padding: 8px 4px !important; }
.forge-nr-preset-toolbar input { min-width: 0 !important; width: 100%; margin: 0 !important; padding-right: 24px !important; box-sizing: border-box; text-overflow: ellipsis; }
.forge-nr-preset-toolbar .icon-wrap { width: 16px !important; height: 16px !important; right: 8px !important; }
.forge-nr-preset-toolbar .options { min-width: 160px; }
.forge-nr-preset-toolbar .options .item { width: auto !important; white-space: normal; overflow-wrap: anywhere; }
.forge-nr-preset-toolbar button.forge-nr-preset-button { width: 32px !important; min-width: 32px !important; height: 32px; padding: 0; border-radius: 4px; font-size: 0; flex: none; }
.forge-nr-preset-button img { width: 18px; height: 18px; margin: 0 !important; flex: none; pointer-events: none; }
.dark .forge-nr-preset-button img { filter: invert(1); }
.forge-nr-preset-status { padding: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; }
.forge-nr-preset-status p { font-size: 12px; margin: 0 !important; overflow-wrap: anywhere; }
.forge-nr-preset-state { font-size: 12px; min-height: 0 !important; }
.forge-nr-preset-state:has(span:empty) { display: none !important; }
</style>""", visible=False)
        with gr.Column(elem_id=elem_prefix + "_presets"):
            try:
                records, preset_error = preset_catalog(), ""
            except Exception as exc:
                records, preset_error = {}, text("read_presets_failed", language, error=exc)
            with gr.Row(elem_id=elem_prefix + "_preset_toolbar", elem_classes=["forge-nr-preset-toolbar"]):
                saved = component("Dropdown", "preset_list", "preset_picker", choices=sorted(records), value=None,
                                  allow_custom_value=True, show_label=False, container=False, min_width=0)
                buttons = []
                for key, translation in (("save_preset", "save"), ("delete_preset", "delete")):
                    icon_path = Path(__file__).resolve().parents[1] / "javascript/icons" / (translation + ".svg")
                    icon = {"path": icon_path.name,
                            "url": "data:image/svg+xml;base64," + b64encode(icon_path.read_bytes()).decode("ascii")}
                    buttons.append(component("Button", key, translation, property="value", size="sm", scale=0, min_width=32,
                                             icon=icon,
                                             elem_classes=["tool", "forge-nr-preset-button", "forge-nr-" + key]))
                save, delete = buttons
            selected_preset = component("Textbox", "preset_selection", value="", visible=False)
            catalog = component("Textbox", "preset_catalog", value=json.dumps(records, ensure_ascii=False), visible=False)
            confirmed = component("Checkbox", "preset_confirmed", value=False, visible=False)
            component("HTML", "preset_state", value=(
                f'<span role="status" data-saved="{text("preset_state_saved", language)}" '
                f'data-modified="{text("preset_state_modified", language)}" '
                f'data-new="{text("preset_state_new", language)}"></span>'),
                elem_classes=["forge-nr-preset-status", "forge-nr-preset-state"])
            preset_message = component("Markdown", "preset_message", value=preset_error,
                                       visible=bool(preset_error), elem_classes=["forge-nr-preset-status"])
        with gr.Tabs(elem_id=elem_prefix + "_passes"):
            for index in range(1, MAX_PASSES + 1):
                prefix = "" if index == 1 else f"pass_{index}_"
                with gr.Tab(("1st", "2nd", "3rd")[index - 1], id=index, elem_id=f"{elem_prefix}_tab_{index}"):
                    with gr.Row():
                        page_enabled = component("Checkbox", f"pass_{index}_enabled",
                                                 label=text("enable_pass", language, index=index), value=index == 1)
                        if index > 1:
                            copy_buttons.append(component("Button", f"copy_pass_{index}", "copy_previous", property="value"))
                    stage = component("Radio", prefix + "stage", label=text("stage", language), value="before_hr",
                                      choices=stage_choices(hr is not None and hr.value), interactive=not (is_img2img or direct),
                                      visible=not direct)
                    with gr.Row():
                        component("Dropdown", prefix + "style", "style", value=1,
                                  choices=[(str(number), number) for number in range(3)], type="value")
                        component("Dropdown", prefix + "preset", "preset", value=3,
                                  choices=[(str(number), number) for number in range(4)], type="value")
                    for pair in (("intensity", "tone"), ("structure", "skin")):
                        with gr.Row():
                            for key in pair:
                                component("Slider", prefix + key, key, minimum=-1 if key == "skin" else 0,
                                          maximum=2, step=0.01, value=DEFAULT_PARAMS[key])
                    component("Slider", prefix + "mix", "mix", minimum=0, maximum=1, step=0.01, value=1.)
                    with gr.Row():
                        component("Checkbox", prefix + "auto_mask", "auto_mask", value=False)
                        component("Checkbox", prefix + "flow", "flow", value=True)
                    pages.append([page_enabled, stage, *(controls[prefix + key] for key in PARAM_KEYS)])
        if setup is not None:
            with group("setup", open=not direct, elem_id=elem_prefix + "_setup"):
                runtime_mode = component("Radio", "runtime_mode", "runtime_mode", value=setup.mode,
                                         choices=[(text("runtime_auto", language), "auto"), (text("runtime_manual", language), "manual")])
                runtime_note = component("Markdown", "runtime_note", value=text("runtime_manual_note" if setup.mode == "manual" else "runtime_auto_note", language))
                runtime_location = component("Textbox", "runtime_location", "runtime_location", value=str(setup.runtime_dir), interactive=False)
                setup_message = component("Markdown", "setup_message", value=text("preparing", language))
                prepare_button = component("Button", "prepare_runtime", "prepare", property="value")

        with group("environment_group", open=True):
            note("ownership")
            note("environment_note")
            with gr.Row():
                device = component("Dropdown", "device", "device", choices=[], value=None)
                check = component("Button", "check_environment", "check", property="value")
                enumerate_button = component("Button", "list_devices", "enumerate", property="value")
            runtime = component("Textbox", "runtime", "runtime_json", value="{}", visible=False)
            environment = component("Markdown", "environment", value=text("environment_waiting", language))
            with gr.Row():
                status_button = component("Button", "refresh_status", "status", property="value")
                release = component("Button", "release", "release", property="value")
            backend_status = component("Textbox", "backend_status", "worker_status", interactive=False, lines=4,
                                       value=text("status_waiting", language))

        if setup is not None:
            with group("manual_setup", open=False):
                note("repair_note")
                repair_button = component("Button", "repair_dependencies", "repair_dependencies", property="value")
                setup_details = component("Textbox", "setup_details", "setup_details", value="", interactive=False, lines=4)
                runtime_file = component("File", "runtime_file", "runtime_file", file_types=[".dll"], type="filepath")
                import_button = component("Button", "import_runtime", "import_runtime", property="value")

    def snapshot(selected):
        try:
            status = service.runtime(device=selected or None)
            if not status.get("ready"):
                value = status.get("runtime") or {}
                missing = list(map(str, status.get("missing", [])))
                if len(missing) == 1 and "物理卡映射" in missing[0]:
                    return gr.update(), json.dumps(value), text("choose_gpu", language)
                return gr.update(), json.dumps(value), text("environment_failed", language,
                    error="; ".join(missing) or text("not_ready", language))
            value = validate_runtime(status.get("runtime"))
            if selected and value["device"] != selected:
                raise RuntimeError(text("device_mismatch", language))
            label = f'{value["device"]} · {value["gpu_name"]} · NVIDIA adapter {value["gpu_index"]}'
            manual = setup is not None and setup.mode == "manual"
            detail = text("direct_ready_gpu" if direct else "ready_gpu", language, gpu=value["gpu_name"]) + "\n" + text("manual_files_only" if manual else "files_only", language)
            return gr.update(choices=[(label, value["device"])], value=value["device"]), json.dumps(value, ensure_ascii=False), detail
        except Exception as exc:
            # Keep the selected label visible, invalidate only the executable snapshot.
            return gr.update(), "{}", text("environment_failed", language, error=exc)

    def enumerate_devices(raw):
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(value, dict):
                raise ValueError(text("not_ready", language))
            result = service.inspect(value)
            choices = []
            for row in result["devices"]:
                if row.get("device") and row.get("luid"):
                    choices.append((f'{row["device"]} · {row["gpu_name"]} · NVIDIA adapter {row["gpu_index"]}', row["device"]))
            if not choices:
                raise RuntimeError(text("no_devices", language))
            selected = value.get("device") if value.get("gpu_name") and value.get("device") in [choice[1] for choice in choices] else None
            return gr.update(choices=choices, value=selected), text("enumerated", language, fingerprint=value.get("runtime_id", ""))
        except Exception as exc:
            return gr.update(), text("enumerate_failed", language, error=exc)

    def backend(action):
        try:
            if action == "release":
                service.stop()
            status = service.status()
            safe = {key: status.get(key) for key in ("running", "busy", "pid", "device", "instance", "mode")}
            mode = text("private", language)
            return (text("released", language) if action == "release" else "") + mode + "\n" + json.dumps(safe, ensure_ascii=False, indent=2)
        except Exception as exc:
            return text("worker_failed", language, error=exc)

    def preset_feedback(key, **values):
        return gr.update(value=text(key, language, **values), visible=True)

    def save_preset(title, confirmation, *values):
        try:
            records = []
            for offset in range(0, len(values), len(PASS_KEYS)):
                record = dict(zip(PASS_KEYS, values[offset:offset + len(PASS_KEYS)]))
                records.append(dict(enabled=record["enabled"], stage=record["stage"],
                                    params={key: record[key] for key in PARAM_KEYS}))
            with presets.lock:
                if (title or "").strip() in presets.names() and not confirmation:
                    return gr.update(), gr.update(), preset_feedback("presets_unchanged")
                presets.save_passes(title, validate_passes(records))
                return *update_presets(value=title.strip()), preset_feedback("direct_saved" if direct else "saved_ok")
        except Exception as exc:
            return gr.update(), gr.update(), preset_feedback("save_failed", error=exc)

    def load_preset(selection, hires=False):
        try:
            title = json.loads(selection)["name"]
            records = preset_passes(presets.load(title))
            message = text("direct_loaded" if direct else "loaded_ok", language)
            if not hires and any(record["stage"] == "after_hr" for record in records):
                message += text("direct_reset" if direct else "img2img_reset" if is_img2img else "hires_reset", language)
            values = []
            for record in records:
                values.extend([record["enabled"], stage_update(hires, record["stage"]),
                               *(record["params"][key] for key in PARAM_KEYS)])
            return [*update_presets(value=title), *values, gr.update(value=message, visible=True)]
        except Exception as exc:
            return [*(gr.update() for _ in range(2 + MAX_PASSES * len(PASS_KEYS))), preset_feedback("load_failed", error=exc)]

    def delete_preset(title, confirmation):
        if not confirmation:
            return gr.update(), gr.update(), preset_feedback("presets_unchanged")
        try:
            presets.delete(title)
            return *update_presets(value=None), preset_feedback("deleted_ok")
        except Exception as exc:
            return gr.update(), gr.update(), preset_feedback("delete_failed", error=exc)

    def refresh_presets(previous):
        try:
            if json.loads(previous) == preset_catalog():
                return gr.update(), gr.update()
            return update_presets()
        except Exception as exc:
            raise gr.Error(text("read_presets_failed", language, error=exc)) from exc

    def select_device(selected):
        _, raw, detail = snapshot(selected)
        if setup is not None and raw != "{}":
            try:
                setup.select_device(validate_runtime(raw))
            except Exception as error:
                detail = text("environment_failed", language, error=error)
                return "{}", detail, detail
        return (raw, detail, detail) if setup is not None else (raw, detail)

    def preparation_failure(error):
        details = f"{type(error).__name__}: {error}"
        if error.__cause__ is not None:
            details += f"\n{type(error.__cause__).__name__}: {error.__cause__}"
        return gr.update(), "{}", text("not_ready", language), preparation_message(error, language), details

    def prepare(selected=None, repair=False, progress=gr.Progress()):
        try:
            def ensure():
                return setup.ensure(repair=repair, progress=lambda received, total: progress(
                    (received, total) if total else 0, desc=text("downloading", language)))
            if repair:
                service.repair(ensure)
            else:
                ensure()
            service.start()
            device_update, raw, detail = snapshot(selected)
            result = service.runtime(device=selected or None)
            message = text("direct_prepared" if direct else "prepared", language) if result.get("ready") else detail
            if raw == "{}":
                message = detail
            return device_update, raw, detail, message, ""
        except Exception as error:
            return preparation_failure(error)

    def prepare_and_list(selected, progress=gr.Progress()):
        device_update, raw, detail, message, diagnostics = prepare(selected, progress=progress)
        if raw != "{}":
            device_update, detail = enumerate_devices(raw)
            message = detail
        return device_update, raw, detail, message, diagnostics

    def repair_dependencies(selected, progress=gr.Progress()):
        return prepare(selected, repair=True, progress=progress)

    def change_runtime_mode(mode, selected, progress=gr.Progress()):
        try:
            service.repair(lambda: setup.set_mode(mode))
            result = prepare(selected, progress=progress)
        except Exception as error:
            result = preparation_failure(error)
        note = text("runtime_manual_note" if setup.mode == "manual" else "runtime_auto_note", language)
        return gr.update(value=setup.mode), str(setup.runtime_dir), note, *result

    def load_runtime_mode(selected, progress=gr.Progress()):
        result = prepare(selected, progress=progress)
        note = text("runtime_manual_note" if setup.mode == "manual" else "runtime_auto_note", language)
        return gr.update(value=setup.mode), str(setup.runtime_dir), note, *result

    def import_runtime_file(filename, selected, progress=gr.Progress()):
        try:
            service.repair(lambda: setup.import_runtime(filename))
            return prepare(selected, progress=progress)
        except Exception as error:
            return preparation_failure(error)

    event = dict(api_name=False)
    if setup is None:
        check.click(snapshot, inputs=[device], outputs=[device, runtime, environment], **event)
    else:
        def check_snapshot(selected):
            result = snapshot(selected)
            return *result, result[-1]
        check.click(check_snapshot, inputs=[device], outputs=[device, runtime, environment, setup_message], **event)
    # input fires only on human selection. change would also fire on our own
    # dropdown updates, wiping the enumerated choices or recursively refreshing.
    device.input(select_device, inputs=[device], outputs=[runtime, environment, *([setup_message] if setup is not None else [])],
                 trigger_mode="always_last", **event)
    enumerate_button.click(enumerate_devices, inputs=[runtime], outputs=[device, environment], **event)
    status_button.click(lambda: backend("status"), outputs=[backend_status], **event)
    release.click(lambda: backend("release"), outputs=[backend_status], **event)
    page_controls = [control for page in pages for control in page]
    save.click(save_preset, inputs=[saved, confirmed, *page_controls], outputs=[saved, catalog, preset_message],
               js=confirm_action("confirm_overwrite"), **event)
    selected_preset.input(load_preset, inputs=[selected_preset, *([hr] if hr is not None else [])],
                          outputs=[saved, catalog, *page_controls, preset_message], trigger_mode="always_last", **event)
    delete.click(delete_preset, inputs=[saved, confirmed], outputs=[saved, catalog, preset_message],
                 js=confirm_action("confirm_delete"), **event)
    saved.focus(refresh_presets, inputs=[catalog], outputs=[saved, catalog], **event)
    for control in [saved, catalog, *page_controls]:
        control.change(fn=None, js="() => { window.forgeNRPresets?.refresh(); }", queue=False, **event)
    for index, button in enumerate(copy_buttons, 1):
        button.click(lambda where, *values: [gr.update(value=where), *values],
                     inputs=pages[index - 1][1:], outputs=pages[index][1:], **event)
    if setup is not None:
        preparation_outputs = [device, runtime, environment, setup_message, setup_details]
        runtime_outputs = [runtime_mode, runtime_location, runtime_note, *preparation_outputs]
        runtime_mode.input(change_runtime_mode, inputs=[runtime_mode, device],
                           outputs=runtime_outputs, **event)
        prepare_button.click(prepare_and_list, inputs=[device], outputs=preparation_outputs, **event)
        repair_button.click(repair_dependencies, inputs=[device], outputs=preparation_outputs, **event)
        import_button.click(import_runtime_file, inputs=[runtime_file, device], outputs=preparation_outputs, **event)
    if hr is not None:
        for page in pages:
            hr.change(stage_update, inputs=[hr, page[1]], outputs=[page[1]], queue=False, **event)
    if block is not None:
        if setup is None:
            block.load(snapshot, inputs=[device], outputs=[device, runtime, environment], **event)
        else:
            block.load(load_runtime_mode, inputs=[device], outputs=runtime_outputs,
                       concurrency_id="forge_nr_runtime_prepare", concurrency_limit=1, **event)
    return [controls[key] for key in ARG_KEYS]


def build_direct_tab(gr, service, presets, processor, *, localization=None, setup=None):
    from contextlib import closing
    from PIL import Image
    from .direct import DirectError, normalize_image

    language = forge_language(localization)

    def failure(error):
        if isinstance(error, DirectError):
            return text(error.code, language)
        return text("direct_unknown" if getattr(error, "execution_unknown", False) else "direct_failed", language, error=error)

    def upload(filename):
        if not filename:
            raise DirectError("direct_no_image", "Upload a still image")
        with Image.open(filename) as image:
            return normalize_image(image)

    with gr.Blocks(analytics_enabled=False, elem_id="forge_nr_direct_workspace") as block:
        gr.HTML("""<style>
#forge_nr_direct_source { height: 360px; }
#forge_nr_direct_input { flex: 1 0 64px; min-height: 64px; }
#forge_nr_direct_original { flex-shrink: 0; }
</style>""", visible=False)
        with gr.Row():
            with gr.Column(scale=2, min_width=280):
                with gr.Row():
                    enhance = gr.Button(text("direct_enhance", language), variant="primary", elem_id="forge_nr_direct_enhance")
                    cancel = gr.Button(text("direct_cancel", language), interactive=False, elem_id="forge_nr_direct_cancel")
                status = gr.Textbox(label=text("direct_status", language), value="", interactive=False, lines=2,
                                    elem_id="forge_nr_direct_status")
                with gr.Row():
                    with gr.Column(min_width=240, elem_id="forge_nr_direct_source"):
                        source_file = gr.File(label=text("direct_source", language), file_types=["image"], type="filepath", height=360,
                                              elem_id="forge_nr_direct_input")
                        original = gr.Image(label=text("direct_original", language), interactive=False, format="png",
                                            height="calc(360px - 64px - var(--layout-gap, 16px))", visible=False, show_label=False,
                                            show_share_button=False, show_download_button=False, elem_id="forge_nr_direct_original")
                    output = gr.Image(label=text("direct_result", language), interactive=False, format="png", height=360,
                                      min_width=240, show_share_button=False, show_download_button=True, elem_id="forge_nr_direct_output")
                with gr.Accordion(text("direct_receipt", language), open=False):
                    receipt = gr.JSON(value=None, label=text("direct_receipt", language), elem_id="forge_nr_direct_receipt")
            with gr.Column(scale=1, min_width=280):
                controls = build_ui(gr, service, presets, input_accordion=None, block=block,
                                    localization=localization, setup=setup, direct=True)

        def preview(filename):
            try:
                image, message = upload(filename) if filename else None, ""
            except Exception as error:
                image, message = None, failure(error)
            return (gr.update(value=image, visible=image is not None), None, None, message,
                    gr.update(height=64 if image is not None else 360))

        def run_direct(filename, request: gr.Request, progress=gr.Progress(), *values):
            try:
                with closing(processor.run(request.session_hash, upload(filename), list(values),
                                           progress=lambda event: progress(event.get("ratio", 0), desc=text("direct_running", language)))) as task:
                    for result in task:
                        if result is None:
                            yield {output: None, receipt: None, status: text("direct_running", language),
                                   source_file: gr.update(interactive=False), enhance: gr.update(interactive=False),
                                   cancel: gr.update(interactive=True)}
                        else:
                            image, record = result
                            yield {output: image, receipt: record,
                                   status: text("direct_done", language, width=image.width, height=image.height, passes=record["pass_count"]),
                                   source_file: gr.update(interactive=True), enhance: gr.update(interactive=True),
                                   cancel: gr.update(interactive=False)}
            except Exception as error:
                if isinstance(error, DirectError) and error.code == "direct_busy":
                    gr.Warning(failure(error))
                    return
                yield {output: None, receipt: None, status: failure(error), source_file: gr.update(interactive=True),
                       enhance: gr.update(interactive=True), cancel: gr.update(interactive=False)}

        def cancel_direct(request: gr.Request):
            if processor.cancel(request.session_hash):
                return text("direct_canceling", language)
            return gr.update()

        source_file.change(preview, inputs=[source_file], outputs=[original, output, receipt, status, source_file], queue=False, api_name=False)
        source_file.clear(lambda: preview(None), outputs=[original, output, receipt, status, source_file], queue=False, api_name=False)
        enhance.click(run_direct, inputs=[source_file, *controls], outputs=[output, receipt, status, source_file, enhance, cancel],
                      concurrency_limit=None, trigger_mode="once", api_name=False)
        cancel.click(cancel_direct, outputs=[status], queue=False, api_name=False)
    return block
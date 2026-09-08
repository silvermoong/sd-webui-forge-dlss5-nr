"""Actual txt2img controls; exactly contract.ARG_KEYS are returned to Forge."""
import json

from nr_shared.contract import ARG_KEYS, DEFAULT_PARAMS, PARAM_KEYS, SCRIPT_TITLE, validate_runtime
from .i18n import forge_language, preparation_message, text

STAGES = ("before_hr", "after_hr")


def build_ui(gr, service, presets, *, input_accordion, hr=None, block=None, localization=None, setup=None):
    language = forge_language(localization)
    controls = {}

    def component(kind, key, translation=None, property="label", **kwargs):
        if translation:
            kwargs[property] = text(translation, language)
        item = getattr(gr, kind)(elem_id="forge_nr_" + key, **kwargs)
        controls[key] = item
        return item

    def note(key):
        return gr.Markdown(text(key, language))

    def group(key, **kwargs):
        return gr.Accordion(text(key, language), **kwargs)

    def stage_update(hires, stage):
        return gr.update(label=text("stage", language), choices=[(text(key, language), key) for key in STAGES[:2 if hires else 1]],
                         value=stage if hires else "before_hr")

    with input_accordion(False, label=SCRIPT_TITLE, elem_id="forge_nr") as enabled:
        controls["enabled"] = enabled
        if setup is not None:
            with group("setup", open=True):
                runtime_mode = component("Radio", "runtime_mode", "runtime_mode", value=setup.mode,
                                         choices=[(text("runtime_auto", language), "auto"), (text("runtime_manual", language), "manual")])
                runtime_note = component("Markdown", "runtime_note", value=text("runtime_manual_note" if setup.mode == "manual" else "runtime_auto_note", language))
                runtime_location = component("Textbox", "runtime_location", "runtime_location", value=str(setup.runtime_dir), interactive=False)
                setup_message = component("Markdown", "setup_message", value=text("preparing", language))
                prepare_button = component("Button", "prepare_runtime", "prepare", property="value")
        stage = component("Radio", "stage", label=text("stage", language), value="before_hr",
                          choices=[(text(key, language), key) for key in STAGES[:2 if hr is not None and hr.value else 1]])
        note("behavior")
        with gr.Row():
            component("Dropdown", "style", "style", value=1,
                      choices=[(str(i), i) for i in range(3)], type="value")
            component("Dropdown", "preset", "preset", value=3,
                      choices=[(str(i), i) for i in range(4)], type="value")
        for pair in (("intensity", "tone"), ("structure", "skin")):
            with gr.Row():
                for key in pair:
                    component("Slider", key, key,
                              minimum=-1 if key == "skin" else 0, maximum=2, step=0.01, value=DEFAULT_PARAMS[key])
        component("Slider", "mix", "mix",
                  minimum=0, maximum=1, step=0.01, value=1.)
        with gr.Row():
            component("Checkbox", "auto_mask", "auto_mask", value=False)
            component("Checkbox", "flow", "flow", value=True)
        note("experimental")

        with group("presets", open=False):
            note("preset_scope")
            try:
                names, preset_error = presets.names(), ""
            except Exception as exc:
                names, preset_error = [], text("read_presets_failed", language, error=exc)
            with gr.Row():
                saved = component("Dropdown", "preset_list", "saved", choices=names, value=None)
                load = component("Button", "load_preset", "load", property="value")
                delete = component("Button", "delete_preset", "delete", property="value")
            with gr.Row():
                name = component("Textbox", "preset_name", "name", value="", max_lines=1)
                save = component("Button", "save_preset", "save", property="value")
            preset_message = component("Textbox", "preset_message", "preset_status", value=preset_error, interactive=False)

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
            detail = text("ready_gpu", language, gpu=value["gpu_name"]) + "\n" + text("manual_files_only" if manual else "files_only", language)
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

    def save_preset(title, where, *values):
        try:
            presets.save(title, where, dict(zip(PARAM_KEYS, values)))
            return gr.update(choices=presets.names(), value=title.strip()), text("saved_ok", language)
        except Exception as exc:
            return gr.update(), text("save_failed", language, error=exc)

    def load_preset(title, hires=False):
        try:
            record = presets.load(title)
            where = record["stage"] if hires else "before_hr"
            message = text("loaded_ok", language)
            if where != record["stage"]:
                message += text("hires_reset", language)
            return [stage_update(hires, where), *(record["params"][key] for key in PARAM_KEYS), message]
        except Exception as exc:
            return [*(gr.update() for _ in range(1 + len(PARAM_KEYS))), text("load_failed", language, error=exc)]

    def delete_preset(title):
        try:
            presets.delete(title)
            return gr.update(choices=presets.names(), value=None), text("deleted_ok", language)
        except Exception as exc:
            return gr.update(), text("delete_failed", language, error=exc)

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
            message = text("prepared", language) if result.get("ready") else detail
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
    save.click(save_preset, inputs=[name, stage, *(controls[key] for key in PARAM_KEYS)], outputs=[saved, preset_message], **event)
    load.click(load_preset, inputs=[saved, *([hr] if hr is not None else [])],
               outputs=[stage, *(controls[key] for key in PARAM_KEYS), preset_message], **event)
    delete.click(delete_preset, inputs=[saved], outputs=[saved, preset_message], **event)
    if setup is not None:
        preparation_outputs = [device, runtime, environment, setup_message, setup_details]
        runtime_outputs = [runtime_mode, runtime_location, runtime_note, *preparation_outputs]
        runtime_mode.input(change_runtime_mode, inputs=[runtime_mode, device],
                           outputs=runtime_outputs, **event)
        prepare_button.click(prepare_and_list, inputs=[device], outputs=preparation_outputs, **event)
        repair_button.click(repair_dependencies, inputs=[device], outputs=preparation_outputs, **event)
        import_button.click(import_runtime_file, inputs=[runtime_file, device], outputs=preparation_outputs, **event)
    if hr is not None:
        hr.change(stage_update, inputs=[hr, stage], outputs=[stage], queue=False, **event)
    if block is not None:
        if setup is None:
            block.load(snapshot, inputs=[device], outputs=[device, runtime, environment], **event)
        else:
            block.load(load_runtime_mode, inputs=[device], outputs=runtime_outputs, **event)
    return [controls[key] for key in ARG_KEYS]
"""Forge XYZ axes over per-cell script arguments, never shared UI settings."""
from copy import deepcopy
from pathlib import Path

from nr_shared.contract import ARG_KEYS, PARAM_KEYS, SCRIPT_TITLE

ENABLED_LABEL = f"[{SCRIPT_TITLE}] Enabled"
PRESET_LABEL = f"[{SCRIPT_TITLE}] Preset"
_SNAPSHOT = "_forge_nr_xyz_presets"


def _enabled(value):
    if isinstance(value, str):
        selected = value.strip().casefold()
        if selected in ("off", "false", "0"):
            return False
        if selected in ("on", "true", "1"):
            return True
    raise ValueError("NR Enabled expects Off or On")


def _update(processing, values):
    runner = getattr(processing, "scripts", None)
    matches = [script for script in getattr(runner, "alwayson_scripts", ())
               if script.title() == SCRIPT_TITLE]
    if len(matches) != 1:
        raise RuntimeError("XYZ requires exactly one enabled NR extension")
    script = matches[0]
    source = processing.script_args
    start, end = script.args_from, script.args_to
    if (not isinstance(source, (list, tuple)) or type(start) is not int or type(end) is not int
            or start < 1 or end > len(source) or end - start != len(ARG_KEYS)):
        raise RuntimeError("NR XYZ script arguments are incompatible; reload Forge")
    arguments = list(source)
    arguments[start:end] = deepcopy(source[start:end])
    for key, value in values.items():
        arguments[start + ARG_KEYS.index(key)] = value
    processing.script_args = tuple(arguments) if isinstance(source, tuple) else arguments
    processing.extra_generation_params = dict(processing.extra_generation_params)


def apply_enabled(processing, value, values):
    _update(processing, {"enabled": _enabled(value)})


def confirm_enabled(processing, values):
    if not values:
        raise ValueError("Select at least one NR Enabled value: Off or On")
    for value in values:
        _enabled(value)


class PresetAxis:
    def __init__(self, presets):
        self.presets = presets

    def confirm(self, processing, values):
        if not values:
            raise ValueError("Save and select at least one NR preset before running XYZ")
        snapshot = self.presets.snapshot()
        for name in values:
            if not isinstance(name, str) or name.strip() not in snapshot:
                raise ValueError(f"Unknown NR preset: {name}")
        setattr(processing, _SNAPSHOT, snapshot)

    def apply(self, processing, name, values):
        snapshot = getattr(processing, _SNAPSHOT, {})
        if not isinstance(name, str) or name.strip() not in snapshot:
            raise ValueError("NR preset was not included in the grid snapshot")
        record = deepcopy(snapshot[name.strip()])
        stage = record["stage"] if processing.enable_hr else "before_hr"
        _update(processing, {"stage": stage, **{key: record["params"][key] for key in PARAM_KEYS}})
        processing.extra_generation_params["NR preset"] = name.strip()


class Registration:
    def __init__(self, module, axes):
        self.module, self.axes = module, axes

    def close(self):
        self.module.axis_options[:] = [axis for axis in self.module.axis_options
                                        if not any(axis is owned for owned in self.axes)]


def register(script_data, presets):
    modules = {id(data.module): data.module for data in script_data
               if Path(data.path).name == "xyz_grid.py"}
    if len(modules) != 1:
        raise RuntimeError("Forge X/Y/Z plot script was not uniquely loaded")
    module = next(iter(modules.values()))
    preset = PresetAxis(presets)
    definitions = [(ENABLED_LABEL, apply_enabled, confirm_enabled, lambda: ["Off", "On"]),
                   (PRESET_LABEL, preset.apply, preset.confirm, presets.names)]
    axes = []
    for label, apply, confirm, choices in definitions:
        existing = [axis for axis in module.axis_options if axis.label == label]
        if existing:
            if len(existing) != 1 or getattr(existing[0], "_forge_nr_xyz", None) != label:
                raise RuntimeError(f"Another extension already registered XYZ axis {label}")
            axis = existing[0]
            axis.apply, axis.confirm, axis.choices = apply, confirm, choices
        else:
            axis = module.AxisOptionTxt2Img(label, str, apply, confirm=confirm, choices=choices)
            axis._forge_nr_xyz = label
            module.axis_options.append(axis)
        axes.append(axis)
    return Registration(module, axes)
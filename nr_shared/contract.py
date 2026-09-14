"""Versioned NR value contract shared by two clients; stdlib only, no UI state."""
from copy import deepcopy
import hashlib
import json
import math
import re

PROTOCOL = 1
SCRIPT_TITLE = "DLSS5 NR"
PARAM_KEYS = ("style", "preset", "intensity", "tone", "structure", "skin", "auto_mask", "mix", "flow")
DEFAULT_PARAMS = dict(style=1, preset=3, intensity=1., tone=1., structure=1., skin=-1.,
                      auto_mask=False, mix=1., flow=True)
STAGES = ("before_hr", "after_hr")
RUNTIME_KEYS = ("bridge", "runtime_dir", "gpu_index", "device", "gpu_name", "channel_order", "runtime_id")
MAX_PASSES = 3
LEGACY_ARG_KEYS = ("enabled", "stage", *PARAM_KEYS, "runtime")
PASS_KEYS = ("enabled", "stage", *PARAM_KEYS)
ARG_KEYS = (*LEGACY_ARG_KEYS, "pass_1_enabled",
            *(f"pass_{index}_{key}" for index in range(2, MAX_PASSES + 1) for key in PASS_KEYS))


class SharedError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def validate_params(value):
    if not isinstance(value, dict) or set(value) - DEFAULT_PARAMS.keys():
        raise ValueError("Invalid/unknown NR params")
    params = {**DEFAULT_PARAMS, **value}
    for name in ("auto_mask", "flow"):
        if type(params[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    for name, low, high in (("style", 0, 2), ("preset", 0, 3), ("intensity", 0, 2),
                            ("tone", 0, 2), ("structure", 0, 2), ("skin", -1, 2), ("mix", 0, 1)):
        number = params[name]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{name} must be finite and within {low}..{high}")
        if name in ("style", "preset"):
            if type(number) is not int:
                raise ValueError(f"{name} must be an integer")
        else:
            params[name] = float(number)
    return params


def validate_runtime(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) != set(RUNTIME_KEYS):
        raise ValueError("NR需要完整运行环境快照，先检查运行库并选择显卡")
    if (type(value["gpu_index"]) is not int or not 0 <= value["gpu_index"] <= 63
            or not isinstance(value["device"], str) or not re.fullmatch(r"cuda:\d+", value["device"])
            or value["channel_order"] not in ("RGBA", "BGRA")
            or not isinstance(value["runtime_id"], str) or not re.fullmatch(r"[a-f0-9]{64}", value["runtime_id"])):
        raise ValueError("NR显卡或运行库身份无效")
    for key in ("bridge", "runtime_dir", "gpu_name"):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 4096 or "\0" in value[key]:
            raise ValueError(f"NR {key} 无效")
    return deepcopy(value)


def make_spec(enabled, stage, params, runtime, *, hires):
    if type(enabled) is not bool:
        raise ValueError("NR enabled必须是开关")
    if not enabled:
        return None
    if stage not in STAGES:
        raise ValueError("未知NR插入时机")
    if stage == "after_hr" and not hires:
        raise ValueError("未开启高清修复，NR只能位于高清修复前")
    return {"stage": stage, "params": validate_params(params), "runtime": validate_runtime(runtime)}


def default_passes():
    return [dict(enabled=index == 0, stage="before_hr", params=deepcopy(DEFAULT_PARAMS))
            for index in range(MAX_PASSES)]


def validate_pass_params(value):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PASSES:
        raise ValueError("NR需要1至3组执行参数")
    return [validate_params(params) for params in value]


def validate_passes(value):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PASSES:
        raise ValueError("NR需要1至3页参数")
    result = []
    for record in value:
        if (not isinstance(record, dict) or set(record) != {"enabled", "stage", "params"}
                or type(record["enabled"]) is not bool or record["stage"] not in STAGES):
            raise ValueError("NR每页需要独立开关、插入时机与参数")
        result.append(dict(enabled=record["enabled"], stage=record["stage"],
                           params=validate_params(record["params"])))
    return result


def make_pass_spec(enabled, passes, runtime, *, hires):
    if type(enabled) is not bool:
        raise ValueError("NR enabled必须是开关")
    if not enabled:
        return None
    passes = validate_passes(passes)
    active = [record for record in passes if record["enabled"]]
    if not active:
        return None
    if not hires and any(record["stage"] == "after_hr" for record in active):
        raise ValueError("未开启高清修复，NR只能位于高清修复前")
    if len(active) == 1 and passes[0]["enabled"]:
        return make_spec(True, passes[0]["stage"], passes[0]["params"], runtime, hires=hires)
    return dict(passes=passes, runtime=validate_runtime(runtime))


def active_passes(spec, stage=None):
    if spec is None:
        return []
    records = spec.get("passes")
    if records is None:
        records = [dict(enabled=True, stage=spec["stage"], params=spec["params"])]
    return [dict(index=index + 1, stage=record["stage"], params=deepcopy(record["params"]))
            for index, record in enumerate(records)
            if record["enabled"] and (stage is None or record["stage"] == stage)]


def script_args(spec, *, expanded=False):
    if spec is None:
        values = [False, "before_hr", *(DEFAULT_PARAMS[key] for key in PARAM_KEYS), {}]
    elif "passes" in spec:
        passes = default_passes()
        passes[:len(spec["passes"])] = deepcopy(spec["passes"])
        first = passes[0]
        values = [True, first["stage"], *(first["params"][key] for key in PARAM_KEYS),
                  deepcopy(spec["runtime"]), first["enabled"]]
        for record in passes[1:]:
            values.extend([record["enabled"], record["stage"],
                           *(record["params"][key] for key in PARAM_KEYS)])
        return values
    else:
        values = [True, spec["stage"], *(spec["params"][key] for key in PARAM_KEYS), deepcopy(spec["runtime"])]
    if expanded:
        values.append(True)
        for record in default_passes()[1:]:
            values.extend([False, record["stage"], *(record["params"][key] for key in PARAM_KEYS)])
    return values


def from_script_args(args, *, hires):
    if not isinstance(args, (tuple, list)) or len(args) not in (len(LEGACY_ARG_KEYS), len(ARG_KEYS)):
        raise ValueError("NR插件参数版本不一致，先刷新界面/更新插件")
    if len(args) == len(ARG_KEYS):
        values = dict(zip(ARG_KEYS, args))
        passes = [dict(enabled=values["pass_1_enabled"], stage=values["stage"],
                       params={key: values[key] for key in PARAM_KEYS})]
        for index in range(2, MAX_PASSES + 1):
            prefix = f"pass_{index}_"
            passes.append(dict(enabled=values[prefix + "enabled"], stage=values[prefix + "stage"],
                               params={key: values[prefix + key] for key in PARAM_KEYS}))
        return make_pass_spec(args[0], passes, args[11], hires=hires)
    return make_spec(args[0], args[1], dict(zip(PARAM_KEYS, args[2:11])), args[11], hires=hires)


def signature(spec):
    raw = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def execution_summary(spec, count):
    passes = [dict(index=record["index"], stage=stage, count=count)
              for stage in STAGES for record in active_passes(spec, stage)]
    stages = {record["stage"] for record in passes}
    return dict(stage=passes[0]["stage"] if len(stages) == 1 else "mixed",
                pass_count=len(passes), passes=passes)


def validate_receipt(info, spec, count):
    record = info.get("extra_generation_params", {}).get(SCRIPT_TITLE)
    if isinstance(record, str):
        try:
            record = json.loads(record)
        except ValueError:
            record = None
    expected = execution_summary(spec, count)
    if (not isinstance(record, dict) or record.get("protocol") != PROTOCOL or record.get("status") != "done"
            or record.get("signature") != signature(spec) or record.get("count") != count
            or record.get("stage") != expected["stage"]
            or ("passes" in spec or "passes" in record)
            and (record.get("passes") != expected["passes"]
                 or record.get("pass_count") != expected["pass_count"])):
        raise ValueError("forge没有交回与本次参数/张数一致的NR成功收据，不能把原图当增强结果")
    return deepcopy(record)
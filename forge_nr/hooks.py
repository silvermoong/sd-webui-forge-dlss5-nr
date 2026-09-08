"""Exception-transparent sampling hooks; never use Forge's swallowing script hooks."""
from functools import wraps
from copy import deepcopy
import json

from nr_shared.contract import PROTOCOL, SCRIPT_TITLE, from_script_args, signature

_CALL = "_forge_nr_call"
_REQUEST = "_forge_nr_request"


def read_spec(p):
    runner = getattr(p, "scripts", None)
    scripts = getattr(runner, "alwayson_scripts", ())
    found = [s for s in scripts if s.title() == SCRIPT_TITLE]
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("Duplicate NR alwayson scripts")
    script = found[0]
    return from_script_args(p.script_args[script.args_from:script.args_to], hires=bool(p.enable_hr))


def install(processing, adapter):
    """Idempotent even after importlib reload; keep later plugins' outer wrappers."""
    cls = processing.StableDiffusionProcessingTxt2Img
    owners = [_owner(getattr(cls, name), name) for name in ("sample", "sample_hr_pass")]
    found = [owner for owner in owners if owner is not None]
    tracked = cls.__dict__.get("_forge_nr_installation")
    if not found and tracked is not None:
        found = [tracked]
    if found:
        owner = found[0]
        if any(other is not owner for other in found):
            raise RuntimeError("Conflicting NR wrapper chains")
        for name in ("sample", "sample_hr_pass"):
            current = getattr(cls, name)
            if _owner(current, name) is owner:
                continue
            if current is not owner.originals[name]:
                raise RuntimeError("NR hook was replaced by an opaque plugin; reload Forge before enabling NR")
        for name in ("sample", "sample_hr_pass"):
            if getattr(cls, name) is owner.originals[name]:
                setattr(cls, name, owner.wrappers[name])
        owner.adapter = adapter
        return owner
    return Installation(processing, adapter)


def _owner(function, name):
    seen = set()
    while callable(function) and id(function) not in seen:
        seen.add(id(function))
        owner = getattr(function, "_forge_nr_owner", None)
        # wraps() copies attributes; identity prevents mistaking a foreign wrapper
        # for ours and clobbering it on unload.
        if owner is not None and owner.wrappers.get(name) is function:
            return owner
        function = getattr(function, "__wrapped__", None)
    return None


def _clear(p):
    p.extra_generation_params.pop(SCRIPT_TITLE, None)
    for name in (_CALL, _REQUEST):
        if hasattr(p, name):
            delattr(p, name)


class Installation:
    def ready(self):
        return self.adapter is not None and all(
            _owner(getattr(self.cls, name), name) is self for name in self.wrappers)

    def __init__(self, processing, adapter):
        self.cls = processing.StableDiffusionProcessingTxt2Img
        self.img_cls = getattr(processing, "StableDiffusionProcessingImg2Img", ())
        self.adapter = adapter
        self.originals = {name: getattr(self.cls, name) for name in ("sample", "sample_hr_pass")}
        self.wrappers = {}
        self.cls._forge_nr_installation = self
        for name, invoke in (("sample", self.sample), ("sample_hr_pass", self.sample_hr_pass)):
            self._wrap(name, invoke)

    def _wrap(self, name, invoke):
        original = self.originals[name]

        @wraps(original)
        def wrapper(p, *args, **kwargs):
            if self.adapter is None or not isinstance(p, self.cls) or isinstance(p, self.img_cls):
                return original(p, *args, **kwargs)
            return invoke(p, *args, **kwargs)

        wrapper._forge_nr_owner = self
        wrapper._forge_nr_original = original
        self.wrappers[name] = wrapper
        setattr(self.cls, name, wrapper)

    def close(self):
        self.adapter = None
        for name, wrapper in self.wrappers.items():
            if getattr(self.cls, name) is wrapper:
                setattr(self.cls, name, self.originals[name])

    def sample(self, p, *args, **kwargs):
        adapter = self.adapter
        if hasattr(p, _CALL):
            raise RuntimeError("Recursive txt2img NR sample invocation")
        iteration = getattr(p, "iteration", 0)
        previous = getattr(p, _REQUEST, None)
        continuing = previous is not None and iteration > 0 and iteration == previous["iteration"] + 1
        p.extra_generation_params.pop(SCRIPT_TITLE, None)
        try:
            if continuing:
                context = previous
                if bool(p.enable_hr) != context["hires"]:
                    raise ValueError("NR高清开关在同一批次内变化")
            else:
                _clear(p)
                spec = read_spec(p)
                if spec is None:
                    return self.originals["sample"](p, *args, **kwargs)
                context = dict(spec=deepcopy(spec), identities=[], hires=bool(p.enable_hr))
            context.update(iteration=iteration, hr_seen=False, model=p.sd_model)
            spec = context["spec"]
            setattr(p, _CALL, context)
            adapter.prepare(p, spec)
            with adapter.cancellation() as cancel:
                context["cancel"] = cancel
                result = self.originals["sample"](p, *args, **kwargs)
                cancel.check()
                if not p.enable_hr or spec["stage"] == "after_hr":
                    pixels, identities = adapter.enhance(adapter.decode(p, result), spec, cancel)
                    context["identities"].extend(identities)
                    result = adapter.decoded_result(pixels)
                elif not context["hr_seen"]:
                    raise RuntimeError("NR未经过高清入口，拒绝把原图当成功结果")
                identities = context["identities"]
                receipt = dict(protocol=PROTOCOL, status="done", signature=signature(spec), stage=spec["stage"],
                               count=len(identities), **identities[-1])
                p.extra_generation_params[SCRIPT_TITLE] = json.dumps(receipt, ensure_ascii=False)
                setattr(p, _REQUEST, {key: context[key] for key in ("spec", "identities", "hires", "iteration")})
                return result
        except BaseException:
            _clear(p)
            raise
        finally:
            if hasattr(p, _CALL):
                delattr(p, _CALL)

    def sample_hr_pass(self, p, samples, decoded_samples, *args, **kwargs):
        context = getattr(p, _CALL, None)
        if context and context["spec"]["stage"] == "before_hr":
            adapter = self.adapter
            context["cancel"].check()
            if context["hr_seen"]:
                raise RuntimeError("同一轮采样重复进入 NR 高清前入口")
            if p.latent_scale_mode is not None:
                if p.sd_model is not context["model"]:
                    raise ValueError("NR latent 前置的 VAE/模型已改变")
                decoded_samples = adapter.decode(p, samples)
            decoded_samples, identities = adapter.enhance(decoded_samples, context["spec"], context["cancel"])
            context["identities"].extend(identities)
            if p.latent_scale_mode is not None:
                samples = adapter.encode(p, decoded_samples, samples)
            context["hr_seen"] = True
        return self.originals["sample_hr_pass"](p, samples, decoded_samples, *args, **kwargs)
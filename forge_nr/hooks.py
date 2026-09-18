"""Exception-transparent sampling hooks; never use Forge's swallowing script hooks."""
from functools import wraps
from copy import deepcopy
from contextvars import ContextVar
import json

from nr_shared.contract import PROTOCOL, SCRIPT_TITLE, active_passes, execution_summary, from_script_args, signature

_CALL = "_forge_nr_call"
_REQUEST = "_forge_nr_request"


def read_spec(p, *, img2img=False):
    runner = getattr(p, "scripts", None)
    scripts = getattr(runner, "alwayson_scripts", ())
    found = [s for s in scripts if s.title() == SCRIPT_TITLE]
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("Duplicate NR alwayson scripts")
    script = found[0]
    if img2img and not getattr(script, "is_img2img", False):
        return None
    return from_script_args(p.script_args[script.args_from:script.args_to],
                            hires=False if img2img else bool(p.enable_hr))


def _targets(processing):
    cls = processing.StableDiffusionProcessingTxt2Img
    targets = {name: (cls, name) for name in ("sample", "sample_hr_pass")}
    if hasattr(processing, "StableDiffusionProcessingImg2Img"):
        targets["process_images_inner"] = (processing, "process_images_inner")
        targets["post_sample"] = (processing.scripts.ScriptRunner, "post_sample")
    return targets


def install(processing, adapter):
    """Idempotent even after importlib reload; keep later plugins' outer wrappers."""
    cls = processing.StableDiffusionProcessingTxt2Img
    targets = _targets(processing)
    owners = [_owner(getattr(target, attribute), name) for name, (target, attribute) in targets.items()]
    found = [owner for owner in owners if owner is not None]
    tracked = cls.__dict__.get("_forge_nr_installation")
    if not found and tracked is not None:
        found = [tracked]
    if found:
        owner = found[0]
        if any(other is not owner for other in found) or owner.targets != targets:
            raise RuntimeError("Conflicting NR wrapper chains")
        for name, (target, attribute) in targets.items():
            current = getattr(target, attribute)
            if _owner(current, name) is owner:
                continue
            if current is not owner.originals[name]:
                raise RuntimeError("NR hook was replaced by an opaque plugin; reload Forge before enabling NR")
        for name, (target, attribute) in targets.items():
            if getattr(target, attribute) is owner.originals[name]:
                setattr(target, attribute, owner.wrappers[name])
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
            _owner(getattr(target, attribute), name) is self
            for name, (target, attribute) in self.targets.items())

    def __init__(self, processing, adapter):
        self.cls = processing.StableDiffusionProcessingTxt2Img
        self.img_cls = getattr(processing, "StableDiffusionProcessingImg2Img", ())
        self.adapter = adapter
        self.targets = _targets(processing)
        self.originals = {name: getattr(target, attribute) for name, (target, attribute) in self.targets.items()}
        self.wrappers = {}
        self.requests = ContextVar("forge_nr_requests", default=())
        self.cls._forge_nr_installation = self
        for name in self.targets:
            self._wrap(name, getattr(self, name))

    def _wrap(self, name, invoke):
        original = self.originals[name]

        @wraps(original)
        def wrapper(p, *args, **kwargs):
            if self.adapter is None or (name in ("sample", "sample_hr_pass")
                    and (not isinstance(p, self.cls) or isinstance(p, self.img_cls))):
                return original(p, *args, **kwargs)
            return invoke(p, *args, **kwargs)

        wrapper._forge_nr_owner = self
        wrapper._forge_nr_original = original
        self.wrappers[name] = wrapper
        target, attribute = self.targets[name]
        setattr(target, attribute, wrapper)

    def close(self):
        self.adapter = None
        for name, wrapper in self.wrappers.items():
            target, attribute = self.targets[name]
            if getattr(target, attribute) is wrapper:
                setattr(target, attribute, self.originals[name])

    def process_images_inner(self, request, *args, **kwargs):
        parents = self.requests.get()
        token = self.requests.set((*parents, request))
        active = False
        try:
            if parents or not isinstance(request, self.img_cls):
                return self.originals["process_images_inner"](request, *args, **kwargs)
            _clear(request)
            spec = read_spec(request, img2img=True)
            if spec is None:
                return self.originals["process_images_inner"](request, *args, **kwargs)
            active = True
            context = dict(spec=deepcopy(spec), identities=[], count=0, pass_counts={}, iteration=-1)
            setattr(request, _CALL, context)
            self.adapter.prepare(request, spec)
            with self.adapter.cancellation() as cancel:
                context["cancel"] = cancel
                result = self.originals["process_images_inner"](request, *args, **kwargs)
                cancel.check()
                if not context["count"]:
                    raise RuntimeError("NR img2img did not reach post_sample; no enhancement receipt")
                return result
        except BaseException:
            if active:
                _clear(request)
            raise
        finally:
            if active and hasattr(request, _CALL):
                delattr(request, _CALL)
            self.requests.reset(token)

    def post_sample(self, runner, request, result, *args, **kwargs):
        native_result = self.originals["post_sample"](runner, request, result, *args, **kwargs)
        parents = self.requests.get()
        if len(parents) != 1 or parents[0] is not request or not isinstance(request, self.img_cls):
            return native_result
        context = getattr(request, _CALL, None)
        if context is None:
            return native_result
        iteration = getattr(request, "iteration", 0)
        if iteration != context["iteration"] + 1:
            raise RuntimeError("NR img2img post_sample iteration was repeated or skipped")
        context.update(iteration=iteration, iteration_count=None)
        context["cancel"].check()
        pixels = self._enhance(self.adapter.decode(request, result.samples), context, "before_hr")
        result.samples = self.adapter.decoded_result(pixels)
        self._receipt(request, context)
        return native_result

    def _receipt(self, request, context):
        spec = context["spec"]
        context["count"] += context["iteration_count"] or 0
        expected = {record["index"]: context["count"] for record in active_passes(spec)}
        if not context["count"] or context["pass_counts"] != expected:
            raise RuntimeError("NR没有完成全部启用页，拒绝签发成功收据")
        identities = context["identities"]
        receipt = dict(protocol=PROTOCOL, status="done", signature=signature(spec), count=context["count"],
                       **execution_summary(spec, context["count"]),
                       **{key: identities[-1][key] for key in ("instance", "worker_pid")})
        request.extra_generation_params[SCRIPT_TITLE] = json.dumps(receipt, ensure_ascii=False)

    def _enhance(self, pixels, context, stage):
        enhanced, identities = self.adapter.enhance(pixels, context["spec"], context["cancel"], stage=stage)
        count = len(identities)
        indices = [record["index"] for record in active_passes(context["spec"], stage)]
        if (not count or any(identity.get("pass_indices") != indices for identity in identities)
                or context["iteration_count"] not in (None, count)):
            raise RuntimeError("NR各阶段的图片张数或执行页不一致")
        context["iteration_count"] = count
        for index in indices:
            context["pass_counts"][index] = context["pass_counts"].get(index, 0) + count
        context["identities"].extend(identities)
        return enhanced

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
                context = dict(spec=deepcopy(spec), identities=[], hires=bool(p.enable_hr), count=0, pass_counts={})
            context.update(iteration=iteration, hr_seen=False, model=p.sd_model, iteration_count=None)
            spec = context["spec"]
            setattr(p, _CALL, context)
            adapter.prepare(p, spec)
            with adapter.cancellation() as cancel:
                context["cancel"] = cancel
                result = self.originals["sample"](p, *args, **kwargs)
                cancel.check()
                if p.enable_hr and not context["hr_seen"]:
                    raise RuntimeError("NR未经过高清入口，拒绝把原图当成功结果")
                stage = "after_hr" if p.enable_hr else "before_hr"
                if active_passes(spec, stage):
                    pixels = self._enhance(adapter.decode(p, result), context, stage)
                    result = adapter.decoded_result(pixels)
                self._receipt(p, context)
                setattr(p, _REQUEST, {key: context[key] for key in
                                     ("spec", "identities", "hires", "iteration", "count", "pass_counts")})
                return result
        except BaseException:
            _clear(p)
            raise
        finally:
            if hasattr(p, _CALL):
                delattr(p, _CALL)

    def sample_hr_pass(self, p, samples, decoded_samples, *args, **kwargs):
        context = getattr(p, _CALL, None)
        if context:
            adapter = self.adapter
            context["cancel"].check()
            if context["hr_seen"]:
                raise RuntimeError("同一轮采样重复进入 NR 高清前入口")
            if active_passes(context["spec"], "before_hr"):
                if p.latent_scale_mode is not None:
                    if p.sd_model is not context["model"]:
                        raise ValueError("NR latent 前置的 VAE/模型已改变")
                    decoded_samples = adapter.decode(p, samples)
                decoded_samples = self._enhance(decoded_samples, context, "before_hr")
                if p.latent_scale_mode is not None:
                    samples = adapter.encode(p, decoded_samples, samples)
            context["hr_seen"] = True
        return self.originals["sample_hr_pass"](p, samples, decoded_samples, *args, **kwargs)
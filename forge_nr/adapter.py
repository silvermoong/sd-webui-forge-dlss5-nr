"""Native Forge tensor <-> shared worker PNG boundary; no core or Pydantic dependency."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
import tempfile
import threading
import uuid

import numpy as np
from PIL import Image, ImageCms

from nr_shared.contract import validate_runtime


@contextmanager
def request_directory(parent):
    root = Path(tempfile.mkdtemp(prefix="nr-", dir=parent))
    retain = False
    try:
        yield root
    except BaseException as error:
        if getattr(error, "execution_unknown", False):
            retain = True
            error.args = (str(error) + f"；执行是否结束尚未确认，已保留本次临时目录：{root}",)
        raise
    finally:
        if not retain:
            shutil.rmtree(root)


class Cancellation:
    """Poll only this Forge invocation; the client scopes cancel to its own ticket."""
    def __init__(self, state):
        self.state = state
        self.event = threading.Event()
        self.finished = threading.Event()

    def check(self):
        if any(getattr(self.state, key, False) for key in ("interrupted", "skipped", "stopping_generation")):
            self.event.set()
        if self.event.is_set():
            raise RuntimeError("NR：本次生成已取消")

    def __enter__(self):
        self.check()

        def watch():
            while not self.finished.wait(0.1):
                try:
                    self.check()
                except RuntimeError:
                    return

        self.thread = threading.Thread(target=watch, name="forge-nr-cancel", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.finished.set()
        self.thread.join(timeout=1)


class ForgeAdapter:
    """Injected modules/client are the isolated CPU test seam, not user/API options."""
    def __init__(self, processing, shared, devices, torch, client, temp_root):
        self.processing = processing
        self.shared = shared
        self.devices = devices
        self.torch = torch
        self.client = client
        self.temp_root = temp_root

    def cancellation(self):
        return Cancellation(self.shared.state)

    def prepare(self, p, spec):
        if p.enable_hr and spec["stage"] == "before_hr" and p.latent_scale_mode is not None:
            if (getattr(p, "hr_checkpoint_name", None) not in (None, "", "Use same checkpoint")
                    or getattr(p, "hr_additional_modules", None) not in (None, ["Use same choices"])
                    or getattr(p, "refiner_checkpoint", None) not in (None, "", "None", "none")):
                raise ValueError("NR高清前 latent 放大不支持跨 checkpoint/VAE/refiner；请选择像素放大或高清后")
        status = self.client.runtime(device=spec["runtime"]["device"])
        if not status.get("ready"):
            raise RuntimeError("NR环境不可用：" + "; ".join(map(str, status.get("missing", []))))
        if validate_runtime(status.get("runtime")) != spec["runtime"]:
            raise RuntimeError("NR环境快照已过期，请重新选择显卡并确认 fingerprint")

    def decode(self, p, samples):
        if not getattr(samples, "already_decoded", False):
            samples = self.processing.decode_latent_batch(
                p.sd_model, samples, target_device=self.devices.cpu, check_for_nans=True)
        return self.torch.clamp((self.torch.stack(samples).float() + 1.) / 2., min=0., max=1.)

    def decoded_result(self, pixels):
        # Forge processing.py normalizes every DecodedSamples entry from [-1,1].
        # Feeding [0,1] here would silently halve contrast and brighten the image.
        return self.processing.DecodedSamples([image * 2. - 1. for image in pixels])

    def encode(self, p, pixels, samples):
        pixels = pixels.to(self.shared.device, dtype=self.torch.float32)
        method = self.shared.opts.sd_vae_encode_method
        result = self.processing.images_tensor_to_samples(
            pixels, self.processing.approximation_indexes.get(method), p.sd_model)
        if method != "Full":
            p.extra_generation_params["VAE Encoder"] = method
        p.sd_model.ini_latent = None
        return result.to(device=samples.device, dtype=samples.dtype)

    def enhance(self, pixels, spec, cancel):
        array = pixels.detach().float().cpu().numpy()
        # Anima's Qwen image VAE decodes N,T,C,H,W, even for a still (T=1).
        # Latents use N,C,T,H,W instead: never squeeze that axis or flatten a video.
        if array.ndim == 5 and array.shape[1:3] == (1, 3):
            array = array[:, 0]
        if (array.ndim != 4 or array.shape[1] != 3 or not array.size
                or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1):
            raise ValueError(f"NR仅支持有限的 N×3×H×W [0,1] 图片，不支持视频/未知解码形状：{array.shape}")
        if max(array.shape[2:]) > 8192 or np.prod(array.shape[2:]) > 32 * 1024 ** 2:
            raise ValueError("NR图片超过媒体尺寸上限")
        temp_root = Path(self.temp_root() if callable(self.temp_root) else self.temp_root).resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        enhanced, identities = [], []
        for image in array:
            cancel.check()
            # Each request owns one fresh physical directory. Only that tree is removed.
            with request_directory(temp_root) as directory:
                root = Path(directory)
                source, output = root / "input.png", root / "output"
                output.mkdir()
                rgb = np.rint(np.moveaxis(image, 0, 2) * 255.).astype(np.uint8)
                Image.fromarray(rgb).save(source, icc_profile=icc)
                request_id = uuid.uuid4().hex
                command = dict(op="run", id=request_id, source_path=str(source), output_dir=str(output),
                               source=dict(id=request_id, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                           kind="image", width=rgb.shape[1], height=rgb.shape[0]),
                               params=deepcopy(spec["params"]), runtime=deepcopy(spec["runtime"]),
                               start=0., end=0., preview_kind="", max_edge=0)

                def progress(event):
                    cancel.check()
                    self.shared.state.textinfo = "NR：" + str(event.get("message", "处理中"))

                result = self.client.run(command, progress=progress, cancel_event=cancel.event)
                cancel.check()
                native = result.get("native", {})
                if (result.get("kind") != "image" or result.get("width") != rgb.shape[1]
                        or result.get("height") != rgb.shape[0] or result.get("params") != spec["params"]
                        or result.get("runtime_id") != spec["runtime"]["runtime_id"]
                        or native.get("hardware_verified") is not True
                        or native.get("device") != spec["runtime"]["device"]
                        or " ".join(str(native.get("gpu_name", "")).split()).casefold()
                        != " ".join(spec["runtime"]["gpu_name"].split()).casefold()
                        or result.get("preview") is not False or result.get("approximate") is not False):
                    raise RuntimeError("NR输出尺寸/参数/硬件/运行库证据与本次提交不一致")
                if result.get("name") != "output.png":
                    raise RuntimeError("NR返回了非本轮输出路径")
                path = output / "output.png"
                if output.resolve() != output or path.is_symlink() or path.resolve().parent != output:
                    raise RuntimeError("NR输出路径越界")
                with Image.open(path) as rendered:
                    if rendered.format != "PNG" or rendered.size != (rgb.shape[1], rgb.shape[0]) or rendered.mode not in ("RGB", "RGBA"):
                        raise RuntimeError("NR输出不是原尺寸 RGB PNG")
                    enhanced.append(np.moveaxis(np.asarray(rendered.convert("RGB"), dtype=np.float32) / 255., 2, 0))
                # The broker stamps these onto this terminal result while it still
                # owns the execution lease. A later status() can describe somebody
                # else's worker after release/restart; never use it as a receipt.
                instance = result.get("instance")
                worker_pid = result.get("worker_pid")
                if not isinstance(instance, str) or not instance or type(worker_pid) is not int or worker_pid <= 0:
                    raise RuntimeError("NR后台没有返回执行实例/worker PID")
                identities.append(dict(instance=instance, worker_pid=worker_pid))
        return self.torch.from_numpy(np.stack(enhanced)), identities
"""Direct still-image NR jobs, independent of Forge sampling and global state."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import threading

import numpy as np
from PIL import Image, ImageCms, ImageOps

from nr_shared.contract import PROTOCOL, SCRIPT_TITLE, active_passes, execution_summary, from_script_args, signature, validate_receipt
from .adapter import Cancellation


class DirectError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def normalize_image(image):
    if not isinstance(image, Image.Image):
        raise DirectError("direct_no_image", "Upload a still image")
    if getattr(image, "is_animated", False) or image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK"):
        raise DirectError("direct_image_format", "Only ordinary 8-bit still images are supported")
    if not min(image.size) or max(image.size) > 8192 or image.width * image.height > 32 * 1024 ** 2:
        raise DirectError("direct_image_size", "Image exceeds the NR size limit")
    source = ImageOps.exif_transpose(image).copy()
    alpha = source.convert("RGBA").getchannel("A") if "A" in source.getbands() or "transparency" in source.info else None
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    if source.info.get("icc_profile"):
        try:
            rgb = ImageCms.profileToProfile(source if source.mode in ("RGB", "CMYK") else source.convert("RGB"),
                                            ImageCms.ImageCmsProfile(BytesIO(source.info["icc_profile"])),
                                            profile, outputMode="RGB")
        except Exception as error:
            raise DirectError("direct_color_profile", "The embedded image color profile could not be converted to sRGB") from error
    else:
        rgb = source.convert("RGB")
    if alpha is not None:
        rgb.putalpha(alpha)
    rgb.info.clear()
    rgb.info["icc_profile"] = profile.tobytes()
    return rgb


class DirectProcessor:
    def __init__(self, adapter):
        self.adapter = adapter
        self.lock = threading.Lock()
        self.jobs = {}
        self.closed = False

    def cancel(self, session):
        with self.lock:
            cancel = self.jobs.get(session)
            if cancel is None:
                return False
            cancel.event.set()
            return True

    def close(self):
        with self.lock:
            self.closed = True
            for cancel in self.jobs.values():
                cancel.event.set()

    def run(self, session, image, arguments, progress=None):
        if not isinstance(session, str) or not session:
            raise DirectError("direct_session", "Reload the page before starting a direct NR job")
        cancel = Cancellation(None)
        with self.lock:
            if self.closed:
                raise DirectError("direct_closed", "The direct NR page has been unloaded")
            if session in self.jobs:
                raise DirectError("direct_busy", "This page already has an active NR job")
            self.jobs[session] = cancel
        try:
            spec = from_script_args(deepcopy(arguments), hires=False)
            if spec is None:
                raise DirectError("direct_no_pass", "Enable at least one NR parameter tab")
            source = normalize_image(image)
            yield None
            cancel.check()
            alpha = source.getchannel("A") if source.mode == "RGBA" else None
            rgb = source.convert("RGB")
            pixels = np.moveaxis(np.asarray(rgb, dtype=np.float32) / 255., 2, 0)[None]
            self.adapter.prepare_runtime(spec)
            cancel.check()
            enhanced, identities = self.adapter.enhance_array(pixels, spec, cancel, stage="before_hr", progress_callback=progress)
            cancel.check()
            passes = active_passes(spec)
            if enhanced.shape != pixels.shape or len(identities) != 1 or identities[0].get("pass_indices") != [record["index"] for record in passes]:
                raise RuntimeError("Direct NR did not return the expected image and pass evidence")
            identity = identities[0]
            digest = hashlib.sha256(rgb.tobytes())
            if alpha is not None:
                digest.update(alpha.tobytes())
            receipt = dict(protocol=PROTOCOL, status="done", signature=signature(spec), count=1,
                           **execution_summary(spec, 1), instance=identity["instance"], worker_pid=identity["worker_pid"],
                           request_id=identity["request_id"],
                           mode="direct", width=rgb.width, height=rgb.height, input_sha256=digest.hexdigest(),
                           runtime_id=spec["runtime"]["runtime_id"], gpu_name=spec["runtime"]["gpu_name"])
            validate_receipt({"extra_generation_params": {SCRIPT_TITLE: receipt}}, spec, 1)
            result = Image.fromarray(np.rint(np.moveaxis(enhanced[0], 0, 2) * 255.).astype(np.uint8))
            if alpha is not None:
                result.putalpha(alpha)
            result.info["icc_profile"] = source.info["icc_profile"]
            result.info[SCRIPT_TITLE] = json.dumps(receipt, ensure_ascii=False)
            result.info["NR Parameters"] = json.dumps(passes, ensure_ascii=False)
        except Exception as error:
            if cancel.event.is_set() and not getattr(error, "execution_unknown", False):
                raise DirectError("direct_canceled", "The direct NR job was canceled") from error
            raise
        finally:
            with self.lock:
                if self.jobs.get(session) is cancel:
                    del self.jobs[session]
        yield result, receipt
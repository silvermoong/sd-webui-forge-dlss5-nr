"""CPU-only, local-file SDR media boundary for the independent NR worker.

Importing this module does not import PyAV, Torch or a native/GPU library.
Video dependencies are resolved only when a video is actually opened.
"""
import io
import json
from dataclasses import dataclass
from fractions import Fraction
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import MappingProxyType


# Same public ceilings as nr_assets; no reverse dependency on the asset store.
# Callers may tighten these per scan, never silently raise the application caps.
MEDIA_LIMITS = MappingProxyType(dict(max_edge=8192, max_pixels=32 * 1024 ** 2,
                                   max_frames=1_000_000, max_duration=86_400))
MEDIA_DECODE_STALL_SECONDS = 60.


def _limits(value=None):
    if value is not None and (not isinstance(value, dict) or set(value) - MEDIA_LIMITS.keys()):
        raise MediaError("Invalid media limits")
    result = {**MEDIA_LIMITS, **(value or {})}
    for key, number in result.items():
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= MEDIA_LIMITS[key]:
            raise MediaError(f"Invalid media limit: {key}")
    return result


def _dimensions(width, height, limits):
    if width <= 0 or height <= 0 or max(width, height) > limits["max_edge"]:
        raise MediaError("Media dimensions exceed the allowed limit")
    if width * height > limits["max_pixels"]:
        raise MediaError("Media pixels exceed the allowed limit")


def source_facts(source):
    """Validate an immutable parent's facts, not proof of file identity.

    The parent owns hash/registration checks. The decoder still verifies actual
    dimensions, origin, audio and every frame; completion verifies count/span.
    Legacy Python callers without facts take the exact CPU inspection path.
    """
    if source.get("kind") != "video":
        return None
    required = {"width", "height", "frames", "duration", "fps", "source_first_pts", "has_audio"}
    if not required <= source.keys():
        return None
    for key in ("width", "height", "frames"):
        if type(source[key]) is not int or source[key] <= 0:
            raise MediaError(f"Invalid source facts: {key}")
    _dimensions(source["width"], source["height"], MEDIA_LIMITS)
    for key in ("duration", "fps", "source_first_pts"):
        if type(source[key]) not in (int, float) or not math.isfinite(source[key]):
            raise MediaError(f"Invalid source facts: {key}")
    if (not 0 < source["duration"] <= MEDIA_LIMITS["max_duration"]
            or source["frames"] > MEDIA_LIMITS["max_frames"] or not 0 < source["fps"] <= 1000
            or type(source["has_audio"]) is not bool):
        raise MediaError("Invalid source frame/duration/audio facts")
    return {key: source[key] for key in {*required, "kind"}} | {"warnings": list(source.get("warnings", []))}


class MediaError(ValueError):
    pass


class Cancelled(RuntimeError):
    pass


def check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("NR request canceled")


def local_path(path, *, directory=False):
    """The parent supplies paths, not URLs; do not reinterpret traversal/links."""
    if not isinstance(path, (str, Path)) or not str(path) or "\0" in str(path):
        raise MediaError("A local absolute path is required")
    if str(path).startswith(("\\\\", "//")) or "://" in str(path):
        raise MediaError("Network URLs, UNC shares and Windows device paths are not local media")
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise MediaError("A local absolute path without traversal is required")
    for part in (value, *value.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise MediaError("Media paths must not traverse symlinks or junctions")
    if not directory and not value.is_file():
        raise MediaError("Source is not a regular local file")
    return value.resolve()


def _png_cicp(path):
    """Read pre-IDAT color signalling even when Pillow does not expose cICP."""
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        size = path.stat().st_size
        while True:
            header = stream.read(8)
            if len(header) != 8:
                return None
            length, kind = int.from_bytes(header[:4], "big"), header[4:]
            if stream.tell() + length + 4 > size:
                raise MediaError("Truncated PNG chunk")
            if kind in (b"IDAT", b"IEND"):
                return None
            if kind == b"cICP":
                if length != 4:
                    raise MediaError("Invalid PNG cICP color metadata")
                primaries, trc, matrix, full_range = stream.read(4)
                if primaries == 9 or trc in (14, 15, 16, 18) or matrix in (9, 10):
                    raise MediaError("HDR/PQ/HLG/BT.2020 PNG is unsupported; convert explicitly to SDR sRGB first")
                if primaries not in (1, 5, 6) or trc not in (1, 4, 5, 6, 7, 8, 13) or matrix != 0 or full_range != 1:
                    raise MediaError("Unsupported PNG cICP color encoding; explicit sRGB conversion is required")
                return (primaries, trc)
            stream.seek(length + 4, 1)


def _image_info(path, *, limits=None):
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as image:
            _dimensions(image.width, image.height, _limits(limits))
            if getattr(image, "is_animated", False):
                return None
            notes = []
            cicp = _png_cicp(path)
            xmp = str(image.info.get("xmp", b"")).lower()
            if "hdrgm:version" in xmp or "hdrgainmap" in xmp or image.mode == "F":
                raise MediaError("HDR/gain-map/floating-point images require explicit SDR conversion")
            if image.info.get("icc_profile"):
                notes.append("Embedded ICC is converted to sRGB before NR; out-of-gamut colors may be clipped")
            with path.open("rb") as stream:
                header = stream.read(26)
            if image.mode.startswith("I;16") or (header[:8] == b"\x89PNG\r\n\x1a\n" and header[24:25] == b"\x10"):
                notes.append("16-bit image: this version exports 8-bit sRGB PNG; not a lossless 16-bit pipeline")
            oriented = ImageOps.exif_transpose(image)
            return dict(kind="image", width=oriented.width, height=oriented.height,
                        duration=0., frames=1, fps=None, warnings=notes, cicp=cicp)
    except (Image.UnidentifiedImageError, OSError):
        return None


def inspect_media(path, *, cancel_event=None, limits=None, progress=None):
    """Return media facts, without ever initializing a renderer or GPU."""
    check_cancel(cancel_event)
    path = local_path(path)
    limits = _limits(limits)
    if progress:
        progress("Inspecting CPU media", 0.)
    check_cancel(cancel_event)
    image = _image_info(path, limits=limits)
    if image is not None:
        return image
    # Exact streaming CPU scan, never duration*fps. Container duration can belong
    # to a much longer audio/subtitle stream and is intentionally not consulted.
    with MediaReader(path, cancel_event=cancel_event, limits=limits, progress=progress) as reader:
        count, end, previous_duration, vfr = 0, Fraction(0), None, False
        for frame, absolute, duration in reader.timeline():
            count += 1
            end = absolute - reader.first_pts + duration
            if previous_duration is not None and duration != previous_duration:
                vfr = True
            previous_duration = duration
        if not count or end <= 0:
            raise MediaError("Video has no decodable positive-duration frames")
        if progress:
            progress(f"Inspected {count} video frames", 0.)
        check_cancel(cancel_event)
        return dict(kind="video", width=reader.width, height=reader.height, duration=float(end), frames=count,
                    fps=float(count / end), warnings=list(reader.warnings), frame_count_source="decoded",
                    source_first_pts=float(reader.first_pts), vfr=vfr, has_audio=reader.has_audio)


def resize_rgb(rgb, alpha=None, max_edge=0):
    """Downsize only; identical float32 path for before/NR, without cropping."""
    import numpy as np
    from PIL import Image
    height, width = rgb.shape[:2]
    if not max_edge or max(width, height) <= max_edge:
        return np.ascontiguousarray(rgb, dtype=np.float32), alpha
    factor = max_edge / max(width, height)
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    channels = [np.asarray(Image.fromarray(rgb[..., i]).resize(size, Image.Resampling.LANCZOS),
                           dtype=np.float32) for i in range(3)]
    rgb = np.ascontiguousarray(np.clip(np.stack(channels, axis=-1), 0, 1), dtype=np.float32)
    if alpha is not None:
        alpha = np.asarray(Image.fromarray(alpha).resize(size, Image.Resampling.LANCZOS)).copy()
    return rgb, alpha


def read_image(path, *, max_edge=0):
    """Decode oriented sRGB in float32 [0,1], retaining the original alpha."""
    import numpy as np
    from PIL import Image, ImageCms, ImageOps
    path = local_path(path)
    facts = _image_info(path)
    if facts is None:
        raise MediaError("Not a supported still image")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        alpha = np.asarray(image.convert("RGBA"))[..., 3].copy() if (
            "A" in image.getbands() or "transparency" in image.info) else None
        icc = image.info.get("icc_profile")
        if icc:
            try:
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                # Keep CMYK/LAB until the ICC transform; convert palette/RGBA first.
                colors = image if image.mode in ("RGB", "CMYK", "LAB", "L") else image.convert("RGB")
                image = ImageCms.profileToProfile(colors, profile, ImageCms.createProfile("sRGB"), outputMode="RGB")
            except (OSError, ValueError, ImageCms.PyCMSError) as exc:
                raise MediaError(f"Invalid/unsupported image ICC profile: {exc}") from exc
        if image.mode.startswith("I;16") or image.mode == "I":
            gray = np.asarray(image, dtype=np.float32) / np.float32(65535)
            rgb = np.repeat(np.clip(gray, 0, 1)[..., None], 3, axis=-1)
        else:
            if not icc and image.mode in ("CMYK", "LAB"):
                raise MediaError("CMYK/LAB images require an ICC profile for SDR conversion")
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
        if facts.get("cicp") and not icc:
            primaries, trc = facts["cicp"]
            rgb = _to_srgb(rgb, trc, primaries)
    rgb, alpha = resize_rgb(rgb, alpha, max_edge)
    return dict(rgb=rgb, alpha=alpha, warnings=facts["warnings"], color_space="srgb",
                source_width=facts["width"], source_height=facts["height"])


def save_image(path, rgb, *, alpha=None, metadata=None):
    """Round once at the export boundary; always embed the actual sRGB profile."""
    import numpy as np
    from PIL import Image, ImageCms, PngImagePlugin
    pixels = np.rint(np.clip(rgb, 0, 1) * np.float32(255)).astype(np.uint8)
    if alpha is not None:
        pixels = np.concatenate((pixels, alpha[..., None]), axis=-1)
    info = PngImagePlugin.PngInfo()
    if metadata is not None:
        info.add_text("forge_nr", json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(pixels).save(path, format="PNG", icc_profile=profile, pnginfo=info)


def _av():
    try:
        import av
    except ImportError as exc:
        raise MediaError("Missing CPU media dependency PyAV (av>=16,<19); no GPU/runtime was initialized") from exc
    if not 16 <= int(av.__version__.split(".", 1)[0]) < 19:
        raise MediaError("Supported CPU PyAV versions are >=16,<19")
    return av


def _note(warnings, message):
    if message not in warnings:
        warnings.append(message)


def _color_info(frame, stream, warnings, *, gif=False):
    """Declare assumptions, reject HDR/wide gamut; never quietly relabel HDR SDR."""
    values = {}
    for name in ("colorspace", "color_range", "color_trc", "color_primaries"):
        value = int(getattr(frame, name, 2 if name != "color_range" else 0))
        unknown = (0,) if name == "color_range" else (2,)
        if value in unknown:
            value = int(getattr(stream.codec_context, name, value))
        values[name] = value
    if (values["color_trc"] in (14, 15, 16, 18) or values["color_primaries"] == 9
            or values["colorspace"] in (9, 10)):
        raise MediaError("HDR/PQ/HLG/BT.2020 is not supported by the SDR NR pipeline; convert explicitly before import")
    rgb = bool(frame.format.is_rgb) or gif or frame.format.name == "pal8"
    if values["color_trc"] in (0, 2):
        values["color_trc"] = 13 if rgb else 1
        _note(warnings, "Unspecified SDR transfer: explicitly assumed sRGB" if rgb else "Unspecified SDR transfer: explicitly assumed BT.709")
    if values["color_primaries"] in (0, 2):
        values["color_primaries"] = 1
        _note(warnings, "Unspecified SDR primaries: explicitly assumed BT.709/sRGB")
    if values["color_range"] == 0:
        values["color_range"] = 2 if rgb or frame.format.name.startswith("yuvj") else 1
        _note(warnings, "Unspecified sample range: assumed full RGB/JPEG" if values["color_range"] == 2 else "Unspecified sample range: assumed limited YUV")
    if values["colorspace"] in (0, 2):
        values["colorspace"] = 1 if rgb else 5
        if not rgb:
            _note(warnings, "Unspecified YUV matrix: explicitly assumed BT.601 (FFmpeg default), not guessed by image size")
    if values["color_trc"] not in (1, 4, 5, 6, 7, 8, 13):
        raise MediaError(f"Unsupported SDR transfer characteristic {values['color_trc']}")
    if values["color_primaries"] not in (1, 5, 6):
        raise MediaError(f"Unsupported color primaries {values['color_primaries']}; explicit sRGB/709 conversion is required")
    if values["colorspace"] not in (1, 4, 5, 6, 7):
        raise MediaError(f"Unsupported YUV matrix {values['colorspace']}")
    return values


def _to_srgb(encoded, trc, primaries):
    import numpy as np
    if trc == 13 and primaries == 1:
        return np.ascontiguousarray(encoded, dtype=np.float32)
    u = np.clip(encoded, 0, 1)
    if trc == 13:
        linear = np.where(u <= .04045, u / 12.92, ((u + .055) / 1.055) ** 2.4)
    elif trc in (1, 6):
        linear = np.where(u < .081, u / 4.5, ((u + .099) / 1.099) ** (1 / .45))
    elif trc == 7:
        linear = np.where(u < .0912, u / 4., ((u + .1115) / 1.1115) ** (1 / .45))
    elif trc in (4, 5):
        linear = u ** (2.2 if trc == 4 else 2.8)
    else:
        linear = u
    if primaries != 1:
        # D65 EBU/SMPTE-C -> XYZ -> sRGB; transfer conversion alone does not
        # change primaries. Matrices use published standard chromaticities.
        xyz = ([[.4306190, .3415419, .1783091], [.2220379, .7066384, .0713236], [.0201853, .1295504, .9390944]]
               if primaries == 5 else
               [[.3935209, .3652581, .1916769], [.2123764, .7010599, .0865638], [.0187391, .1119339, .9583847]])
        inverse = [[3.2404542, -1.5371385, -.4985314], [-.9692660, 1.8760108, .0415560], [.0556434, -.2040259, 1.0572252]]
        matrix = np.asarray(inverse, np.float32) @ np.asarray(xyz, np.float32)
        linear = linear @ matrix.T
    linear = np.clip(linear, 0, 1)
    return np.ascontiguousarray(np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055), dtype=np.float32)


def _srgb_to_709(rgb):
    import numpy as np
    u = np.clip(rgb, 0, 1)
    linear = np.where(u <= .04045, u / 12.92, ((u + .055) / 1.055) ** 2.4)
    return np.ascontiguousarray(np.where(linear < .018, linear * 4.5, 1.099 * linear ** .45 - .099), dtype=np.float32)


@dataclass
class MediaFrame:
    rgb: object
    pts: Fraction                 # Relative to the first decoded video PTS.
    duration: Fraction            # Last frame has its own duration, not just PTS.
    absolute_pts: Fraction        # Original container coordinate, for audio trim.
    alpha: object = None


class MediaReader:
    """One-pass software decoder with a single-frame lookahead, preserving VFR.

    frames() selects frame starts in [start,end), relative to first decoded video
    PTS. If a selection begins within a frame, its next frame start is used and
    reported as actual_start. The last duration is clipped at an explicit end.
    No FPS grid, hardware decoder, model, HTTP endpoint or frame batch is used.
    """
    def __init__(self, path, *, start=0, end=0, max_edge=0, cancel_event=None, limits=None, progress=None):
        self.path = local_path(path)
        self.start, self.end = Fraction(str(start)), Fraction(str(end)) if end else None
        if self.start < 0 or (self.end is not None and self.end <= self.start):
            raise MediaError("Invalid media selection")
        self.max_edge, self.cancel_event = max_edge, cancel_event
        self.limits, self.progress = _limits(limits), progress
        self._last_progress = time.monotonic()
        self.warnings = []
        self.container = None
        self._consumed = False
        self.selected_frames = 0
        self.selected_end = None
        self._gif = self.path.suffix.lower() == ".gif"
        check_cancel(cancel_event)
        av = _av()
        try:
            options = {"protocol_whitelist": "file", "format_whitelist": "mov,mp4,matroska,webm,avi,gif,apng",
                       "ignore_loop": "1", "min_delay": "0", "default_delay": "1"}
            self.container = av.open(str(self.path), "r", options=options, timeout=(15., 15.))
            streams = [s for s in self.container.streams.video if not (int(s.disposition) & 0x400)]
            if not streams:
                raise MediaError("Media contains no video stream")
            self.stream = streams[0]
            codec = self.stream.codec_context
            if codec.width and codec.height:
                _dimensions(codec.width, codec.height, self.limits)
            self.stream.codec_context.thread_count = 2
            self.stream.thread_type = "SLICE"
            self.has_audio = bool(self.container.streams.audio)
            self.audio_rate = self.container.streams.audio[0].codec_context.sample_rate if self.has_audio else None
            self._decode = self.container.decode(self.stream)
            self._first_frame = self._next()
            if self._first_frame is None or self._first_frame.pts is None:
                raise MediaError("Video has no decodable frame with a presentation timestamp")
            self.time_base = Fraction(self._first_frame.time_base)
            self.first_pts = self._first_frame.pts * self.time_base
            self._raw_size = (self._first_frame.width, self._first_frame.height)
            _dimensions(*self._raw_size, self.limits)
            self.rotation = int(getattr(self._first_frame, "rotation", 0)) % 360
            if self.rotation not in (0, 90, 180, 270):
                raise MediaError("Non-orthogonal video rotation requires an explicit source conversion")
            self.width, self.height = self._raw_size[::-1] if self.rotation in (90, 270) else self._raw_size
            _color_info(self._first_frame, self.stream, self.warnings, gif=self._gif)
        except BaseException:
            self.close()
            raise

    def _next(self):
        check_cancel(self.cancel_event)
        began = time.monotonic()
        frame = next(self._decode, None)
        check_cancel(self.cancel_event)
        # Cooperative per-decode stall guard, NOT an arbitrary whole-video
        # deadline. A legal long scan can keep making progress indefinitely.
        # A blocked C decoder remains subject to the parent's process timeout.
        if time.monotonic() - began > MEDIA_DECODE_STALL_SECONDS:
            raise MediaError("CPU video decoder stalled beyond its progress deadline")
        return frame

    def timeline(self):
        """Yield (decoded CPU frame, absolute PTS, effective duration), once."""
        if self._consumed:
            raise MediaError("MediaReader is a streaming one-pass reader")
        self._consumed = True
        current, previous_interval = self._first_frame, None
        count = 0
        while current is not None:
            check_cancel(self.cancel_event)
            count += 1
            if count > self.limits["max_frames"]:
                raise MediaError("Media frames exceed the allowed limit")
            if current.pts is None or not current.time_base:
                raise MediaError("Missing video PTS; refusing to invent a constant FPS timeline")
            if (current.width, current.height) != self._raw_size:
                raise MediaError("Changing video dimensions require explicit source normalization")
            _color_info(current, self.stream, self.warnings, gif=self._gif)
            if current.interlaced_frame:
                _note(self.warnings, "Interlaced source: no implicit deinterlacing is performed")
            absolute = Fraction(current.pts) * current.time_base
            if absolute - self.first_pts >= self.limits["max_duration"]:
                raise MediaError("Media duration exceeds the allowed limit")
            following = self._next()
            if following is not None:
                if following.pts is None:
                    raise MediaError("Missing next-frame PTS")
                duration = Fraction(following.pts) * following.time_base - absolute
                if duration <= 0:
                    raise MediaError("Video PTS must be strictly increasing; duplicate frames are not silently dropped")
                previous_interval = duration
            else:
                raw_duration = getattr(current, "duration", 0)
                duration = Fraction(raw_duration) * current.time_base if raw_duration else Fraction(0)
                if duration <= 0 and self.stream.duration:
                    origin = self.stream.start_time * self.stream.time_base if self.stream.start_time is not None else self.first_pts
                    duration = origin + self.stream.duration * self.stream.time_base - absolute
                if duration <= 0 and previous_interval is not None:
                    duration = previous_interval
                    _note(self.warnings, "Last-frame duration unavailable: estimated from the last observed PTS interval")
                if duration <= 0:
                    raise MediaError("Last frame has no usable duration; refusing to invent a playback rate")
            if absolute - self.first_pts + duration > self.limits["max_duration"]:
                raise MediaError("Media duration exceeds the allowed limit")
            now = time.monotonic()
            if self.progress and now - self._last_progress >= 1.:
                self.progress(f"Inspecting/decoding CPU video: {count} frames", 0.)
                self._last_progress = now
                check_cancel(self.cancel_event)
            yield current, absolute, duration
            current = following

    def _pixels(self, frame):
        import numpy as np
        from av.video.reformatter import VideoReformatter, Interpolation
        colors = _color_info(frame, self.stream, self.warnings, gif=self._gif)
        bits = max(component.bits for component in frame.format.components)
        if (frame.format.is_rgb or self._gif or frame.format.name == "pal8") and bits <= 8:
            # Already-RGB must not take an 8->16 swscale promotion: it shifts by
            # 8 rather than consistently replicating bits, changing /255 levels.
            rgb = frame.to_ndarray(format="rgb24").astype(np.float32) / np.float32(255)
        else:
            # For YUV conversion planar RGB16 avoids packed RGB24's known bias.
            planar = VideoReformatter().reformat(frame, format="gbrp16le", src_colorspace=colors["colorspace"],
                                                  dst_colorspace=1, src_color_range=colors["color_range"], dst_color_range=2,
                                                  interpolation=Interpolation.BICUBIC | Interpolation.ACCURATE_RND | Interpolation.FULL_CHR_H_INT)
            planes = [np.frombuffer(plane, dtype="<u2").reshape(plane.height, plane.line_size // 2)[:frame.height, :frame.width]
                      for plane in planar.planes[:3]]
            rgb = np.stack((planes[2], planes[0], planes[1]), axis=-1).astype(np.float32) / np.float32(65535)
        rgb = _to_srgb(rgb, colors["color_trc"], colors["color_primaries"])
        alpha = None
        if any(component.is_alpha for component in frame.format.components):
            alpha = frame.to_ndarray(format="rgba")[..., 3].copy()
            if np.all(alpha == 255):
                alpha = None
        if self.rotation:
            rgb = np.rot90(rgb, self.rotation // 90)
            if alpha is not None:
                alpha = np.rot90(alpha, self.rotation // 90).copy()
        return resize_rgb(rgb, alpha, self.max_edge)

    def frames(self):
        for frame, absolute, duration in self.timeline():
            relative = absolute - self.first_pts
            if relative < self.start:
                continue
            if self.end is not None and relative >= self.end:
                break
            if self.end is not None:
                duration = min(duration, self.end - relative)
            check_cancel(self.cancel_event)
            rgb, alpha = self._pixels(frame)
            self.selected_frames += 1
            self.selected_end = relative + duration
            yield MediaFrame(rgb, relative, duration, absolute, alpha)

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _WindowsProcessTree:
    """Per-call job, attached while suspended so launchers cannot escape it.

    Do not use the application's shared job: cancellation owns this call only.
    ActiveProcesses==0 is required before releasing inherited output handles.
    """
    def __init__(self):
        import ctypes
        from .procs import _kernel32, _EXTENDED_LIMIT
        self.ctypes = ctypes
        self.kernel = k = _kernel32()
        k.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k.TerminateJobObject.restype = ctypes.c_int
        k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                               ctypes.c_ulong, ctypes.c_void_p]
        k.QueryInformationJobObject.restype = ctypes.c_int
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError(ctypes.get_last_error())
            k.CloseHandle(self.handle)
            raise error

    def start(self, process):
        c = self.ctypes
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise c.WinError(c.get_last_error())
        # Popen closes the primary thread handle. Resume the suspended process
        # through its owned process handle, not a racy PID/thread enumeration.
        resume = c.WinDLL("ntdll").NtResumeProcess
        resume.argtypes, resume.restype = [c.c_void_p], c.c_long
        status = resume(int(process._handle))
        if status < 0:
            raise MediaError(f"Cannot resume owned CPU process: NTSTATUS {status:#x}")

    def close(self):
        c = self.ctypes

        class Accounting(c.Structure):
            _fields_ = [(name, c.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")] + [
                (name, c.c_ulong) for name in ("faults", "total", "active", "terminated")]

        # Job accounting drops ActiveProcesses before every exiting process has
        # finished closing its handle table. Hold real process handles and wait
        # for their signaled state as well, especially the basepython child.
        handles = []
        wait = self.kernel.WaitForSingleObject
        wait.argtypes, wait.restype = [c.c_void_p, c.c_ulong], c.c_ulong
        try:
            size = 4096
            while True:
                data = c.create_string_buffer(size)
                if self.kernel.QueryInformationJobObject(self.handle, 3, data, size, None):
                    break
                if c.get_last_error() != 234:  # ERROR_MORE_DATA
                    raise c.WinError(c.get_last_error())
                size *= 2
            count = c.c_ulong.from_buffer(data, 4).value
            for pid in (c.c_size_t * count).from_buffer(data, 8):
                handle = self.kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
                if handle:
                    handles.append(handle)
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise c.WinError(c.get_last_error())
            deadline = time.monotonic() + 3.
            for handle in handles:
                if wait(handle, max(0, round((deadline - time.monotonic()) * 1000))) != 0:
                    raise MediaError("Owned CPU subprocess did not finish exiting")
            while True:
                info = Accounting()
                if not self.kernel.QueryInformationJobObject(self.handle, 1, c.byref(info), c.sizeof(info), None):
                    raise c.WinError(c.get_last_error())
                if not info.active:
                    return
                if time.monotonic() >= deadline:
                    raise MediaError("Owned CPU process tree did not exit after termination")
                time.sleep(.01)
        finally:
            for handle in handles:
                self.kernel.CloseHandle(handle)
            self.kernel.CloseHandle(self.handle)


def run_process(args, *, cancel_event, log_path, timeout=180.):
    """Run a bounded, cancellable CPU subprocess with no shell or inherited bad env."""
    from .procs import launch_env
    check_cancel(cancel_event)
    started = time.monotonic()
    tree = _WindowsProcessTree() if os.name == "nt" else None
    process = None
    try:
        with Path(log_path).open("wb") as log:
            options = {"creationflags": 0x4} if tree else {"start_new_session": True}
            process = subprocess.Popen([str(arg) for arg in args], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=log, env=launch_env(), **options)
            if tree:
                tree.start(process)
            while process.poll() is None:
                check_cancel(cancel_event)
                if time.monotonic() - started > timeout:
                    raise MediaError("CPU ffmpeg/subprocess exceeded its deadline")
                if cancel_event is None:
                    time.sleep(.05)
                else:
                    cancel_event.wait(.05)
            check_cancel(cancel_event)
            if process.returncode:
                log.flush()
                detail = Path(log_path).read_bytes()[-6000:].decode("utf-8", "replace")
                raise MediaError(f"CPU ffmpeg/subprocess failed ({process.returncode}): {detail}")
    finally:
        try:
            if tree:
                tree.close()
            elif process is not None:
                import signal
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3.)


class VideoWriter:
    """Software H.264, explicit BT.709/limited color, exact input PTS/durations.

    rate is only an encoder hint. Frames are individually timestamped; neither
    PyAV nor an ffmpeg FPS filter duplicates/drops frames. B-frames are disabled
    so the final packet's independent duration can be set before muxing.
    """
    def __init__(self, path, width, height, *, time_base, rate):
        av = _av()
        self.time_base = Fraction(time_base)
        self.container = av.open(str(path), "w", format="mp4", options={"movflags": "+faststart", "avoid_negative_ts": "disabled"})
        self.stream = self.container.add_stream("libx264", rate=Fraction(rate).limit_denominator(100000))
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv444p" if width % 2 or height % 2 else "yuv420p"
        self.stream.time_base = self.stream.codec_context.time_base = self.time_base
        self.stream.codec_context.max_b_frames = 0
        self.stream.codec_context.thread_count = 2
        for name, value in (("colorspace", 1), ("color_range", 1), ("color_trc", 1), ("color_primaries", 1)):
            setattr(self.stream.codec_context, name, value)
        self.stream.options = {"crf": "17", "preset": "fast", "tune": "zerolatency", "bf": "0"}
        self._durations = {}
        self._last_pts = None
        self.closed = False

    def _mux(self, packet):
        if packet.pts is None or packet.time_base is None:
            raise MediaError("Encoder returned a packet without PTS")
        stamp = packet.pts * packet.time_base
        if stamp not in self._durations:
            raise MediaError("Encoder unexpectedly resampled/reordered the video timeline")
        packet.duration = max(1, round(self._durations.pop(stamp) / packet.time_base))
        self.container.mux(packet)

    def write(self, rgb, pts, duration):
        import numpy as np
        from av.video.reformatter import VideoReformatter, Interpolation
        av = _av()
        integer_pts = round(pts / self.time_base)
        if self._last_pts is not None and integer_pts <= self._last_pts:
            raise MediaError("Encoder time base cannot express strictly increasing source PTS")
        self._last_pts = integer_pts
        # Both tracks and both comparison columns take precisely this conversion.
        # Metadata fields alone do NOT perform a transfer-function conversion.
        rgb709 = _srgb_to_709(rgb)
        pixels = np.rint(np.clip(rgb709, 0, 1) * np.float32(65535)).astype(np.uint16)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb48le")
        yuv = VideoReformatter().reformat(frame, format=self.stream.pix_fmt, src_colorspace=1, dst_colorspace=1,
                                          src_color_range=2, dst_color_range=1,
                                          interpolation=Interpolation.BICUBIC | Interpolation.ACCURATE_RND | Interpolation.FULL_CHR_H_INT)
        yuv.time_base, yuv.pts = self.time_base, integer_pts
        yuv.duration = max(1, round(duration / self.time_base))
        for name in ("colorspace", "color_range", "color_trc", "color_primaries"):
            setattr(yuv, name, 1)
        self._durations[integer_pts * self.time_base] = Fraction(duration)
        for packet in self.stream.encode(yuv):
            self._mux(packet)

    def close(self, *, complete=True):
        if self.closed:
            return
        self.closed = True
        try:
            if complete:
                for packet in self.stream.encode():
                    self._mux(packet)
                if self._durations:
                    raise MediaError("Encoder did not return every submitted video frame")
        finally:
            self.container.close()


def verify_video(path, expected_pts, expected_duration, *, cancel_event, expect_audio=False,
                 expected_frame_count=None, progress=None):
    """Fully decode after encoding/muxing, checking actual frame PTS and color."""
    if expected_frame_count is not None and len(expected_pts) != expected_frame_count:
        raise MediaError("Processed frames disagree with the decoded source selection")
    with MediaReader(path, cancel_event=cancel_event, progress=progress) as reader:
        frames, duration, previous = 0, Fraction(0), None
        min_step = max_step = None
        tolerance = max(float(reader.time_base), .000002)
        if abs(float(reader.first_pts)) > tolerance:
            raise MediaError("Encoded video first PTS is not zero (audio/container timestamp shift)")
        for frame, absolute, length in reader.timeline():
            for name in ("colorspace", "color_range", "color_trc", "color_primaries"):
                if int(getattr(frame, name)) != 1:
                    raise MediaError(f"Encoded output is not explicitly BT.709/limited: {name}")
            relative = absolute - reader.first_pts
            if frames >= len(expected_pts) or abs(float(relative - expected_pts[frames])) > tolerance:
                raise MediaError("Encoded output did not preserve source VFR presentation timestamps")
            if previous is not None:
                step = relative - previous
                min_step = step if min_step is None else min(min_step, step)
                max_step = step if max_step is None else max(max_step, step)
            previous = relative
            duration = relative + length
            frames += 1
        if frames != len(expected_pts) or abs(float(duration - expected_duration)) > tolerance * 2:
            raise MediaError("Encoded output lost frames or changed the final frame duration")
        summary = dict(frames=frames, duration=float(duration), fps=float(frames / duration),
                       width=reader.width, height=reader.height, has_audio=reader.has_audio,
                   timeline=dict(frame_count=frames, first_pts=0., last_pts=float(previous),
                         min_step=float(min_step) if min_step is not None else None,
                         max_step=float(max_step) if max_step is not None else None))
    # A valid video track is not evidence that all muxed audio decodes.
    with _av().open(str(path), options={"protocol_whitelist": "file"}) as container:
        if expect_audio and not container.streams.audio:
            raise MediaError("Encoded output lost the required source audio track")
        if container.streams.audio:
            stream = container.streams.audio[0]
            count = 0
            audio_start = audio_end = None
            last_progress = time.monotonic()
            for frame in container.decode(stream):
                check_cancel(cancel_event)
                if frame.pts is None or not frame.time_base or not frame.sample_rate:
                    raise MediaError("Muxed audio frame has no usable timestamp/sample clock")
                stamp = Fraction(frame.pts) * frame.time_base
                if audio_start is None:
                    audio_start = stamp
                audio_end = stamp + Fraction(frame.samples, frame.sample_rate)
                count += frame.samples
                if progress and time.monotonic() - last_progress >= 1.:
                    progress("Verifying synchronized audio samples", 0.)
                    last_progress = time.monotonic()
            if not count:
                raise MediaError("Muxed audio stream has no decodable samples")
            rate = stream.codec_context.sample_rate
            if expect_audio and (audio_start > Fraction(1, rate)
                                 or audio_end < expected_duration - Fraction(1, rate)):
                raise MediaError("Muxed audio does not cover the selected video duration")
            summary["audio"] = dict(sample_rate=rate, decoded_samples=count,
                                    first_pts=float(audio_start), end_pts=float(audio_end))
    return summary


def _mux_audio(silent, source, output, *, start, duration, sample_rate, cancel_event, log_path):
    executable = shutil.which("ffmpeg")
    if not executable:
        candidate = Path("C:/ffmpeg/bin/ffmpeg.exe")
        executable = str(candidate) if candidate.is_file() else None
    if not executable:
        raise MediaError("Missing CPU ffmpeg for accurately trimmed AAC audio; refusing a silently muted result")
    if not sample_rate or sample_rate <= 0:
        raise MediaError("Source audio has no usable sample rate")
    first, last, length = (f"{float(value):.12f}" for value in (start, start + duration, duration))
    # -copyts preserves the source container coordinate. Unlike STARTPTS, this
    # subtracts the first *selected video frame* absolute PTS, keeping late audio
    # late. aresample materializes any leading silence on the zero-based timeline.
    filters = (f"atrim=start={first}:end={last},asetpts=PTS-({first})/TB,"
               f"aresample=async=1:first_pts=0,apad=whole_dur={length},atrim=duration={length}")
    args = [str(Path(executable).resolve()), "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-copyts",
            "-protocol_whitelist", "file", "-i", str(silent), "-protocol_whitelist", "file", "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-vsync", "0",
            "-af", filters, "-c:a", "aac", "-b:a", "192k", "-ar", str(sample_rate),
            "-avoid_negative_ts", "disabled", "-use_editlist", "1", "-movflags", "+faststart", str(output)]
    run_process(args, cancel_event=cancel_event, log_path=log_path)


def publish_images(directory, result, *, mixed, original, neural, alpha, cancel_event):
    """Atomic-at-publication image staging shared by images and video stills."""
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = result["files"]
    staging = [directory / (".nr-" + name) for name in names]
    try:
        for name, pixels in zip(names, (mixed, original, neural)):
            check_cancel(cancel_event)
            pending = directory / (".nr-" + name)
            save_image(pending, pixels, alpha=alpha, metadata=result)
            with Image.open(pending) as image:
                image.verify()
        check_cancel(cancel_event)
        for name, pending in zip(names, staging):
            pending.replace(directory / name)
        check_cancel(cancel_event)
    except BaseException:
        for path in [*staging, *(directory / name for name in names)]:
            path.unlink(missing_ok=True)
        raise


def process_video(command, *, facts, native_begin, process_frame, progress, cancel_event):
    """Public media pipeline; native renderer is injected only as Python calls."""
    import numpy as np
    preview = command["preview_kind"]
    still = preview == "still"
    end = min(command["end"] or facts["duration"], facts["duration"])
    if command["start"] >= end:
        raise MediaError("Selection contains no video frames")
    if preview == "clip" and end - command["start"] > 10.0000001:
        raise MediaError("Continuous NR previews are limited to 10 seconds")
    directory = Path(command["output_dir"])
    temporal = bool(command["params"]["flow"]) and not still
    names = (["output.png", "original.png", "neural.png"] if still else
             ["output.mp4", "original.mp4", "comparison.mp4"] if preview else ["output.mp4"])
    generated = []  # Empty until this request owns the output directory.
    writers = []
    try:
        with MediaReader(command["source_path"], start=command["start"], end=command["end"],
                         max_edge=command["max_edge"], cancel_event=cancel_event, progress=progress) as reader:
            if ((reader.width, reader.height) != (facts["width"], facts["height"])
                    or ("source_first_pts" in facts and abs(float(reader.first_pts) - facts["source_first_pts"]) > float(reader.time_base))
                    or ("has_audio" in facts and reader.has_audio != facts["has_audio"])):
                raise MediaError("Decoded source dimensions/origin/audio disagree with registered source facts")
            iterator = reader.frames()
            first = next(iterator, None)
            if first is None:
                raise MediaError("Selection contains no video frame starts")
            check_cancel(cancel_event)
            native_info = native_begin(temporal=temporal) or {}
            height, width = first.rgb.shape[:2]
            scaled = (width, height) != (facts["width"], facts["height"])
            approximate = bool(preview and (scaled or first.pts > 0 or (still and command["params"]["flow"])))
            notes = []
            if scaled:
                notes.append("缩小预览（近似，不裁切）")
            if still:
                notes.append("静帧检查，无时序历史；不等于连续片段")
            elif preview and first.pts > 0:
                notes.append("连续预览从所选起点重置历史，与整片该时刻近似")
            warnings = list(dict.fromkeys([*facts.get("warnings", []), *reader.warnings, *native_info.get("warnings", [])]))
            selection = dict(start=command["start"], end=command["end"], actual_start=float(first.pts),
                             source_first_pts=float(reader.first_pts), first_selected_pts=float(first.absolute_pts))
            result = dict(name=names[0], kind="image" if still else "video", width=width, height=height,
                          duration=0., frames=0, fps=None, files=names, warnings=warnings, parent=command["source"],
                          params=command["params"], runtime_id=command["runtime"].get("runtime_id", ""),
                          selection=selection, preview=bool(preview), approximate=approximate,
                          note="；".join(notes) or "原尺寸，独立历史", color_space="srgb" if still else "bt709",
                          mix_space="display-encoded sRGB float32", native=native_info)
            if preview:
                result["original_name"] = names[1]
                result["neural_name" if still else "comparison_name"] = names[2]
            if still:
                neural, mixed = process_frame(first.rgb, reset=True, temporal=False)
                result.update(frames=1)
                selection["actual_end"] = float(first.pts + first.duration)
                publish_images(directory, result, mixed=mixed, original=first.rgb, neural=neural,
                               alpha=first.alpha, cancel_event=cancel_event)
                generated = [directory / name for name in names]
                progress("Verified SDR still preview", 1.)
                check_cancel(cancel_event)
                return result
            directory.mkdir(parents=True, exist_ok=True)
            staging = [directory / (".nr-silent-" + name) for name in names]
            ready = [directory / (".nr-ready-" + name) for name in names]
            generated = [*staging, *ready, *(directory / name for name in names), directory / ".nr-ffmpeg.log"]
            hint = Fraction(str(facts["fps"])) if facts.get("fps") else 1 / first.duration
            for i, path in enumerate(staging):
                writers.append(VideoWriter(path, width * (2 if i == 2 else 1), height, time_base=reader.time_base, rate=hint))
            if width % 2 or height % 2:
                _note(warnings, "Odd source dimensions preserved with H.264 yuv444p; browser playback support may vary")
            stamps, durations, frame = [], [], first
            while frame is not None:
                check_cancel(cancel_event)
                original = frame.rgb
                if frame.alpha is not None:
                    original = np.ascontiguousarray(original * (frame.alpha[..., None].astype(np.float32) / 255))
                    _note(warnings, "Animated transparency is composited on black for opaque MP4; still PNG previews preserve alpha")
                neural, mixed = process_frame(original, reset=not stamps, temporal=temporal)
                pts = frame.absolute_pts - first.absolute_pts
                check_cancel(cancel_event)
                tracks = [mixed]
                if preview:
                    tracks.extend([original, np.concatenate((original, mixed), axis=1)])
                for writer, pixels in zip(writers, tracks):
                    check_cancel(cancel_event)
                    writer.write(pixels, pts, frame.duration)
                stamps.append(pts)
                durations.append(frame.duration)
                span = pts + frame.duration
                progress(f"Processed {len(stamps)} video frames", min(.92, .92 * float(span) / max(.001, end - float(first.pts))))
                frame = next(iterator, None)
            duration = stamps[-1] + durations[-1]
            tolerance = max(float(reader.time_base), .000002) * 2
            if (len(stamps) != reader.selected_frames or reader.selected_end is None
                    or abs(float(first.pts + duration - reader.selected_end)) > tolerance):
                raise MediaError("Processed output lost frames from the decoded source selection")
            # Never accept a shorter decode just because its own output is
            # self-consistent. Selected duration comes from immutable facts;
            # frame count comes from this one-pass decoder, not nominal FPS.
            if abs(float(first.pts + duration) - end) > tolerance:
                raise MediaError("Decoded source ended before/after the requested selection duration")
            whole_source = command["start"] == 0 and (not command["end"] or command["end"] >= facts["duration"])
            if whole_source and len(stamps) != facts["frames"]:
                raise MediaError("Decoded source frame count disagrees with registered source facts")
            selection["actual_end"] = float(first.pts + duration)
            sample_rate, has_audio = reader.audio_rate, reader.has_audio
            selected_count = reader.selected_frames
            for warning in reader.warnings:
                _note(warnings, warning)
            for writer in writers:
                check_cancel(cancel_event)
                writer.close()
        progress("Encoding complete; verifying video and synchronized audio", .94)
        for silent, destination in zip(staging, ready):
            check_cancel(cancel_event)
            if has_audio:
                _mux_audio(silent, command["source_path"], destination, start=first.absolute_pts, duration=duration,
                           sample_rate=sample_rate, cancel_event=cancel_event, log_path=directory / ".nr-ffmpeg.log")
            else:
                silent.replace(destination)
            summary = verify_video(destination, stamps, duration, cancel_event=cancel_event,
                                   expect_audio=has_audio, expected_frame_count=selected_count,
                                   progress=lambda message, ratio: progress(message, .94))
            if destination == ready[0]:
                result.update(summary)
        check_cancel(cancel_event)
        for name, path in zip(names, ready):
            path.replace(directory / name)
        for path in [*staging, directory / ".nr-ffmpeg.log"]:
            path.unlink(missing_ok=True)
        progress("Verified every output frame and audio stream", 1.)
        check_cancel(cancel_event)
        return result
    except BaseException:
        for writer in writers:
            try:
                writer.close(complete=False)
            except Exception:
                pass
        for path in generated:
            path.unlink(missing_ok=True)
        raise
"""CPU-only public Forge wrapper/adapter tests. No server or installed extension writes."""
import sys
from pathlib import Path
import socket
import subprocess
import copy
import hashlib
import json
import tempfile
import os
import importlib.abc
from contextlib import ExitStack
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

NATIVE_TORCH = "--native-torch" in sys.argv
NATIVE_GRADIO = "--native-gradio" in sys.argv
sys.argv[:] = [arg for arg in sys.argv if arg not in ("--native-torch", "--native-gradio")]
REAL_TORCH = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

from nr_shared.contract import SCRIPT_TITLE, script_args, make_spec, DEFAULT_PARAMS, validate_receipt

RUNTIME = dict(bridge="D:/fake/bridge.dll", runtime_dir="D:/fake/runtime", gpu_index=1,
               device="cuda:1", gpu_name="Test NVIDIA", channel_order="RGBA", runtime_id="a" * 64)


class Tensor(np.ndarray):
    """Numpy tensor at the CPU Torch boundary, never a modules/torch import stub."""
    device = "cpu"

    def cpu(self):
        return self

    def detach(self):
        return self

    def float(self):
        return self.astype(np.float32)

    def to(self, *args, **kwargs):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(a):
    if REAL_TORCH is not None:
        return REAL_TORCH.from_numpy(np.asarray(a, dtype=np.float32))
    return np.asarray(a, dtype=np.float32).view(Tensor)


TORCH = SimpleNamespace(stack=lambda a: tensor(np.stack(a)), from_numpy=tensor,
                        clamp=lambda a, min, max: tensor(np.clip(a, min, max)), float32=np.float32)


class PNGClient:
    """Shared-client boundary: actual local PNG IO and nonlinear pixel transform."""
    def __init__(self):
        self.commands = []
        self.error = None
        self.mutate = lambda: None
        self.corrupt = lambda result: None
        self.process_pixels = lambda pixels, params: np.where(pixels >= 128, 224, 16).astype(np.uint8)

    def runtime(self, device=None):
        return dict(ready=True, missing=[], runtime=copy.deepcopy(RUNTIME), runtime_id=RUNTIME["runtime_id"], max_passes=3)

    def status(self):
        return dict(instance="shared-test", pid=73, worker_pid=73, busy=False, running=True)

    def run(self, command, progress, cancel_event):
        self.commands.append(copy.deepcopy(command))
        self.mutate()
        if self.error:
            raise self.error
        if cancel_event.is_set():
            raise RuntimeError("NR canceled")
        src, out = Path(command["source_path"]), Path(command["output_dir"])
        assert out.is_dir() and not list(out.iterdir())
        assert hashlib.sha256(src.read_bytes()).hexdigest() == command["source"]["sha256"]
        with Image.open(src) as im:
            assert im.info.get("icc_profile"), "input must explicitly be sRGB"
            a = np.asarray(im.convert("RGB"))
        # Nonlinear, so NR(resize(x)) differs from resize(NR(x)).
        pixels = a
        for params in command.get("passes", [command["params"]]):
            pixels = self.process_pixels(pixels, params)
        Image.fromarray(pixels).save(out / "output.png")
        progress({"message": "Synthetic NR", "ratio": 1.})
        result = dict(name="output.png", width=a.shape[1], height=a.shape[0], kind="image",
                      params=copy.deepcopy(command["params"]), runtime_id=RUNTIME["runtime_id"],
                      native=dict(hardware_verified=True, gpu_name=RUNTIME["gpu_name"], device="cuda:1"),
                      instance="shared-test", worker_pid=73, preview=False, approximate=False)
        if "passes" in command:
            result.update(passes=copy.deepcopy(command["passes"]), completed_passes=len(command["passes"]),
                          pass_results=[dict(params=copy.deepcopy(params), native=copy.deepcopy(result["native"]))
                                        for params in command["passes"]])
        self.corrupt(result)
        return result


def host(client, directory, *, hires=True, stage="before_hr", latent=False):
    from forge_nr.adapter import ForgeAdapter
    from forge_nr.hooks import install

    class DecodedSamples(list):
        already_decoded = True

    class Img:
        def sample(self):
            return "ADetailer native img2img"

    class Txt:
        def sample(self):
            self.native_calls += 1
            if self.enable_hr:
                return self.sample_hr_pass(self.pixels * 2 - 1, None if self.latent_scale_mode else self.pixels)
            return self.pixels * 2 - 1

        def sample_hr_pass(self, samples, decoded_samples, *args, **kwargs):
            px = (samples + 1) / 2 if self.latent_scale_mode else decoded_samples
            result = []
            for image in px:
                data = np.rint(np.moveaxis(np.asarray(image), 0, 2) * 255).astype(np.uint8)
                resized = Image.fromarray(data).resize((5, 4), Image.Resampling.BILINEAR)
                # Synthetic second sampling pass, independently visible in oracle.
                data = np.minimum(np.asarray(resized).astype(int) + 24, 255).astype(np.uint8)
                result.append(tensor(np.moveaxis(data.astype(np.float32) / 255, 2, 0) * 2 - 1))
            return DecodedSamples(result)

    p = Txt()
    p.native_calls = 0
    p.enable_hr = hires
    p.iteration = 0
    p.n_iter = 1
    p.latent_scale_mode = {"mode": "nearest"} if latent else None
    p.hr_checkpoint_name = "Use same checkpoint"
    p.hr_additional_modules = ["Use same choices"]
    p.sd_model = SimpleNamespace(ini_latent=None)
    p.extra_generation_params = {}
    rgb = np.array([[[10, 80, 150], [170, 110, 230], [90, 200, 40]],
                    [[250, 120, 60], [50, 190, 130], [210, 20, 100]]], dtype=np.uint8)
    p.pixels = tensor([np.moveaxis(rgb.astype(np.float32) / 255, 2, 0)])
    spec = make_spec(True, stage, DEFAULT_PARAMS, RUNTIME, hires=hires)
    p.script_args = [0, *script_args(spec)]
    p.scripts = SimpleNamespace(alwayson_scripts=[SimpleNamespace(
        title=lambda: SCRIPT_TITLE, args_from=1, args_to=13)])
    processing = SimpleNamespace(StableDiffusionProcessingTxt2Img=Txt, StableDiffusionProcessingImg2Img=Img,
                                 DecodedSamples=DecodedSamples,
                                 decode_latent_batch=lambda model, a, **kw: DecodedSamples(list(a)),
                                 images_tensor_to_samples=lambda a, *args: a * 2 - 1,
                                 approximation_indexes={"Full": 0})
    state = SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False, textinfo="")
    shared = SimpleNamespace(state=state, device="cpu", opts=SimpleNamespace(sd_vae_encode_method="Full"))
    adapter = ForgeAdapter(processing, shared, SimpleNamespace(cpu="cpu"), TORCH, client, directory)
    install(processing, adapter)
    return p, spec, rgb, processing, adapter, state


def output_rgb(result):
    return np.rint((np.moveaxis(np.asarray(result), 1, -1) + 1) / 2 * 255).astype(np.uint8)


class ContractTests(unittest.TestCase):
    def test_three_pages_are_independent_and_stage_order_is_explicit(self):
        from nr_shared.contract import active_passes, default_passes, execution_summary, make_pass_spec, signature
        pages = default_passes()
        for index, record in enumerate(pages):
            record.update(enabled=True, stage="before_hr" if index == 1 else "after_hr")
            record["params"]["mix"] = .2 + index * .2
        spec = make_pass_spec(True, pages, RUNTIME, hires=True)
        self.assertEqual([record["index"] for record in active_passes(spec, "before_hr")], [2])
        self.assertEqual([record["index"] for record in active_passes(spec, "after_hr")], [1, 3])
        before = signature(spec)
        pages[1]["params"]["mix"] = 1.
        active_passes(spec)[1]["params"]["mix"] = 0.
        self.assertEqual(signature(spec), before)
        summary = execution_summary(spec, 2)
        self.assertEqual(summary["stage"], "mixed")
        self.assertEqual([record["index"] for record in summary["passes"]], [2, 1, 3])

    def test_legacy_arguments_round_trip_and_expansion_disables_extra_pages(self):
        from nr_shared.contract import ARG_KEYS, LEGACY_ARG_KEYS, from_script_args
        spec = make_spec(True, "before_hr", DEFAULT_PARAMS, RUNTIME, hires=False)
        legacy = script_args(spec)
        self.assertEqual(len(legacy), 12)
        expanded = script_args(spec, expanded=True)
        self.assertEqual(len(expanded), 35)
        self.assertEqual(ARG_KEYS[:12], LEGACY_ARG_KEYS)
        self.assertEqual(expanded[:12], legacy)
        self.assertIs(expanded[ARG_KEYS.index("pass_2_enabled")], False)
        self.assertIs(expanded[ARG_KEYS.index("pass_3_enabled")], False)
        for values in (legacy, expanded):
            self.assertEqual(from_script_args(values, hires=False), spec)
            values[0], values[11] = False, None
            self.assertIsNone(from_script_args(values, hires=False))

    def test_extended_snapshot_freezes_all_three_pages(self):
        from nr_shared.contract import default_passes, from_script_args, make_pass_spec
        pages = default_passes()
        pages[1].update(enabled=True, stage="after_hr")
        pages[2]["params"]["tone"] = .2
        spec = make_pass_spec(True, pages, RUNTIME, hires=True)
        values = script_args(spec)
        self.assertEqual(from_script_args(values, hires=True), spec)
        values[11]["gpu_index"] = 0
        self.assertEqual(spec["runtime"], RUNTIME)
        self.assertEqual(spec["passes"][2]["params"]["tone"], .2)

    def test_disabled_pages_and_master_never_require_runtime_or_hires(self):
        from nr_shared.contract import active_passes, default_passes, make_pass_spec
        pages = default_passes()
        pages[2]["stage"] = "after_hr"
        spec = make_pass_spec(True, pages, RUNTIME, hires=False)
        self.assertEqual(len(active_passes(spec)), 1)
        self.assertEqual(pages[1]["params"]["mix"], 1.)
        pages[0]["enabled"] = False
        self.assertIsNone(make_pass_spec(True, pages, None, hires=False))
        self.assertIsNone(make_pass_spec(False, None, None, hires=False))

    def test_invalid_pages_and_enabled_after_hr_without_hires_are_rejected(self):
        from nr_shared.contract import default_passes, make_pass_spec, validate_pass_params, validate_passes
        pages = default_passes()
        for invalid in ([], pages * 2, {}, [dict(enabled=1, stage="before_hr", params={})],
                        [dict(enabled=True, stage="before_hr", params={}, runtime=RUNTIME)]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_passes(invalid)
        for invalid in ([], [DEFAULT_PARAMS] * 4, [dict(DEFAULT_PARAMS, mix=float("nan"))]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_pass_params(invalid)
        pages[1].update(enabled=True, stage="after_hr")
        with self.assertRaisesRegex(ValueError, "高清"):
            make_pass_spec(True, pages, RUNTIME, hires=False)

    def test_receipt_requires_every_enabled_page_and_image_count(self):
        from nr_shared.contract import PROTOCOL, default_passes, execution_summary, make_pass_spec, signature
        pages = default_passes()
        pages[1]["enabled"] = True
        spec = make_pass_spec(True, pages, RUNTIME, hires=False)
        record = dict(protocol=PROTOCOL, status="done", signature=signature(spec), count=2,
                      **execution_summary(spec, 2))
        info = {"extra_generation_params": {SCRIPT_TITLE: record}}
        self.assertEqual(validate_receipt(info, spec, 2)["count"], 2)
        record["passes"][1]["count"] = 1
        with self.assertRaises(ValueError):
            validate_receipt(info, spec, 2)


class WrapperTests(unittest.TestCase):
    def test_legacy_worker_multipass_is_rejected_before_sampling(self):
        from nr_shared.contract import default_passes, make_pass_spec
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            processing, _, _, _, _, _ = host(client, directory, hires=False)
            pages = default_passes()
            pages[1]["enabled"] = True
            processing.script_args = [0, *script_args(make_pass_spec(True, pages, RUNTIME, hires=False))]
            processing.scripts.alwayson_scripts[0].args_to = len(processing.script_args)
            status = client.runtime()
            status.pop("max_passes")
            with patch.object(client, "runtime", return_value=status), self.assertRaisesRegex(RuntimeError, "多页支持"):
                processing.sample()
            self.assertEqual(processing.native_calls, 0)
            self.assertEqual(client.commands, [])

    def test_three_pages_split_stages_follow_page_order_and_keep_image_count(self):
        from nr_shared.contract import default_passes, make_pass_spec
        for latent in (False, True):
            with self.subTest(latent=latent), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                client = PNGClient()
                client.process_pixels = lambda pixels, params: np.clip(
                    np.rint(pixels.astype(np.float32) * (.5 + .2 * params["style"]) + 20 * params["tone"]),
                    0, 255).astype(np.uint8)
                processing, _, rgb, _, _, _ = host(client, directory, latent=latent)
                pages = default_passes()
                for index, record in enumerate(pages):
                    record.update(enabled=True, stage="before_hr" if index == 1 else "after_hr")
                    record["params"].update(style=index, tone=.4 + index * .3)
                spec = make_pass_spec(True, pages, RUNTIME, hires=True)
                processing.script_args = [0, *script_args(spec)]
                processing.scripts.alwayson_scripts[0].args_to = len(processing.script_args)
                expected = client.process_pixels(rgb, pages[1]["params"])
                expected = np.minimum(np.asarray(Image.fromarray(expected).resize((5, 4),
                    Image.Resampling.BILINEAR)).astype(int) + 24, 255).astype(np.uint8)
                for index in (0, 2):
                    expected = client.process_pixels(expected, pages[index]["params"])
                np.testing.assert_array_equal(output_rgb(processing.sample())[0], expected)
                self.assertEqual(len(client.commands), 2)
                self.assertEqual(client.commands[0]["params"], pages[1]["params"])
                self.assertEqual(client.commands[1]["passes"], [pages[0]["params"], pages[2]["params"]])
                receipt = validate_receipt({"extra_generation_params": processing.extra_generation_params}, spec, 1)
                self.assertEqual([record["index"] for record in receipt["passes"]], [2, 1, 3])

    def test_multipass_batch_snapshot_ignores_midrun_control_edits(self):
        from nr_shared.contract import ARG_KEYS, default_passes, make_pass_spec
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            processing, _, _, _, _, _ = host(client, directory)
            pages = default_passes()
            pages[1].update(enabled=True, stage="after_hr")
            pages[2]["enabled"] = True
            spec = make_pass_spec(True, pages, RUNTIME, hires=True)
            processing.script_args = [0, *script_args(spec)]
            processing.scripts.alwayson_scripts[0].args_to = len(processing.script_args)
            processing.pixels = tensor(np.concatenate([processing.pixels, 1 - processing.pixels]))
            processing.n_iter = 2
            client.mutate = lambda: processing.script_args.__setitem__(1 + ARG_KEYS.index("pass_2_enabled"), False)
            processing.sample()
            processing.iteration = 1
            processing.sample()
            receipt = validate_receipt({"extra_generation_params": processing.extra_generation_params}, spec, 4)
            self.assertEqual(len(client.commands), 8)
            self.assertEqual([record["count"] for record in receipt["passes"]], [4, 4, 4])
            self.assertEqual([record["index"] for record in receipt["passes"]], [1, 3, 2])

    def test_disabled_middle_page_is_skipped_and_all_disabled_is_passthrough(self):
        from nr_shared.contract import ARG_KEYS, default_passes, make_pass_spec
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            processing, _, _, _, _, _ = host(client, directory, hires=False)
            pages = default_passes()
            pages[2]["enabled"] = True
            spec = make_pass_spec(True, pages, RUNTIME, hires=False)
            processing.script_args = [0, *script_args(spec)]
            processing.scripts.alwayson_scripts[0].args_to = len(processing.script_args)
            processing.sample()
            self.assertEqual(len(client.commands[0]["passes"]), 2)
            receipt = validate_receipt({"extra_generation_params": processing.extra_generation_params}, spec, 1)
            self.assertEqual([record["index"] for record in receipt["passes"]], [1, 3])
            for index in (1, 3):
                processing.script_args[1 + ARG_KEYS.index(f"pass_{index}_enabled")] = False
            with patch.object(client, "runtime", side_effect=AssertionError("All disabled must not contact NR")):
                processing.sample()
            self.assertEqual(len(client.commands), 1)
            self.assertNotIn(SCRIPT_TITLE, processing.extra_generation_params)

    def test_missing_or_wrong_later_pass_evidence_never_issues_receipt(self):
        from nr_shared.contract import default_passes, make_pass_spec
        corruptions = [lambda result: result.update(completed_passes=1),
                       lambda result: result["pass_results"][1]["native"].update(device="cuda:0"),
                       lambda result: result["pass_results"][1]["params"].update(tone=0)]
        for corrupt in corruptions:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                client = PNGClient()
                client.corrupt = corrupt
                processing, _, _, _, _, _ = host(client, directory, hires=False)
                pages = default_passes()
                pages[1]["enabled"] = True
                spec = make_pass_spec(True, pages, RUNTIME, hires=False)
                processing.script_args = [0, *script_args(spec)]
                processing.scripts.alwayson_scripts[0].args_to = len(processing.script_args)
                with self.assertRaisesRegex(RuntimeError, "全部启用页"):
                    processing.sample()
                self.assertNotIn(SCRIPT_TITLE, processing.extra_generation_params)

    def test_anima_qwen_vae_single_frame_pixels_not_video_or_channel_axis(self):
        for hires in (False, True):
            with self.subTest(hires=hires), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                client = PNGClient()
                p, _, rgb, processing, _, _ = host(client, directory, hires=hires)
                if hires:
                    p.pixels = tensor(np.asarray(p.pixels)[:, None, ...])
                else:
                    processing.decode_latent_batch = lambda model, a, **kw: processing.DecodedSamples(
                        [tensor(np.asarray(image)[None, ...]) for image in a])
                expected = np.where(rgb >= 128, 224, 16).astype(np.uint8)
                if hires:
                    expected = np.minimum(np.asarray(Image.fromarray(expected).resize((5, 4),
                        Image.Resampling.BILINEAR)).astype(int) + 24, 255).astype(np.uint8)
                np.testing.assert_array_equal(output_rgb(p.sample())[0], expected)
                self.assertEqual(len(client.commands), 1)
        for shape in ((1, 2, 3, 4, 5), (1, 3, 1, 4, 5)):
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                client = PNGClient()
                _, spec, _, _, adapter, _ = host(client, directory)
                with adapter.cancellation() as cancel, self.assertRaises(ValueError):
                    adapter.enhance(tensor(np.zeros(shape)), spec, cancel)
                self.assertEqual(client.commands, [])

    def test_unknown_execution_keeps_only_this_requests_files(self):
        from nr_shared.contract import SharedError
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            failure = SharedError(503, "submission reply lost")
            failure.execution_unknown = True
            client.error = failure
            p, _, _, _, _, _ = host(client, directory, hires=False)
            with self.assertRaisesRegex(RuntimeError, "保留"):
                p.sample()
            requests = list(Path(directory).iterdir())
            self.assertEqual(len(requests), 1)
            self.assertTrue((requests[0] / "input.png").is_file())
            self.assertTrue((requests[0] / "output").is_dir())
            self.assertNotIn(SCRIPT_TITLE, p.extra_generation_params)

    def test_after_hr_and_no_hr_normalized_pixels(self):
        for hires, stage in [(True, "after_hr"), (False, "before_hr")]:
            with self.subTest(hires=hires), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                client = PNGClient()
                p, spec, rgb, _, _, _ = host(client, directory, hires=hires, stage=stage)
                if hires:
                    rgb = np.minimum(np.asarray(Image.fromarray(rgb).resize((5, 4), Image.Resampling.BILINEAR)).astype(int) + 24, 255)
                expected = np.where(rgb >= 128, 224, 16).astype(np.uint8)
                got = p.sample()
                self.assertTrue(got.already_decoded)
                np.testing.assert_array_equal(output_rgb(got)[0], expected)
                validate_receipt({"extra_generation_params": p.extra_generation_params}, spec, 1)

    def test_batch_snapshot_survives_edits_and_accumulates_all_iterations(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            p, spec, rgb, _, _, _ = host(client, directory, hires=False)
            p.n_iter = 2
            p.pixels = tensor(np.concatenate([p.pixels, 1 - p.pixels]))
            client.mutate = lambda: p.script_args.__setitem__(5, 0.25)
            first = p.sample()
            p.iteration = 1
            second = p.sample()
            self.assertEqual([x["params"] for x in client.commands], [spec["params"]] * 4)
            np.testing.assert_array_equal(output_rgb(first), output_rgb(second))
            self.assertFalse(np.array_equal(output_rgb(first)[0], output_rgb(first)[1]))
            validate_receipt({"extra_generation_params": p.extra_generation_params}, spec, 4)
            p.iteration = 0
            p.script_args[1] = False
            count = len(client.commands)
            p.sample()
            self.assertEqual(len(client.commands), count)
            self.assertNotIn(SCRIPT_TITLE, p.extra_generation_params)
            self.assertFalse(any(name.startswith("_forge_nr_") for name in vars(p)))

    def test_after_without_hr_rejected_before_sampler(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            p, _, _, _, _, _ = host(client, directory, hires=False)
            p.script_args[2] = "after_hr"
            with self.assertRaisesRegex(ValueError, "高清"):
                p.sample()
            self.assertEqual(p.native_calls, 0)
            self.assertEqual(client.commands, [])

    def test_latent_before_decodes_and_encodes_before_resize(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            p, _, rgb, _, _, _ = host(PNGClient(), directory, latent=True)
            expected = np.asarray(Image.fromarray(np.where(rgb >= 128, 224, 16).astype(np.uint8)).resize((5, 4), Image.Resampling.BILINEAR))
            expected = np.minimum(expected.astype(int) + 24, 255).astype(np.uint8)
            np.testing.assert_array_equal(output_rgb(p.sample())[0], expected)

    def test_cross_checkpoint_or_vae_latent_rejected_before_sampling(self):
        for name, value in [("hr_checkpoint_name", "other-model"), ("hr_additional_modules", []), ("refiner_checkpoint", "refiner")]:
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                client = PNGClient()
                p, _, _, _, _, _ = host(client, directory, latent=True)
                setattr(p, name, value)
                with self.assertRaisesRegex(ValueError, "latent"):
                    p.sample()
                self.assertEqual(p.native_calls, 0)
                self.assertEqual(client.commands, [])

    def test_failures_propagate_without_receipt_and_only_own_files_removed(self):
        cases = [lambda r: r.update(runtime_id="b" * 64), lambda r: r["native"].update(hardware_verified=False),
                 lambda r: r.update(width=999), lambda r: r.update(name="../input.png"),
                 lambda r: r["params"].update(tone=0.), lambda r: r["native"].update(device="cuda:0")]
        for corrupt in cases:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                keep = Path(directory) / "not-this-request.txt"
                keep.write_text("retained")
                client = PNGClient()
                client.corrupt = corrupt
                p, _, _, _, _, _ = host(client, directory, hires=False)
                with self.assertRaises(RuntimeError):
                    p.sample()
                self.assertNotIn(SCRIPT_TITLE, p.extra_generation_params)
                self.assertEqual(list(Path(directory).iterdir()), [keep])
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            error = RuntimeError("worker deliberately failed")
            client.error = error
            p, _, _, _, _, _ = host(client, directory)
            with self.assertRaises(RuntimeError) as caught:
                p.sample()
            self.assertIs(caught.exception, error)
            self.assertNotIn(SCRIPT_TITLE, p.extra_generation_params)

    def test_interrupt_cancels_own_request_and_no_img2img_recursion(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            p, _, _, processing, _, state = host(client, directory, hires=False)
            self.assertEqual(processing.StableDiffusionProcessingImg2Img().sample(), "ADetailer native img2img")
            client.mutate = lambda: setattr(state, "interrupted", True)
            with self.assertRaisesRegex(RuntimeError, "取消"):
                p.sample()
            self.assertEqual(len(client.commands), 1)
            self.assertNotIn(SCRIPT_TITLE, p.extra_generation_params)

    def test_idempotent_reload_preserves_other_plugins_wrapper_chain(self):
        from functools import wraps
        from forge_nr.hooks import install
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            p, _, _, processing, adapter, _ = host(client, directory, hires=False)
            cls = processing.StableDiffusionProcessingTxt2Img
            first_wrapper = cls.sample
            handle = install(processing, adapter)
            self.assertIs(cls.sample, first_wrapper)
            visited = []

            @wraps(cls.sample)
            def foreign(p, *args, **kwargs):
                visited.append("foreign")
                return first_wrapper(p, *args, **kwargs)

            cls.sample = foreign
            handle.close()
            self.assertIs(cls.sample, foreign)
            p.sample()
            self.assertEqual(client.commands, [])
            install(processing, adapter)
            self.assertIs(cls.sample, foreign)
            p.sample()
            self.assertEqual(len(client.commands), 1)
            self.assertEqual(visited, ["foreign", "foreign"])

    def test_opaque_foreign_wrappers_are_not_silently_double_wrapped(self):
        from forge_nr.hooks import install
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            _, _, _, processing, adapter, _ = host(PNGClient(), directory)
            cls = processing.StableDiffusionProcessingTxt2Img
            sample, hr = cls.sample, cls.sample_hr_pass
            cls.sample = lambda p, *a, **kw: sample(p, *a, **kw)
            cls.sample_hr_pass = lambda p, *a, **kw: hr(p, *a, **kw)
            before = (cls.sample, cls.sample_hr_pass)
            with self.assertRaisesRegex(RuntimeError, "opaque"):
                install(processing, adapter)
            self.assertEqual((cls.sample, cls.sample_hr_pass), before)

    def test_interrupt_poll_sets_only_this_clients_cancel_event(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            p, _, _, _, _, state = host(client, directory, hires=False)
            events = []

            def waiting(command, progress, cancel_event):
                state.interrupted = True
                self.assertTrue(cancel_event.wait(2), "Forge interrupt did not reach shared request")
                events.append(command["id"])
                raise RuntimeError("own request canceled")

            client.run = waiting
            with self.assertRaisesRegex(RuntimeError, "own request"):
                p.sample()
            self.assertEqual(len(events), 1)
            self.assertNotIn(SCRIPT_TITLE, p.extra_generation_params)

    def test_stale_runtime_is_rejected_before_sampling_or_worker_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            p, _, _, _, _, _ = host(client, directory)
            p.script_args[-1]["runtime_id"] = "b" * 64
            with self.assertRaisesRegex(RuntimeError, "过期"):
                p.sample()
            self.assertEqual(p.native_calls, 0)
            self.assertEqual(client.commands, [])

    def test_receipt_requires_execution_bound_identity_not_later_global_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            client.corrupt = lambda result: result.pop("worker_pid")
            p, _, _, _, _, _ = host(client, directory, hires=False)
            with self.assertRaisesRegex(RuntimeError, "PID"):
                p.sample()
            self.assertNotIn(SCRIPT_TITLE, p.extra_generation_params)

    def test_before_hr_processes_real_png_before_pixel_resize(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            p, spec, rgb, _, _, _ = host(client, directory)
            got = p.sample()
            nr = np.where(rgb >= 128, 224, 16).astype(np.uint8)
            resized = np.asarray(Image.fromarray(nr).resize((5, 4), Image.Resampling.BILINEAR))
            expected = np.minimum(resized.astype(int) + 24, 255).astype(np.uint8)
            np.testing.assert_array_equal(output_rgb(got)[0], expected)
            self.assertTrue(got.already_decoded)
            self.assertEqual(len(client.commands), 1)
            validate_receipt({"extra_generation_params": p.extra_generation_params}, spec, 1)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_disabled_preserves_native_object_and_never_touches_adapter(self):
        from forge_nr.hooks import install

        original = object()

        class Txt:
            def sample(self, *args, **kwargs):
                return original

            def sample_hr_pass(self, *args, **kwargs):
                return original

        class NoCalls:
            def __getattr__(self, name):
                raise AssertionError("Disabled NR touched adapter: " + name)

        p = Txt()
        p.enable_hr = False
        p.script_args = [0, *script_args(None)]
        p.scripts = SimpleNamespace(alwayson_scripts=[SimpleNamespace(
            title=lambda: SCRIPT_TITLE, args_from=1, args_to=13)])
        p.extra_generation_params = {SCRIPT_TITLE: "stale"}
        processing = SimpleNamespace(StableDiffusionProcessingTxt2Img=Txt)
        install(processing, NoCalls())
        self.assertIs(p.sample(), original)
        self.assertIs(p.sample_hr_pass(None, None), original)
        self.assertNotIn(SCRIPT_TITLE, p.extra_generation_params)


class XYZTests(unittest.TestCase):
    def test_three_page_preset_freezes_pages_and_legacy_load_clears_extra_pages(self):
        from nr_shared.contract import ARG_KEYS, default_passes, from_script_args, make_pass_spec
        from forge_nr.controls import Presets, preset_passes
        from forge_nr.xyz import PresetAxis
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            presets = Presets(Path(directory) / "presets")
            pages = default_passes()
            pages[0]["enabled"] = False
            pages[1].update(enabled=True, stage="after_hr")
            pages[1]["params"]["tone"] = .4
            pages[2]["enabled"] = True
            presets.save_passes("chain", pages)
            presets.save("legacy", "before_hr", {**DEFAULT_PARAMS, "mix": .3})
            axis = PresetAxis(presets)
            parent, _, _, _, _, _ = host(PNGClient(), directory)
            parent.script_args = [0, *script_args(make_pass_spec(True, pages, RUNTIME, hires=True))]
            parent.scripts.alwayson_scripts[0].args_to = len(parent.script_args)
            parent.script_args[1] = False
            original = copy.deepcopy(parent.script_args)
            axis.confirm(parent, ["chain", "legacy"])
            pages[1]["params"]["tone"] = 1.7
            presets.save_passes("chain", pages)
            cell = copy.copy(parent)
            axis.apply(cell, "chain", ["chain"])
            self.assertIs(cell.script_args[1], False)
            self.assertEqual(cell.script_args[1 + ARG_KEYS.index("pass_2_tone")], .4)
            self.assertEqual(cell.script_args[12], RUNTIME)
            self.assertEqual(parent.script_args, original)
            cell.script_args[1] = True
            spec = from_script_args(cell.script_args[1:], hires=True)
            self.assertFalse(spec["passes"][0]["enabled"])
            axis.apply(cell, "legacy", ["legacy"])
            loaded = from_script_args(cell.script_args[1:], hires=True)
            self.assertEqual(loaded["params"]["mix"], .3)
            self.assertNotIn("passes", loaded)
            self.assertFalse(cell.script_args[1 + ARG_KEYS.index("pass_2_enabled")])
            self.assertFalse(cell.script_args[1 + ARG_KEYS.index("pass_3_enabled")])
            copied = preset_passes(presets.load("chain"))
            copied[2]["params"]["mix"] = 0
            self.assertEqual(presets.load("chain")["passes"][2]["params"]["mix"], 1.)
            cell.enable_hr = False
            axis.apply(cell, "chain", ["chain"])
            self.assertTrue(all(record["stage"] == "before_hr"
                                for record in from_script_args(cell.script_args[1:], hires=False)["passes"]))

    def axes(self, directory):
        import ast
        from forge_nr.controls import Presets
        from forge_nr.xyz import register
        path = Path(os.environ.get("FORGE_ROOT", "_forge_not_installed")) / "scripts/xyz_grid.py"
        if not path.is_file():
            self.skipTest("installed Forge XYZ axis definitions not available")
        definitions = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                       if isinstance(node, ast.ClassDef) and node.name in ("AxisOption", "AxisOptionTxt2Img", "AxisOptionImg2Img")]
        namespace = {"format_value_add_label": lambda *args: ""}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
        other = namespace["AxisOption"]("Other", str, lambda *args: None)
        module = SimpleNamespace(AxisOptionTxt2Img=namespace["AxisOptionTxt2Img"], axis_options=[other])
        scripts = [SimpleNamespace(path=str(path), module=module)]
        presets = Presets(Path(directory) / "presets")
        registration = register(scripts, presets)
        self.addCleanup(registration.close)
        return module, scripts, presets, registration

    def test_toggle_cells_do_not_mutate_parent_siblings_or_other_script_parameters(self):
        from forge_nr.xyz import apply_enabled
        for initial in (False, True):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                client = PNGClient()
                parent, spec, _, _, _, _ = host(client, directory, hires=False)
                parent.script_args[1] = initial
                parent.script_args = tuple([*parent.script_args, {"other_script": True}])
                parent.extra_generation_params = {"Other extension": "retained"}
                original = copy.deepcopy(parent.script_args)
                cells = []
                for value, enabled in (("Off", False), ("On", True), ("Off", False)):
                    cell = copy.copy(parent)
                    apply_enabled(cell, value, ["Off", "On"])
                    self.assertEqual(cell.script_args[1], enabled)
                    self.assertEqual(cell.script_args[:1] + cell.script_args[2:], original[:1] + original[2:])
                    cell.n_iter = 2
                    cell.sample()
                    cell.iteration = 1
                    cell.sample()
                    cells.append(cell)
                    if enabled:
                        validate_receipt({"extra_generation_params": cell.extra_generation_params}, spec, 2)
                    else:
                        self.assertNotIn(SCRIPT_TITLE, cell.extra_generation_params)
                self.assertEqual(len(client.commands), 2)
                self.assertEqual(parent.script_args, original)
                self.assertEqual(parent.extra_generation_params, {"Other extension": "retained"})
                self.assertEqual([cell.script_args[1] for cell in cells], [False, True, False])

    def test_preset_snapshot_and_toggle_axes_commute_and_freeze_grid_values(self):
        from forge_nr.xyz import ENABLED_LABEL, PRESET_LABEL
        from nr_shared.contract import from_script_args
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            module, _, presets, _ = self.axes(directory)
            axes = {axis.label: axis for axis in module.axis_options}
            enabled, preset = axes[ENABLED_LABEL], axes[PRESET_LABEL]
            soft = {**DEFAULT_PARAMS, "mix": .25, "tone": .8}
            strong = {**DEFAULT_PARAMS, "intensity": 1.5, "mix": .8}
            presets.save("soft", "before_hr", soft)
            presets.save("strong", "after_hr", strong)
            self.assertEqual(preset.choices(), ["soft", "strong"])
            client = PNGClient()
            parent, _, _, _, _, _ = host(client, directory, hires=True)
            parent.script_args = tuple(parent.script_args)
            original = copy.deepcopy(parent.script_args)
            preset.confirm(parent, ["soft", "strong"])
            presets.save("soft", "after_hr", DEFAULT_PARAMS)
            presets.delete("strong")
            for name, params, stage in (("soft", soft, "before_hr"), ("strong", strong, "after_hr")):
                for toggle in ("Off", "On"):
                    outcomes = []
                    for reverse in (False, True):
                        cell = copy.copy(parent)
                        operations = [(enabled, toggle, ["Off", "On"]), (preset, name, ["soft", "strong"])]
                        for axis, value, values in (reversed(operations) if reverse else operations):
                            axis.apply(cell, value, values)
                        outcomes.append(copy.deepcopy(cell.script_args))
                        spec = from_script_args(cell.script_args[1:13], hires=True)
                        before = len(client.commands)
                        cell.sample()
                        if toggle == "On":
                            self.assertEqual(spec["params"], params)
                            self.assertEqual(spec["stage"], stage)
                            self.assertEqual(client.commands[-1]["params"], params)
                            self.assertEqual(len(client.commands), before + 1)
                        else:
                            self.assertIsNone(spec)
                            self.assertEqual(len(client.commands), before)
                        self.assertEqual(cell.script_args[-1], RUNTIME)
                    self.assertEqual(outcomes[0], outcomes[1])
            self.assertEqual(parent.script_args, original)
            self.assertEqual(parent.extra_generation_params, {})

    def test_preset_only_keeps_enable_and_no_hr_matches_preset_load(self):
        from forge_nr.xyz import PRESET_LABEL
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            module, _, presets, _ = self.axes(directory)
            presets.save("after", "after_hr", {**DEFAULT_PARAMS, "skin": .4})
            axis = next(axis for axis in module.axis_options if axis.label == PRESET_LABEL)
            parent, _, _, _, _, _ = host(PNGClient(), directory, hires=False)
            parent.script_args[1] = False
            axis.confirm(parent, ["after"])
            cell = copy.copy(parent)
            axis.apply(cell, "after", ["after"])
            self.assertIs(cell.script_args[1], False)
            self.assertEqual(cell.script_args[2], "before_hr")
            self.assertEqual(cell.script_args[-1], RUNTIME)
            with self.assertRaises(ValueError):
                axis.confirm(parent, ["missing"])
            with self.assertRaises(ValueError):
                axis.confirm(parent, [])

    def test_off_skips_invalid_environment_but_on_still_validates(self):
        from forge_nr.xyz import apply_enabled
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            client = PNGClient()
            parent, _, _, _, _, _ = host(client, directory, hires=False)
            parent.script_args[-1] = {}
            off, on = copy.copy(parent), copy.copy(parent)
            apply_enabled(off, "Off", [])
            off.sample()
            apply_enabled(on, "On", [])
            with self.assertRaises(ValueError):
                on.sample()
            self.assertEqual(client.commands, [])

    def test_registration_is_txt2img_only_idempotent_and_unload_is_scoped(self):
        from forge_nr.xyz import register, ENABLED_LABEL, PRESET_LABEL
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            module, scripts, presets, first = self.axes(directory)
            original = list(module.axis_options)
            second = register(scripts, presets)
            self.assertEqual(module.axis_options, original)
            self.assertEqual([axis.label for axis in module.axis_options[1:]], [ENABLED_LABEL, PRESET_LABEL])
            self.assertTrue(all(axis.is_img2img is False for axis in module.axis_options[1:]))
            self.assertEqual(module.axis_options[1].choices(), ["Off", "On"])
            module.axis_options[1].confirm(None, ["Off", "On", "True", "False"])
            with self.assertRaises(ValueError):
                module.axis_options[1].confirm(None, ["Off", "typo"])
            with self.assertRaises(ValueError):
                module.axis_options[1].confirm(None, [])
            second.close()
            first.close()
            self.assertEqual(module.axis_options, original[:1])

    def test_native_xyz_run_applies_both_axes_from_any_xyz_position(self):
        import ast
        import csv
        from collections import namedtuple
        from io import StringIO
        from itertools import chain, permutations, product
        from forge_nr.xyz import ENABLED_LABEL, PRESET_LABEL
        path = Path(os.environ.get("FORGE_ROOT", "_forge_not_installed")) / "scripts/xyz_grid.py"
        if not path.is_file():
            self.skipTest("installed Forge XYZ execution loop not available")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        script = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Script")
        run = next(node for node in script.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        helpers = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in ("csv_string_to_list_strip", "list_to_csv_string", "SharedSettingsStackHelper")]
        for enabled_position in range(3):
            with self.subTest(axis=enabled_position), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                module, _, presets, _ = self.axes(directory)
                presets.save("soft", "before_hr", {**DEFAULT_PARAMS, "mix": .25})
                presets.save("strong", "before_hr", {**DEFAULT_PARAMS, "mix": .8})
                client = PNGClient()
                parent, _, _, processing, _, _ = host(client, directory, hires=False)
                parent.prompt, parent.seed, parent.styles = "CPU XYZ fixture", 1234, []
                parent.width = parent.height = 16
                parent.steps = 8
                parent.batch_size = parent.n_iter = 1
                parent.script_args[1] = False
                original = copy.deepcopy(parent.script_args)
                seen = []
                state = SimpleNamespace(interrupted=False, stopping_generation=False)

                def process_images(cell):
                    cell.iteration = 0
                    cell.all_prompts, cell.all_seeds, cell.all_subseeds = [cell.prompt], [cell.seed], [0]
                    cell.sample()
                    seen.append((cell.script_args[1], cell.script_args[10], cell.seed,
                                 SCRIPT_TITLE in cell.extra_generation_params))
                    return SimpleNamespace(images=[object()])

                def draw(*args, **kwargs):
                    for (index_x, value_x), (index_y, value_y), (index_z, value_z) in product(
                            enumerate(kwargs["xs"]), enumerate(kwargs["ys"]), enumerate(kwargs["zs"])):
                        kwargs["cell"](value_x, value_y, value_z, index_x, index_y, index_z)
                    return SimpleNamespace(images=[])

                def fail(error, *args):
                    raise AssertionError("Native XYZ cell swallowed an error") from error

                processing.fix_seed = lambda _: None
                processing.create_infotext = lambda *args: "CPU fixture infotext"
                namespace = dict(csv=csv, StringIO=StringIO, chain=chain, permutations=permutations, Image=Image,
                    copy=copy.copy, processing=processing, modules=SimpleNamespace(processing=processing),
                    opts=SimpleNamespace(return_grid=True, img_max_size_mp=100), state=state,
                    shared=SimpleNamespace(state=state, total_tqdm=SimpleNamespace(updateTotal=lambda _: None)),
                    logger=SimpleNamespace(info=lambda _: None), AxisInfo=namedtuple("AxisInfo", "axis values"),
                    str_permutations=lambda value: value,
                    StableDiffusionProcessingTxt2Img=processing.StableDiffusionProcessingTxt2Img,
                    Processed=SimpleNamespace, process_images=process_images, draw_xyz_grid=draw,
                    errors=SimpleNamespace(display=fail), refresh_loading_params_for_xyz_grid=lambda: None)
                exec(compile(ast.Module(body=[*helpers, run], type_ignores=[]), str(path), "exec"), namespace)
                nothing = module.axis_options[0]
                nothing.label, nothing.choices, nothing.format_value = "Nothing", None, lambda *args: ""
                axes = {axis.label: index for index, axis in enumerate(module.axis_options)}
                chosen = [(axes[PRESET_LABEL], "", ["soft", "strong"]), (0, "", [])]
                chosen.insert(enabled_position, (axes[ENABLED_LABEL], "", ["Off", "On"]))
                axis_args = [value for fields in chosen for value in fields]
                plot = SimpleNamespace(current_axis_options=module.axis_options, title=lambda: "X/Y/Z plot")
                with patch.object(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS):
                    namespace["run"](plot, parent, *axis_args, True, True, False, False, False, False, False, 1, 0, False)
                self.assertCountEqual(seen, [(False, .25, 1234, False), (True, .25, 1234, True),
                                             (False, .8, 1234, False), (True, .8, 1234, True)])
                self.assertEqual(len(client.commands), 2)
                self.assertEqual(parent.script_args, original)
                self.assertEqual(parent.extra_generation_params, {})


class ControlTests(unittest.TestCase):
    def test_actual_ui_arguments_preset_callbacks_and_explicit_device_enumeration(self):
        from forge_nr.ui import build_ui
        from forge_nr.controls import Presets
        from nr_shared.contract import ARG_KEYS, from_script_args

        class Component:
            def __init__(self, kind, value=None, **kwargs):
                self.kind, self.value = kind, value
                self.elem_id = kwargs.get("elem_id")
                self.label = kwargs.get("label")
                self.choices = kwargs.get("choices", [])
                self.events = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def click(self, fn, inputs=None, outputs=None, **kwargs):
                self.events["click"] = (fn, inputs or [], outputs or [])

            def change(self, fn, inputs=None, outputs=None, **kwargs):
                self.events.setdefault("change", []).append((fn, inputs or [], outputs or []))

            def input(self, fn, inputs=None, outputs=None, **kwargs):
                self.events["input"] = (fn, inputs or [], outputs or [])

            def load(self, fn, inputs=None, outputs=None, **kwargs):
                self.events["load"] = (fn, inputs or [], outputs or [])

            def fire(self, event):
                events = self.events[event] if isinstance(self.events[event], list) else [self.events[event]]
                for fn, inputs, outputs in events:
                    result = fn(*(component.value for component in inputs))
                    if len(outputs) == 1:
                        result = [result]
                    for component, value in zip(outputs, result):
                        if isinstance(value, dict) and value.get("__type__") == "update":
                            for name, val in value.items():
                                if name != "__type__":
                                    setattr(component, name, val)
                        else:
                            component.value = value

        class Gradio:
            Error = RuntimeError

            def __init__(self):
                self.by_id = {}

            def __getattr__(self, kind):
                def create(value=None, **kwargs):
                    result = Component(kind, value, **kwargs)
                    if result.elem_id:
                        self.by_id[result.elem_id] = result
                    return result
                return create

            @staticmethod
            def update(**kwargs):
                return dict(__type__="update", **kwargs)

        class Service(PNGClient):
            def __init__(self):
                super().__init__()
                self.enumerations = 0
                self.failed = False

            def runtime(self, device=None):
                if self.failed:
                    raise RuntimeError("chosen GPU mapping missing")
                return super().runtime(device)

            def inspect(self, runtime):
                self.enumerations += 1
                return dict(devices=[dict(device="cuda:1", gpu_index=1, gpu_name="Test NVIDIA", luid="abc")])

            def stop(self):
                raise RuntimeError("another application is busy")

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            gr, service = Gradio(), Service()
            hr, block = Component("Checkbox", False), Component("Blocks")
            def input_accordion(value, *, label, elem_id):
                return Component("InputAccordion", value, elem_id=elem_id + "-checkbox")
            controls = build_ui(gr, service, Presets(Path(directory)), input_accordion=input_accordion, hr=hr, block=block)
            self.assertEqual(controls[0].kind, "InputAccordion")
            self.assertFalse(controls[0].value)
            self.assertEqual([x.elem_id for x in controls], ["forge_nr-checkbox", *["forge_nr_" + key for key in ARG_KEYS[1:]]])
            block.fire("load")
            self.assertEqual(service.enumerations, 0)
            self.assertEqual(json.loads(controls[11].value), RUNTIME)
            self.assertEqual(len(controls), 35)
            self.assertEqual([gr.by_id[f"forge_nr_pass_{index}_enabled"].value for index in range(1, 4)],
                             [True, False, False])
            self.assertTrue(all(f"forge_nr_tab_{index}" in gr.by_id for index in range(1, 4)))
            self.assertIn(RUNTIME["gpu_name"], gr.by_id["forge_nr_environment"].value)
            before_language = copy.deepcopy([control.value for control in controls])
            self.assertNotIn("forge_nr_language", gr.by_id)
            for locale, label in (("None", "Insertion point"), ("zh_CN", "插入时机"), ("zh-Hans", "插入时机")):
                translated_gr, translated_block = Gradio(), Component("Blocks")
                translated_hr = Component("Checkbox", False)
                translated = build_ui(translated_gr, service, Presets(Path(directory)), input_accordion=input_accordion,
                                      hr=translated_hr, block=translated_block, localization=locale)
                translated_block.fire("load")
                self.assertEqual(translated_gr.by_id["forge_nr_stage"].label, label)
                self.assertNotIn("forge_nr_language", translated_gr.by_id)
                self.assertEqual([control.value for control in translated], before_language)
            controls[0].value = True
            spec = from_script_args([x.value for x in controls], hires=False)
            self.assertEqual(spec["params"], DEFAULT_PARAMS)
            gr.by_id["forge_nr_list_devices"].fire("click")
            self.assertEqual(service.enumerations, 1)
            hr.value = True
            hr.fire("change")
            controls[1].value = "after_hr"
            gr.by_id["forge_nr_preset_name"].value = "saved"
            gr.by_id["forge_nr_save_preset"].fire("click")
            controls[4].value = 0.2
            hr.value = False
            hr.fire("change")
            self.assertEqual(controls[1].value, "before_hr")
            gr.by_id["forge_nr_load_preset"].fire("click")
            self.assertEqual(controls[4].value, DEFAULT_PARAMS["intensity"])
            self.assertEqual(controls[1].value, "before_hr")
            self.assertTrue(controls[0].value)
            self.assertEqual(json.loads(controls[11].value), RUNTIME)
            gr.by_id["forge_nr_mix"].value = .3
            gr.by_id["forge_nr_copy_pass_2"].fire("click")
            self.assertEqual(gr.by_id["forge_nr_pass_2_mix"].value, .3)
            self.assertFalse(gr.by_id["forge_nr_pass_2_enabled"].value)
            self.assertEqual(gr.by_id["forge_nr_pass_3_mix"].value, 1.)
            gr.by_id["forge_nr_pass_2_enabled"].value = True
            gr.by_id["forge_nr_pass_3_enabled"].value = True
            gr.by_id["forge_nr_pass_2_tone"].value = .7
            hr.value = True
            hr.fire("change")
            gr.by_id["forge_nr_pass_3_stage"].value = "after_hr"
            gr.by_id["forge_nr_preset_name"].value = "three-pages"
            gr.by_id["forge_nr_save_preset"].fire("click")
            gr.by_id["forge_nr_pass_2_tone"].value = 1.8
            gr.by_id["forge_nr_pass_3_enabled"].value = False
            gr.by_id["forge_nr_load_preset"].fire("click")
            spec = from_script_args([control.value for control in controls], hires=True)
            self.assertEqual(spec["passes"][1]["params"]["tone"], .7)
            self.assertTrue(spec["passes"][2]["enabled"])
            self.assertEqual(spec["passes"][2]["stage"], "after_hr")
            hr.value = False
            hr.fire("change")
            self.assertTrue(all(gr.by_id["forge_nr_" + prefix + "stage"].value == "before_hr"
                                for prefix in ("", "pass_2_", "pass_3_")))
            service.failed = True
            gr.by_id["forge_nr_device"].fire("input")
            self.assertEqual(gr.by_id["forge_nr_device"].value, "cuda:1")
            self.assertIn("mapping missing", gr.by_id["forge_nr_environment"].value)
            with self.assertRaises(ValueError):
                from_script_args([x.value for x in controls], hires=False)
            gr.by_id["forge_nr_release"].fire("click")
            self.assertIn("another application", gr.by_id["forge_nr_backend_status"].value)

            import hashlib
            from nr_runtime import settings
            from nr_runtime.setup import RuntimeSetup
            extension = Path(directory) / "retry-extension"
            helper_bytes = b"CPU HELPER FIXTURE"
            for relative in (".venv/Scripts/python.exe", "native/nr/bin/dlss5nr_bridge.dll",
                             "models/dlssnr/caller/nvngx.dll_comfy.dll"):
                target = extension / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(helper_bytes)
            helper = "models/dlssnr/caller/nvngx.dll_comfy.dll"
            (extension / "native-artifacts.json").write_text(json.dumps({"files": {helper: hashlib.sha256(helper_bytes).hexdigest()}}))
            (extension / "runtime-download.json").write_text(json.dumps({"url": None, "member": None}))
            (extension / "install.py").write_bytes((ROOT / "install.py").read_bytes())
            retry_gr = Gradio()
            retry_setup = RuntimeSetup(extension, models_dir=Path(directory) / "forge-models")
            build_ui(retry_gr, service, Presets(Path(directory)), input_accordion=input_accordion,
                     localization="zh_CN", setup=retry_setup)
            with patch.object(settings, "ROOT", extension), patch.object(settings, "CONFIG_PATH", extension / "runtime-settings.json"), \
                    patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["pip", "install"])) as packages:
                retry_gr.by_id["forge_nr_prepare_runtime"].fire("click")
                self.assertEqual(packages.call_count, 0)
                self.assertIn("下载源", retry_gr.by_id["forge_nr_setup_message"].value)
                self.assertEqual(retry_gr.by_id["forge_nr_runtime"].value, "{}")

            from nr_runtime.download import DownloadError
            from unittest.mock import Mock
            setup = SimpleNamespace(ensure=Mock(), import_runtime=Mock(), mode="auto", runtime_dir=Path(directory) / "DLSS-NR")
            def select_mode(mode):
                setup.mode = mode
                setup.runtime_dir = Path(directory) / "DLSS-NR" / "manual" if mode == "manual" else Path(directory) / "DLSS-NR"
            setup.set_mode = Mock(side_effect=select_mode)
            service.failed = False
            service.start = Mock()
            service.repair = Mock(side_effect=lambda action: action())
            prepared_gr, prepared_block = Gradio(), Component("Blocks")
            prepared = build_ui(prepared_gr, service, Presets(Path(directory)), input_accordion=input_accordion,
                                block=prepared_block, localization="zh_CN", setup=setup)
            setup.ensure.side_effect = DownloadError("source_missing", "No publisher source")
            prepared_block.fire("load")
            self.assertEqual(prepared[11].value, "{}")
            self.assertIn("下载源", prepared_gr.by_id["forge_nr_setup_message"].value)
            self.assertNotIn("forge_nr_permission", prepared_gr.by_id)
            self.assertEqual(prepared_gr.by_id["forge_nr_runtime_file"].kind, "File")
            setup.ensure.side_effect = None
            prepared_gr.by_id["forge_nr_prepare_runtime"].fire("click")
            self.assertFalse(setup.ensure.call_args.kwargs["repair"])
            self.assertEqual(json.loads(prepared[11].value), RUNTIME)
            self.assertIn("检测完成", prepared_gr.by_id["forge_nr_environment"].value)
            service.repair.assert_not_called()
            setup.ensure.side_effect = DownloadError("dependencies_failed", "Raw pip command must stay in details")
            prepared_gr.by_id["forge_nr_repair_dependencies"].fire("click")
            self.assertTrue(setup.ensure.call_args.kwargs["repair"])
            service.repair.assert_called_once()
            self.assertIn("依赖修复未完成", prepared_gr.by_id["forge_nr_setup_message"].value)
            self.assertNotIn("pip", prepared_gr.by_id["forge_nr_setup_message"].value)
            self.assertIn("Raw pip command", prepared_gr.by_id["forge_nr_setup_details"].value)
            setup.ensure.side_effect = None
            prepared_gr.by_id["forge_nr_runtime_file"].value = "selected-runtime.dll"
            service.failed = True
            service.start.side_effect = lambda: setattr(service, "failed", False)
            prepared_gr.by_id["forge_nr_import_runtime"].fire("click")
            setup.import_runtime.assert_called_once_with("selected-runtime.dll")
            self.assertEqual(json.loads(prepared[11].value), RUNTIME)
            self.assertIn("准备好", prepared_gr.by_id["forge_nr_environment"].value)
            self.assertEqual(prepared_gr.by_id["forge_nr_setup_details"].value, "")
            from forge_nr.lifecycle import RepairBlocked
            service.repair.side_effect = RepairBlocked("Active NR request")
            prepared_gr.by_id["forge_nr_runtime_mode"].value = "manual"
            prepared_gr.by_id["forge_nr_runtime_mode"].fire("input")
            self.assertEqual(prepared_gr.by_id["forge_nr_runtime_mode"].value, "auto")
            setup.set_mode.assert_not_called()
            self.assertIn("未开始切换", prepared_gr.by_id["forge_nr_setup_message"].value)
            service.repair.side_effect = lambda action: action()
            prepared_gr.by_id["forge_nr_runtime_mode"].value = "manual"
            prepared_gr.by_id["forge_nr_runtime_mode"].fire("input")
            setup.set_mode.assert_called_once_with("manual")
            self.assertEqual(prepared_gr.by_id["forge_nr_runtime_location"].value, str(setup.runtime_dir))
            self.assertIn("不会下载", prepared_gr.by_id["forge_nr_runtime_note"].value)
            self.assertIn("未验证发布者签名", prepared_gr.by_id["forge_nr_environment"].value)
            self.assertEqual(len(prepared), 35)
            prepared_gr.by_id["forge_nr_runtime_mode"].value = "auto"
            prepared_gr.by_id["forge_nr_runtime_location"].value = "stale initial render"
            prepared_block.fire("load")
            self.assertEqual(prepared_gr.by_id["forge_nr_runtime_mode"].value, "manual")
            self.assertEqual(prepared_gr.by_id["forge_nr_runtime_location"].value, str(setup.runtime_dir))
            self.assertIn("不会下载", prepared_gr.by_id["forge_nr_runtime_note"].value)
            self.assertEqual(len(prepared), 35)

    def test_entry_reads_forge_localization_without_an_independent_setting(self):
        import ast
        from types import SimpleNamespace
        from unittest.mock import Mock
        source = ROOT / "scripts/forge_nr_script.py"
        script = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                      if isinstance(node, ast.ClassDef) and node.name == "Script")
        method = next(node for node in script.body if isinstance(node, ast.FunctionDef) and node.name == "ui")
        builder = Mock(return_value=[])
        shared = SimpleNamespace(opts=SimpleNamespace(localization="zh_CN"))
        namespace = dict(build_ui=builder, shared=shared, gr=SimpleNamespace(context=SimpleNamespace(Context=SimpleNamespace(root_block=None))),
                         service=None, presets=None, InputAccordion=None, setup=None)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        for locale in ("zh_CN", "None"):
            shared.opts.localization = locale
            namespace["ui"](SimpleNamespace(hr=None), False)
            self.assertEqual(builder.call_args.kwargs["localization"], locale)

    def test_runtime_root_is_always_the_complete_extension(self):
        from forge_nr.discovery import discover_root
        self.assertEqual(discover_root(ROOT), ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            (base / "settings.json").write_text("external settings must not be read", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Install the complete extension"):
                discover_root(base)
            (base / "nr_shared").mkdir()
            (base / "nr_shared/contract.py").write_text("", encoding="utf-8")
            self.assertEqual(discover_root(base), base)

    @unittest.skipUnless(NATIVE_GRADIO, "optional real Forge Gradio CPU validation")
    def test_real_gradio_build_and_numeric_script_arguments(self):
        import ast
        import inspect
        import warnings
        from functools import wraps
        from forge_nr.controls import Presets
        from forge_nr.ui import build_ui
        from nr_shared.contract import from_script_args
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            os.environ["GRADIO_TEMP_DIR"] = directory
            os.environ["MPLCONFIGDIR"] = directory
            os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
            # No event loop is needed to build components. Prevent Windows'
            # auto-created asyncio loop from opening its socketpair listener.
            import asyncio
            asyncio.set_event_loop(None)
            import gradio as gr
            source = Path(os.environ.get("FORGE_ROOT", "_forge_not_installed")) / "modules/ui_components.py"
            definitions = [item for item in ast.parse(source.read_text(encoding="utf-8")).body
                           if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in ("InputAccordionImpl", "InputAccordion")]
            namespace = {"gr": gr, "wraps": wraps, "inspect": inspect, "warnings": warnings,
                         "GradioDeprecationWarning": DeprecationWarning, "__name__": __name__}
            compatibility = source.with_name("gradio_extensions.py")
            events = [item for item in ast.parse(compatibility.read_text(encoding="utf-8")).body
                      if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in ("EventWrapper", "repair")]
            exec(compile(ast.Module(body=events, type_ignores=[]), str(compatibility), "exec"), namespace)
            self.enterContext(patch.object(gr.Checkbox, "__init__", gr.Checkbox.__init__))
            self.enterContext(patch.object(gr.Checkbox, "update", gr.update, create=True))
            namespace["repair"](gr.Checkbox)
            with patch("gradio.component_meta.create_or_modify_pyi", return_value=None):
                exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
            with gr.Blocks(analytics_enabled=False) as block:
                hr = gr.Checkbox(False)
                controls = build_ui(gr, PNGClient(), Presets(Path(directory)), input_accordion=namespace["InputAccordion"], hr=hr, block=block)
            self.assertEqual(controls[0].accordion_id, "forge_nr")
            self.assertEqual(controls[0].elem_id, "forge_nr-checkbox")
            self.assertFalse(controls[0].accordion.open)
            args = [component.preprocess(component.value) for component in controls]
            args[0] = True
            args[11] = json.dumps(RUNTIME)
            self.assertEqual(from_script_args(args, hires=False)["params"], DEFAULT_PARAMS)
            config = block.get_config_file()
            self.assertEqual(sum(item["props"].get("elem_id") == "forge_nr-checkbox" for item in config["components"]), 1)
            self.assertTrue(all(len(dependency["outputs"]) == len(set(dependency["outputs"])) for dependency in config["dependencies"]))
            self.assertGreater(len(config["dependencies"]), 5)
            print("REAL_GRADIO", gr.__version__, "controls", len(controls), "dependencies", len(config["dependencies"]))

    def test_local_presets_save_parameters_and_stage_not_environment_or_enable(self):
        from forge_nr.controls import Presets
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            presets = Presets(Path(directory))
            presets.save("细节", "after_hr", DEFAULT_PARAMS)
            self.assertEqual(presets.names(), ["细节"])
            record = presets.load("细节")
            self.assertEqual(set(record), {"stage", "params"})
            self.assertEqual(record["stage"], "after_hr")
            self.assertEqual(record["params"], DEFAULT_PARAMS)
            with self.assertRaises(ValueError):
                presets.save("escape", "before_hr", dict(DEFAULT_PARAMS, enabled=True, runtime=RUNTIME))
            presets.delete("细节")
            self.assertEqual(presets.names(), [])

    def test_app_started_capabilities_and_client_lifecycle_are_scoped(self):
        from forge_nr.lifecycle import Service, app_started
        from nr_shared.contract import ARG_KEYS, PROTOCOL
        events = []

        class Client(PNGClient):
            def __init__(self, *, root, label):
                super().__init__()
                events.append((root, label))

            def start(self):
                events.append("start")

            def close(self):
                events.append("close-only-mine")

            def stop(self):
                raise RuntimeError("busy with another client")

        class App:
            def __init__(self):
                self.routes = []

            def add_api_route(self, path, endpoint, methods):
                self.routes.append(SimpleNamespace(path=path, endpoint=endpoint, methods=methods))

        service = Service(ROOT, client_factory=Client)
        self.assertEqual(events, [])
        app = App()
        app_started(app, service, ready_hook=lambda: True)
        app_started(app, service, ready_hook=lambda: True)
        self.assertEqual(events, [(ROOT, "forge"), "start"])
        self.assertEqual(len(app.routes), 1)
        cap = app.routes[0].endpoint()
        self.assertEqual((cap["protocol"], cap["script"], cap["arg_keys"], cap["ready_hook"]),
                         (PROTOCOL, SCRIPT_TITLE, list(ARG_KEYS), True))
        with self.assertRaisesRegex(RuntimeError, "another client"):
            service.stop()
        service.close()
        service.close()
        self.assertEqual(events[-1], "close-only-mine")
        self.assertEqual(events.count("close-only-mine"), 1)
        with self.assertRaises(RuntimeError):
            service.status()


if __name__ == "__main__":
    # Fail closed even if a future test accidentally constructs a live client.
    def forbidden(*args, **kwargs):
        raise AssertionError("Test forbids network/subprocess/CUDA")

    class ImportGuard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "core" or fullname.startswith("core.") or (fullname == "torch" and not NATIVE_TORCH):
                raise AssertionError("Forbidden in Forge plugin test: " + fullname)

    sys.meta_path.insert(0, ImportGuard())
    with ExitStack() as guards:
        for obj, name in [(socket.socket, "connect"), (socket.socket, "connect_ex"), (socket.socket, "bind"),
                          (socket, "create_connection"), (subprocess, "Popen"), (os, "system")]:
            guards.enter_context(patch.object(obj, name, forbidden))
        if NATIVE_TORCH:
            sys.modules["triton"] = None
            import torch as REAL_TORCH
            TORCH = REAL_TORCH
            for obj, name in [(REAL_TORCH.cuda, "init"), (REAL_TORCH.cuda, "_lazy_init"), (REAL_TORCH._C, "_cuda_init")]:
                if hasattr(obj, name):
                    guards.enter_context(patch.object(obj, name, forbidden))
            print("REAL_TORCH", REAL_TORCH.__version__, "CUDA initialized:", REAL_TORCH.cuda.is_initialized())
        else:
            print("NUMPY_TENSOR_CPU: torch/core imports, HTTP, socket bind and subprocess forbidden")
        program = unittest.main(verbosity=2, exit=False)
        if REAL_TORCH is not None:
            assert not REAL_TORCH.cuda.is_initialized()
            print("FINAL_CUDA_INITIALIZED", REAL_TORCH.cuda.is_initialized())
        raise SystemExit(not program.result.wasSuccessful())
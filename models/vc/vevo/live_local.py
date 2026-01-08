# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from models.vc.vevo.live_engine import (
    AudioRingBuffer,
    VevoStreamingEngine,
    normalize_length,
    smooth_boundary_inplace,
)
from models.vc.vevo.runner import VevoConverter


def _ratio(src_sr: int, dst_sr: int) -> tuple[int, int]:
    frac = Fraction(dst_sr, src_sr).limit_denominator(1000)
    return frac.numerator, frac.denominator


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    up, down = _ratio(src_sr, dst_sr)
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def _normalize_len_end(x: np.ndarray, n: int) -> np.ndarray:
    x = x.reshape(-1).astype(np.float32, copy=False)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[-n:]
    return np.pad(x, (n - len(x), 0), mode="constant")


class OutputRingBuffer:
    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._write_pos = 0
        self._read_pos = 0
        self._size = 0
        self._lock = threading.Lock()

        self.underflows = 0
        self.overflows = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return

        with self._lock:
            if n > self._capacity:
                x = x[-self._capacity :]
                n = len(x)

            free = self._capacity - self._size
            if n > free:
                drop = n - free
                self._read_pos = (self._read_pos + drop) % self._capacity
                self._size -= drop
                self.overflows += 1

            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos : end] = x
            else:
                first = self._capacity - self._write_pos
                self._buf[self._write_pos :] = x[:first]
                self._buf[: end % self._capacity] = x[first:]

            self._write_pos = end % self._capacity
            self._size += n

    def read(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            if self._size < n:
                self.underflows += 1
                out = np.zeros(n, dtype=np.float32)
                if self._size > 0:
                    out[-self._size :] = self._read_no_lock(self._size)
                    self._size = 0
                return out
            return self._read_no_lock(n)

    def _read_no_lock(self, n: int) -> np.ndarray:
        end = self._read_pos + n
        if end <= self._capacity:
            out = self._buf[self._read_pos : end].copy()
        else:
            first = self._capacity - self._read_pos
            out = np.concatenate([self._buf[self._read_pos :], self._buf[: end % self._capacity]]).copy()
        self._read_pos = end % self._capacity
        self._size -= n
        return out

    def size(self) -> int:
        with self._lock:
            return self._size


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Vevo live VC (single-process, local inference).")
    parser.add_argument("--ref", type=str, required=True, help="Reference wav path.")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Optional EvalConfig JSON file (as produced by evaluation.vevo_live.search).",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto; mps on mac)")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")

    parser.add_argument("--io_sample_rate", type=int, default=48000, help="Audio device sample rate.")
    parser.add_argument("--window_ms", type=int, default=1000)
    parser.add_argument("--hop_ms", type=int, default=1000)
    parser.add_argument("--fade_ms", type=int, default=20)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--flow_matching_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--diffusion_cfg", type=float, default=1.0)
    parser.add_argument("--diffusion_rescale_cfg", type=float, default=0.75)

    # AR knobs (vevovoice only)
    parser.add_argument("--ar_max_length", type=int, default=2000)
    parser.add_argument("--ar_temperature", type=float, default=0.8)
    parser.add_argument("--ar_top_k", type=int, default=50)
    parser.add_argument("--ar_top_p", type=float, default=0.9)
    parser.add_argument("--ar_repeat_penalty", type=float, default=1.0)
    parser.add_argument("--ar_min_new_tokens", type=int, default=50)
    parser.add_argument(
        "--no_prepend_style_ref_to_input",
        action="store_true",
        help="(vevovoice) Disable prepending the style reference to each input window.",
    )

    parser.add_argument("--input_device", type=str, default=None)
    parser.add_argument("--output_device", type=str, default=None)
    parser.add_argument("--block_ms", type=int, default=20)
    parser.add_argument("--list_devices", action="store_true")
    args = parser.parse_args(argv)

    if args.config_json:
        raw = json.loads(Path(args.config_json).read_text())
        inf = raw.get("inference", {})
        stream = raw.get("streaming", {})

        args.kind = str(inf.get("kind", args.kind))
        args.window_ms = int(stream.get("window_ms", args.window_ms))
        args.hop_ms = int(stream.get("hop_ms", args.hop_ms))
        args.fade_ms = int(stream.get("fade_ms", args.fade_ms))
        args.normalize_align = str(stream.get("normalize_align", args.normalize_align))

        args.flow_matching_steps = int(inf.get("flow_matching_steps", args.flow_matching_steps))
        args.seed = int(inf.get("seed", args.seed))
        args.diffusion_cfg = float(inf.get("diffusion_cfg", args.diffusion_cfg))
        args.diffusion_rescale_cfg = float(inf.get("diffusion_rescale_cfg", args.diffusion_rescale_cfg))

        args.ar_max_length = int(inf.get("ar_max_length", args.ar_max_length))
        args.ar_temperature = float(inf.get("ar_temperature", args.ar_temperature))
        args.ar_top_k = int(inf.get("ar_top_k", args.ar_top_k))
        args.ar_top_p = float(inf.get("ar_top_p", args.ar_top_p))
        args.ar_repeat_penalty = float(inf.get("ar_repeat_penalty", args.ar_repeat_penalty))
        args.ar_min_new_tokens = int(inf.get("ar_min_new_tokens", args.ar_min_new_tokens))
        if "prepend_style_ref_to_input" in inf:
            args.no_prepend_style_ref_to_input = not bool(inf["prepend_style_ref_to_input"])

    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: sounddevice. Install with `pip install sounddevice`.") from e

    if args.list_devices:
        print(sd.query_devices())
        return 0

    io_sr = int(args.io_sample_rate)
    model_sr = int(VevoStreamingEngine.model_sr)

    window_samples_model = int(round(args.window_ms / 1000 * model_sr))
    hop_samples_model = int(round(args.hop_ms / 1000 * model_sr))
    fade_samples_model = int(round(args.fade_ms / 1000 * model_sr))
    if hop_samples_model <= 0 or window_samples_model <= 0:
        raise ValueError("window_ms and hop_ms must be > 0")
    if hop_samples_model > window_samples_model:
        raise ValueError("hop_ms must be <= window_ms")

    hop_seconds = hop_samples_model / model_sr
    hop_samples_io = int(round(hop_seconds * io_sr))
    block_samples_io = int(round(args.block_ms / 1000 * io_sr))
    if block_samples_io <= 0:
        block_samples_io = 256

    input_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
    out_buf = OutputRingBuffer(capacity=int(io_sr * 10))

    stop = threading.Event()

    def in_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            pass
        x = indata[:, 0].copy().astype(np.float32, copy=False)
        try:
            input_q.put_nowait(x)
        except queue.Full:
            try:
                _ = input_q.get_nowait()
            except queue.Empty:
                pass
            input_q.put_nowait(x)

    def out_callback(outdata, frames, time_info, status):  # noqa: ANN001
        if status:
            pass
        y = out_buf.read(frames)
        outdata[:] = y.reshape(-1, 1)

    def worker() -> None:
        converter = VevoConverter.from_pretrained(
            kind=args.kind,  # type: ignore[arg-type]
            device=args.device,
            repo_cache_dir=args.repo_cache_dir,
        )
        engine = VevoStreamingEngine(converter)
        engine.prepare_reference_bytes(Path(args.ref).read_bytes())

        ring = AudioRingBuffer(window_samples_model)
        prev_last: Optional[float] = None
        buf_io = np.zeros(0, dtype=np.float32)

        window_count = 0
        timings: list[float] = []

        while not stop.is_set():
            try:
                x_block = input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buf_io = np.concatenate([buf_io, x_block])
            while len(buf_io) >= hop_samples_io and not stop.is_set():
                hop_io = buf_io[:hop_samples_io]
                buf_io = buf_io[hop_samples_io:]

                hop_model = _resample(hop_io, io_sr, model_sr)
                hop_model = _normalize_len_end(hop_model, hop_samples_model)
                ring.write(hop_model)

                if ring.size < window_samples_model:
                    prev_last = None
                    out_io = np.zeros(hop_samples_io, dtype=np.float32)
                    out_buf.write(out_io)
                    continue

                window = ring.read_last(window_samples_model)
                t0 = time.time()
                out_window = engine.convert_window(
                    window,
                    flow_matching_steps=args.flow_matching_steps,
                    diffusion_cfg=args.diffusion_cfg,
                    diffusion_rescale_cfg=args.diffusion_rescale_cfg,
                    seed=args.seed + window_count,
                    ar_max_length=args.ar_max_length,
                    ar_temperature=args.ar_temperature,
                    ar_top_k=args.ar_top_k,
                    ar_top_p=args.ar_top_p,
                    ar_repeat_penalty=args.ar_repeat_penalty,
                    ar_min_new_tokens=args.ar_min_new_tokens,
                    prepend_style_ref_to_input=not bool(args.no_prepend_style_ref_to_input),
                )
                timings.append(time.time() - t0)

                out_window = normalize_length(out_window, window_samples_model, align=args.normalize_align)
                out_hop = out_window[-hop_samples_model:].astype(np.float32, copy=False)
                out_hop = smooth_boundary_inplace(out_hop, prev_last, fade_samples_model)
                prev_last = float(out_hop[-1]) if len(out_hop) else prev_last

                out_io = _resample(out_hop, model_sr, io_sr)
                out_buf.write(out_io)
                window_count += 1

                if window_count % max(1, int(2.5 / max(hop_seconds, 1e-6))) == 0 and timings:
                    mean_sec = float(np.mean(np.asarray(timings, dtype=np.float64)))
                    rtf = mean_sec / max(hop_seconds, 1e-9)
                    print(
                        f"[live_local] device={converter.device} windows={window_count} "
                        f"mean_win_s={mean_sec:.3f} rtf={rtf:.2f} out_buf_s={out_buf.size()/io_sr:.2f} "
                        f"underflows={out_buf.underflows} overflows={out_buf.overflows} in_q={input_q.qsize()}",
                        flush=True,
                    )
                    timings = timings[-50:]

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    stream_in = sd.InputStream(
        samplerate=io_sr,
        channels=1,
        dtype="float32",
        blocksize=block_samples_io,
        device=args.input_device,
        callback=in_callback,
    )
    stream_out = sd.OutputStream(
        samplerate=io_sr,
        channels=1,
        dtype="float32",
        blocksize=block_samples_io,
        device=args.output_device,
        callback=out_callback,
    )

    print(
        f"[live_local] io_sr={io_sr} model_sr={model_sr} window={args.window_ms}ms hop={args.hop_ms}ms "
        f"steps={args.flow_matching_steps} block={args.block_ms}ms kind={args.kind}",
        flush=True,
    )

    with stream_in, stream_out:
        try:
            while thread.is_alive():
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            thread.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


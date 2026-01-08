# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import io
import json
import queue
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
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


def _load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _encode_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32, copy=False), sr, format="WAV")
    return buf.getvalue()


def _read_ref_bytes_trimmed(ref_path: str, *, max_sec: float) -> bytes:
    wav, sr = _load_wav_mono(ref_path)
    if max_sec > 0:
        max_samples = int(round(max_sec * sr))
        if len(wav) > max_samples:
            wav = wav[:max_samples]
            print(f"[live_local] Trimmed ref to {max_sec:.2f}s: {ref_path}", flush=True)
    return _encode_wav_bytes(wav, sr)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Vevo live VC (single-process, local inference).")
    parser.add_argument("--ref", type=str, default=None, help="Reference wav path (required unless --passthrough).")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Optional EvalConfig JSON file (as produced by evaluation.vevo_live.search).",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto; mps on mac)")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")
    parser.add_argument(
        "--ref_max_sec",
        type=float,
        default=10.0,
        help="Trim reference audio to at most this many seconds (0 to disable).",
    )

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
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help="Debug: bypass Vevo and play back the (buffered) mic audio directly.",
    )
    parser.add_argument(
        "--src_wav",
        type=str,
        default=None,
        help="Debug: simulate mic input from a wav file instead of a live device.",
    )
    parser.add_argument(
        "--out_wav",
        type=str,
        default="runs/vevo_live/live_local_sim.wav",
        help="(with --src_wav) output wav path.",
    )
    parser.add_argument(
        "--sim_realtime",
        action="store_true",
        help="(with --src_wav) sleep to simulate real-time device timing.",
    )
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

    converter = None
    engine: Optional[VevoStreamingEngine] = None
    if not args.passthrough:
        if not args.ref:
            raise ValueError("--ref is required unless --passthrough")
        converter = VevoConverter.from_pretrained(
            kind=args.kind,  # type: ignore[arg-type]
            device=args.device,
            repo_cache_dir=args.repo_cache_dir,
        )
        engine = VevoStreamingEngine(converter)
        ref_bytes = _read_ref_bytes_trimmed(str(args.ref), max_sec=float(args.ref_max_sec))
        engine.prepare_reference_bytes(ref_bytes)

    def worker() -> None:
        ring = AudioRingBuffer(window_samples_model)
        prev_last: Optional[float] = None
        buf_io = np.zeros(0, dtype=np.float32)

        window_count = 0
        timings: list[float] = []
        in_rms: list[float] = []

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
                    prev_last = 0.0
                    out_io = np.zeros(hop_samples_io, dtype=np.float32)
                    out_buf.write(out_io)
                    continue

                window = ring.read_last(window_samples_model)
                in_rms.append(float(np.sqrt(np.mean(window * window) + 1e-9)))
                if len(in_rms) > 200:
                    in_rms = in_rms[-200:]

                if args.passthrough:
                    out_window = window
                else:
                    assert engine is not None
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
                    if len(timings) > 200:
                        timings = timings[-200:]

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
                    in_rms_mean = float(np.mean(np.asarray(in_rms, dtype=np.float64))) if in_rms else 0.0
                    print(
                        f"[live_local] device={(converter.device if converter is not None else 'passthrough')} windows={window_count} "
                        f"mean_win_s={mean_sec:.3f} rtf={rtf:.2f} in_rms={in_rms_mean:.4f} out_buf_s={out_buf.size()/io_sr:.2f} "
                        f"underflows={out_buf.underflows} overflows={out_buf.overflows} in_q={input_q.qsize()}",
                        flush=True,
                    )
                    timings = timings[-50:]
                    in_rms = in_rms[-50:]

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    if args.src_wav:
        src, src_sr = _load_wav_mono(args.src_wav)
        if src_sr != io_sr:
            src = _resample(src, src_sr, io_sr)
        block = block_samples_io
        if block <= 0:
            raise ValueError("block_ms too small")

        pad = (-len(src)) % block
        if pad:
            src = np.pad(src, (0, pad), mode="constant")

        outputs: list[np.ndarray] = []
        block_sec = block / io_sr
        start_t = time.time()
        for i in range(0, len(src), block):
            blk = src[i : i + block]
            try:
                input_q.put_nowait(blk)
            except queue.Full:
                try:
                    _ = input_q.get_nowait()
                except queue.Empty:
                    pass
                input_q.put_nowait(blk)

            outputs.append(out_buf.read(block))

            if args.sim_realtime:
                next_t = start_t + (i // block + 1) * block_sec
                time.sleep(max(0.0, next_t - time.time()))

        stop.set()
        thread.join(timeout=10.0)
        out = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)
        Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.out_wav, out, io_sr)
        print(
            f"[live_local] Wrote sim output: {args.out_wav} (underflows={out_buf.underflows} overflows={out_buf.overflows})",
            flush=True,
        )
        return 0

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
        f"steps={args.flow_matching_steps} block={args.block_ms}ms kind={args.kind} passthrough={bool(args.passthrough)}",
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

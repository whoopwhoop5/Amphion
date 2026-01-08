# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import queue
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample_poly


def _ratio(src_sr: int, dst_sr: int) -> tuple[int, int]:
    frac = Fraction(dst_sr, src_sr).limit_denominator(1000)
    return frac.numerator, frac.denominator


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    up, down = _ratio(src_sr, dst_sr)
    y = resample_poly(x, up, down).astype(np.float32, copy=False)
    return y


def _normalize_len_end(x: np.ndarray, n: int) -> np.ndarray:
    x = x.reshape(-1).astype(np.float32, copy=False)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[-n:]
    return np.pad(x, (n - len(x), 0), mode="constant")


def _apply_peak_limiter(x: np.ndarray, peak_limit: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if float(peak_limit) <= 0:
        return x
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if not np.isfinite(peak) or peak <= float(peak_limit) or peak <= 1e-9:
        return x
    return (x * (float(peak_limit) / peak)).astype(np.float32, copy=False)


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
                # Drop oldest to make room (keeps playback moving).
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
                avail = self._size
                if avail > 0:
                    data = self._read_no_lock(avail)
                    out[-avail:] = data
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
    parser = argparse.ArgumentParser(description="Vevo live VC client (local mic/playback).")
    parser.add_argument("--server", type=str, required=True, help="WebSocket URL, e.g. ws://localhost:8080")
    parser.add_argument("--ref", type=str, required=True, help="Reference wav path.")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Optional EvalConfig JSON file (as produced by evaluation.vevo_live.search).",
    )

    parser.add_argument("--io_sample_rate", type=int, default=48000, help="Audio device sample rate.")
    parser.add_argument("--model_sample_rate", type=int, default=24000, help="Server/model sample rate.")
    parser.add_argument("--window_ms", type=int, default=2000)
    parser.add_argument("--hop_ms", type=int, default=1000)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--peak_limit", type=float, default=0.99)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--flow_matching_steps", type=int, default=8)
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
        args.vad_db = float(stream.get("vad_db", args.vad_db))
        args.vad_frame_ms = float(stream.get("vad_frame_ms", args.vad_frame_ms))
        args.peak_limit = float(stream.get("peak_limit", args.peak_limit))
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

    try:
        import websockets
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: websockets. Install with `pip install websockets`.") from e

    if args.list_devices:
        print(sd.query_devices())
        return 0

    io_sr = args.io_sample_rate
    model_sr = args.model_sample_rate

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
    out_buf = OutputRingBuffer(capacity=int(io_sr * 10))  # 10s buffer

    stop = threading.Event()
    stats = {"net_rtt_ms": 0.0, "sent": 0, "recv": 0}

    def in_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            pass
        x = indata[:, 0].copy().astype(np.float32, copy=False)
        try:
            input_q.put_nowait(x)
        except queue.Full:
            # Drop input if client can't keep up.
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

    async def ws_loop() -> None:
        ref_bytes = Path(args.ref).read_bytes()
        init = {
            "type": "init",
            "kind": args.kind,
            "sample_rate": model_sr,
            "window_samples": window_samples_model,
            "hop_samples": hop_samples_model,
            "fade_samples": fade_samples_model,
            "vad_db": args.vad_db,
            "vad_frame_ms": args.vad_frame_ms,
            "peak_limit": args.peak_limit,
            "flow_matching_steps": args.flow_matching_steps,
            "diffusion_cfg": args.diffusion_cfg,
            "diffusion_rescale_cfg": args.diffusion_rescale_cfg,
            "seed": args.seed,
            "ar_max_length": args.ar_max_length,
            "ar_temperature": args.ar_temperature,
            "ar_top_k": args.ar_top_k,
            "ar_top_p": args.ar_top_p,
            "ar_repeat_penalty": args.ar_repeat_penalty,
            "ar_min_new_tokens": args.ar_min_new_tokens,
            "prepend_style_ref_to_input": not bool(args.no_prepend_style_ref_to_input),
            "normalize_align": args.normalize_align,
            "reference_wav_b64": base64.b64encode(ref_bytes).decode("utf-8"),
        }

        async with websockets.connect(args.server, max_size=32 * 1024 * 1024) as ws:
            await ws.send(json.dumps(init))
            ready = json.loads(await ws.recv())
            if ready.get("type") != "ready":
                raise RuntimeError(f"Unexpected server response: {ready}")
            print(f"[live_client] Connected: {ready}", flush=True)

            buf_io = np.zeros(0, dtype=np.float32)
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

                    t_send = time.time()
                    await ws.send(hop_model.astype(np.float32, copy=False).tobytes())
                    out_bytes = await ws.recv()
                    stats["net_rtt_ms"] = (time.time() - t_send) * 1000.0

                    out_model = np.frombuffer(out_bytes, dtype=np.float32)
                    out_model = _normalize_len_end(out_model, hop_samples_model)
                    out_io = _resample(out_model, model_sr, io_sr)
                    out_io = _apply_peak_limiter(out_io, float(args.peak_limit))

                    out_buf.write(out_io)
                    stats["sent"] += 1
                    stats["recv"] += 1

            await ws.send(json.dumps({"type": "close"}))

    def net_thread() -> None:
        asyncio.run(ws_loop())

    net = threading.Thread(target=net_thread, daemon=True)
    net.start()

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
        f"[live_client] io_sr={io_sr} model_sr={model_sr} window={args.window_ms}ms hop={args.hop_ms}ms "
        f"steps={args.flow_matching_steps} block={args.block_ms}ms",
        flush=True,
    )

    with stream_in, stream_out:
        try:
            last = time.time()
            while net.is_alive():
                time.sleep(0.25)
                now = time.time()
                if now - last >= 1.0:
                    last = now
                    print(
                        f"[live_client] out_buf_s={out_buf.size()/io_sr:.2f} "
                        f"underflows={out_buf.underflows} overflows={out_buf.overflows} "
                        f"in_q={input_q.qsize()} rtt_ms={stats['net_rtt_ms']:.1f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            net.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

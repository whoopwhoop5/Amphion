# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import queue
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from evaluation.vc_quest.streaming_utils import (
    apply_peak_limiter,
    build_rms_mask,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
    rms_db,
    smooth_boundary_inplace,
)


def _ratio(src_sr: int, dst_sr: int) -> tuple[int, int]:
    frac = Fraction(dst_sr, src_sr).limit_denominator(1000)
    return frac.numerator, frac.denominator


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    up, down = _ratio(src_sr, dst_sr)
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def _normalize_len_end(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[-n:]
    return np.pad(x, (n - len(x), 0), mode="constant")


def _load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _trim_wav(wav: np.ndarray, sr: int, max_sec: float) -> np.ndarray:
    if max_sec <= 0:
        return np.asarray(wav, dtype=np.float32).reshape(-1)
    max_samples = int(round(float(max_sec) * float(sr)))
    if max_samples <= 0:
        return np.asarray(wav, dtype=np.float32).reshape(-1)
    return np.asarray(wav[:max_samples], dtype=np.float32).reshape(-1)


class AudioRingBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._write_pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return

        if n >= self._capacity:
            x = x[-self._capacity :]
            n = len(x)

        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos : end] = x
        else:
            first = self._capacity - self._write_pos
            self._buf[self._write_pos :] = x[:first]
            self._buf[: end % self._capacity] = x[first:]

        self._write_pos = end % self._capacity
        self._size = min(self._capacity, self._size + n)

    def read_last(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if n > self._size:
            raise ValueError(f"Requested n={n}, but size={self._size}")

        start = (self._write_pos - n) % self._capacity
        if start < self._write_pos:
            return self._buf[start : self._write_pos].copy()
        return np.concatenate([self._buf[start:], self._buf[: self._write_pos]]).copy()


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


def _parse_device_arg(v: Optional[str]) -> Optional[object]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    return s


def _build_audio_efx(
    *,
    mmcxli_dir: str,
    model_device: str,
    absolute_pitch: bool,
    estimate_energy: bool,
    pitch_shift: float,
    content_expand_rate: float,
):
    from evaluation.vc_quest.classicvc_convert import (  # local import to keep --list_devices fast
        _DummySoundControl,
        _import_mmcxli_audio_efx,
        _resolve_weight,
    )

    mmcxli_dir = str(Path(mmcxli_dir).expanduser().resolve())
    AudioEfx, load_make_vc_config = _import_mmcxli_audio_efx(mmcxli_dir)

    vc_cfg = load_make_vc_config("/tmp/mmcxli_vc_config.json", save=False)
    vc_cfg["model"]["model_device"] = str(model_device)
    vc_cfg["auto_encode"] = False
    vc_cfg["spec_rt_o"] = 2
    vc_cfg["absolute_pitch"] = bool(absolute_pitch)
    vc_cfg["estimate_energy"] = bool(estimate_energy)
    vc_cfg["pitch_shift"] = float(pitch_shift)
    vc_cfg["content_expand_rate"] = float(content_expand_rate)

    wdir = Path(mmcxli_dir) / "weights"
    vc_cfg["model"]["harmof0_ckpt"] = _resolve_weight(wdir / "harmof0.onnx", "harmof0.onnx")
    vc_cfg["model"]["CE_ckpt"] = _resolve_weight(wdir / "hubert500.onnx", "hubert500.onnx")
    vc_cfg["model"]["SE_ckpt"] = _resolve_weight(wdir / "style_encoder_304.onnx", "style_encoder_304.onnx")
    vc_cfg["model"]["f0n_ckpt"] = _resolve_weight(wdir / "f0n_predictor_hubert500.onnx", "f0n_predictor_hubert500.onnx")
    vc_cfg["model"]["decoder_ckpt"] = _resolve_weight(wdir / "decoder_24k.onnx", "decoder_24k.onnx")
    vc_cfg["model"]["style_compressor_ckpt"] = _resolve_weight(wdir / "pumap_encoder_2dim.onnx", "pumap_encoder_2dim.onnx")
    vc_cfg["model"]["style_decoder_ckpt"] = _resolve_weight(wdir / "pumap_decoder_2dim.onnx", "pumap_decoder_2dim.onnx")

    in_sr = 16000
    out_sr = int(vc_cfg["backend"]["sr_decode"])

    sc = _DummySoundControl(
        sr_out=out_sr,
        sr_proc=in_sr,
        block_roll_size=int(vc_cfg["backend"]["block_roll_size"]),
        content_expand_rate=float(vc_cfg["content_expand_rate"]),
    )
    audio_efx = AudioEfx(sc=sc, vc_config=vc_cfg, hop_size=160, dim_spec=352, ch_map=[0], bypass=False)
    return audio_efx, sc, in_sr, out_sr


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ClassicVC/MMCXLI live VC (single-process, local inference).")
    parser.add_argument("--mmcxli_dir", type=str, default="~/deps/mmcxli")
    parser.add_argument("--ref", type=str, default=None, help="Reference wav path (required unless --passthrough).")
    parser.add_argument("--ref_max_sec", type=float, default=10.0)
    parser.add_argument("--model_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--absolute_pitch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--estimate_energy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pitch_shift", type=float, default=0.0)
    parser.add_argument("--content_expand_rate", type=float, default=0.1)

    parser.add_argument("--io_sample_rate", type=int, default=48000, help="Audio device sample rate.")
    parser.add_argument("--window_ms", type=int, default=800)
    parser.add_argument("--hop_ms", type=int, default=400)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--emit_align", type=str, default="end", choices=["start", "center", "end"])

    parser.add_argument("--vad_mode", type=str, default="rms", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)

    parser.add_argument("--gain_mode", type=str, default="off", choices=["off", "match_src_rms"])
    parser.add_argument("--gain_target_delta_db", type=float, default=10.0)
    parser.add_argument("--gain_max_boost_db", type=float, default=18.0)
    parser.add_argument("--gain_smoothing", type=float, default=0.0)

    parser.add_argument("--mask_mode", type=str, default="off", choices=["off", "rms"])
    parser.add_argument("--mask_db", type=float, default=-50.0)
    parser.add_argument("--mask_frame_ms", type=float, default=10.0)
    parser.add_argument("--mask_smooth_ms", type=float, default=10.0)

    parser.add_argument("--peak_limit", type=float, default=0.99)
    parser.add_argument("--input_device", type=str, default=None)
    parser.add_argument("--output_device", type=str, default=None)
    parser.add_argument("--block_ms", type=int, default=20)
    parser.add_argument("--list_devices", action="store_true")
    parser.add_argument("--passthrough", action="store_true", help="Debug: bypass ClassicVC and play back mic audio.")

    parser.add_argument("--src_wav", type=str, default=None, help="Debug: simulate mic input from a wav file.")
    parser.add_argument("--out_wav", type=str, default="runs/vc_quest/classicvc_live_local_sim.wav")
    parser.add_argument("--sim_realtime", action="store_true", help="(with --src_wav) sleep to simulate device timing.")
    args = parser.parse_args(argv)

    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: sounddevice. Install with `pip install sounddevice`.") from e

    if args.list_devices:
        print(sd.query_devices())
        return 0

    io_sr = int(args.io_sample_rate)
    block_samples_io = int(round(float(args.block_ms) / 1000.0 * float(io_sr)))
    if block_samples_io <= 0:
        block_samples_io = 256

    input_device = _parse_device_arg(args.input_device)
    output_device = _parse_device_arg(args.output_device)

    if not args.passthrough:
        if not args.ref:
            raise ValueError("--ref is required unless --passthrough")

        audio_efx, sc, in_sr, out_sr = _build_audio_efx(
            mmcxli_dir=str(args.mmcxli_dir),
            model_device=str(args.model_device),
            absolute_pitch=bool(args.absolute_pitch),
            estimate_energy=bool(args.estimate_energy),
            pitch_shift=float(args.pitch_shift),
            content_expand_rate=float(args.content_expand_rate),
        )

        from evaluation.vc_quest.classicvc_convert import _compute_style_embedding  # local import

        ref_wav, ref_sr = _load_wav_mono(str(args.ref))
        ref_wav = _trim_wav(ref_wav, ref_sr, float(args.ref_max_sec))
        ref_16k = _resample(ref_wav, ref_sr, in_sr)
        style = _compute_style_embedding(audio_efx, ref_16k)
        sc.current_target_style = np.asarray(style, dtype=np.float32).reshape(1, -1)
    else:
        audio_efx = None
        sc = None
        in_sr = 16000
        out_sr = 16000

    window_in = int(round(float(args.window_ms) / 1000.0 * float(in_sr)))
    hop_in = int(round(float(args.hop_ms) / 1000.0 * float(in_sr)))
    window_in = max(1, window_in)
    hop_in = max(1, hop_in)
    if hop_in > window_in:
        raise ValueError("hop_ms must be <= window_ms")

    window_out = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
    hop_out = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
    fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))
    window_out = max(1, window_out)
    hop_out = max(1, hop_out)

    if args.emit_align == "start":
        emit_start_out = 0
        emit_start_in = 0
    elif args.emit_align == "center":
        emit_start_out = max(0, (window_out - hop_out) // 2)
        emit_start_in = max(0, (window_in - hop_in) // 2)
    elif args.emit_align == "end":
        emit_start_out = max(0, window_out - hop_out)
        emit_start_in = max(0, window_in - hop_in)
    else:
        raise ValueError(f"Unknown emit_align: {args.emit_align}")

    hop_seconds = hop_in / float(in_sr)
    hop_samples_io = int(round(hop_seconds * float(io_sr)))
    hop_samples_io = max(1, hop_samples_io)

    if args.src_wav:
        src, src_sr = _load_wav_mono(args.src_wav)
        if src_sr != io_sr:
            src = _resample(src, src_sr, io_sr)

        src_len = len(src)
        block = int(block_samples_io)
        pad = (-src_len) % block
        if pad:
            src = np.pad(src, (0, pad), mode="constant")

        ring = AudioRingBuffer(window_in)
        prev_last: Optional[float] = None
        buf_io = np.zeros(0, dtype=np.float32)
        outputs: list[np.ndarray] = []
        timings: list[float] = []

        hop_ms_eff = float(args.hop_ms)
        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(hop_ms_eff, 1e-6)))
        hangover_left = 0
        gain_db_state = 0.0

        start_t = time.time()
        for i in range(0, len(src), block):
            buf_io = np.concatenate([buf_io, src[i : i + block]])
            while len(buf_io) >= hop_samples_io:
                hop_io = buf_io[:hop_samples_io]
                buf_io = buf_io[hop_samples_io:]

                hop_16k = _resample(hop_io, io_sr, in_sr)
                hop_16k = _normalize_len_end(hop_16k, hop_in)
                ring.write(hop_16k)

                if ring.size < window_in:
                    prev_last = 0.0
                    outputs.append(np.zeros(hop_samples_io, dtype=np.float32))
                    continue

                window = ring.read_last(window_in)
                vad_segment = window[emit_start_in : emit_start_in + hop_in]
                silent_rms = bool(
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        vad_segment,
                        sample_rate=in_sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )

                vad_mode = str(args.vad_mode)
                if vad_mode == "off":
                    voiced = True
                elif vad_mode == "rms":
                    voiced = not silent_rms
                elif vad_mode == "webrtc":
                    webrtc_voiced = is_voiced_webrtcvad(
                        vad_segment,
                        sample_rate=in_sr,
                        frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                        aggressiveness=int(args.vad_webrtc_aggressiveness),
                        min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                    )
                    voiced = bool(webrtc_voiced) and (not silent_rms)
                else:
                    raise ValueError(f"Unknown vad_mode: {vad_mode}")

                if not voiced and hangover_left > 0 and (not silent_rms):
                    voiced = True
                    hangover_left -= 1
                elif voiced:
                    hangover_left = hangover_hops

                if args.passthrough or audio_efx is None:
                    out_window = window
                else:
                    if not voiced:
                        out_window = np.zeros(window_out, dtype=np.float32)
                    else:
                        t0 = time.time()
                        out_window = audio_efx.convert_offline(window[None, :])
                        timings.append(time.time() - t0)
                        out_window = np.asarray(out_window, dtype=np.float32).reshape(-1)
                        out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))

                out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))
                out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

                if voiced and str(args.gain_mode) == "match_src_rms":
                    alpha = float(np.clip(float(args.gain_smoothing), 0.0, 0.999))
                    src_db = rms_db(vad_segment, eps=1e-9)
                    out_db = rms_db(out_hop, eps=1e-9)
                    desired_boost_db = (src_db - out_db) - float(args.gain_target_delta_db)
                    desired_boost_db = float(np.clip(desired_boost_db, 0.0, float(args.gain_max_boost_db)))
                    gain_db_state = alpha * gain_db_state + (1.0 - alpha) * desired_boost_db
                    gain = float(10.0 ** (gain_db_state / 20.0))
                    out_hop = (out_hop * gain).astype(np.float32, copy=False)
                elif not voiced:
                    gain_db_state *= float(np.clip(float(args.gain_smoothing), 0.0, 0.999))

                if str(args.mask_mode) == "rms":
                    mask = build_rms_mask(
                        vad_segment,
                        in_sample_rate=in_sr,
                        out_sample_rate=out_sr,
                        out_len=hop_out,
                        frame_ms=float(args.mask_frame_ms),
                        threshold_db=float(args.mask_db),
                        smooth_ms=float(args.mask_smooth_ms),
                    )
                    out_hop = (out_hop * mask).astype(np.float32, copy=False)

                out_hop = smooth_boundary_inplace(out_hop, prev_last, fade_out)
                out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
                prev_last = float(out_hop[-1]) if len(out_hop) else prev_last

                out_io = _resample(out_hop, out_sr, io_sr)
                out_io = _normalize_len_end(out_io, hop_samples_io)
                out_io = apply_peak_limiter(out_io, peak_limit=float(args.peak_limit))
                outputs.append(out_io)

            if args.sim_realtime:
                block_sec = block / float(io_sr)
                next_t = start_t + (i // block + 1) * block_sec
                time.sleep(max(0.0, next_t - time.time()))

        out = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)
        out = out[:src_len]
        out_path = Path(args.out_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out, io_sr)

        if timings:
            mean_sec = float(np.mean(np.asarray(timings, dtype=np.float64)))
            rtf = mean_sec / max(hop_seconds, 1e-9)
            print(f"[classicvc_live_local] file_sim mean_win_s={mean_sec:.3f} rtf={rtf:.2f}", flush=True)
        print(f"[classicvc_live_local] Wrote sim output: {out_path}", flush=True)
        return 0

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
        ring = AudioRingBuffer(window_in)
        prev_last: Optional[float] = None
        buf_io = np.zeros(0, dtype=np.float32)

        timings: list[float] = []
        in_rms: list[float] = []
        gain_db_state = 0.0
        hop_ms_eff = float(args.hop_ms)
        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(hop_ms_eff, 1e-6)))
        hangover_left = 0
        window_count = 0

        while not stop.is_set():
            try:
                x_block = input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buf_io = np.concatenate([buf_io, x_block])
            while len(buf_io) >= hop_samples_io and not stop.is_set():
                hop_io = buf_io[:hop_samples_io]
                buf_io = buf_io[hop_samples_io:]

                hop_16k = _resample(hop_io, io_sr, in_sr)
                hop_16k = _normalize_len_end(hop_16k, hop_in)
                ring.write(hop_16k)

                if ring.size < window_in:
                    prev_last = 0.0
                    out_buf.write(np.zeros(hop_samples_io, dtype=np.float32))
                    continue

                window = ring.read_last(window_in)
                vad_segment = window[emit_start_in : emit_start_in + hop_in]
                in_rms.append(float(np.sqrt(np.mean(vad_segment * vad_segment) + 1e-9)))
                if len(in_rms) > 200:
                    in_rms = in_rms[-200:]

                silent_rms = bool(
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        vad_segment,
                        sample_rate=in_sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )

                vad_mode = str(args.vad_mode)
                if vad_mode == "off":
                    voiced = True
                elif vad_mode == "rms":
                    voiced = not silent_rms
                elif vad_mode == "webrtc":
                    webrtc_voiced = is_voiced_webrtcvad(
                        vad_segment,
                        sample_rate=in_sr,
                        frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                        aggressiveness=int(args.vad_webrtc_aggressiveness),
                        min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                    )
                    voiced = bool(webrtc_voiced) and (not silent_rms)
                else:
                    raise ValueError(f"Unknown vad_mode: {vad_mode}")

                if not voiced and hangover_left > 0 and (not silent_rms):
                    voiced = True
                    hangover_left -= 1
                elif voiced:
                    hangover_left = hangover_hops

                if args.passthrough or audio_efx is None:
                    out_window = window
                    out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))
                else:
                    if not voiced:
                        out_window = np.zeros(window_out, dtype=np.float32)
                    else:
                        t0 = time.time()
                        out_window = audio_efx.convert_offline(window[None, :])
                        timings.append(time.time() - t0)
                        if len(timings) > 200:
                            timings = timings[-200:]
                        out_window = np.asarray(out_window, dtype=np.float32).reshape(-1)
                        out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))

                out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

                if voiced and str(args.gain_mode) == "match_src_rms":
                    alpha = float(np.clip(float(args.gain_smoothing), 0.0, 0.999))
                    src_db = rms_db(vad_segment, eps=1e-9)
                    out_db = rms_db(out_hop, eps=1e-9)
                    desired_boost_db = (src_db - out_db) - float(args.gain_target_delta_db)
                    desired_boost_db = float(np.clip(desired_boost_db, 0.0, float(args.gain_max_boost_db)))
                    gain_db_state = alpha * gain_db_state + (1.0 - alpha) * desired_boost_db
                    gain = float(10.0 ** (gain_db_state / 20.0))
                    out_hop = (out_hop * gain).astype(np.float32, copy=False)
                elif not voiced:
                    gain_db_state *= float(np.clip(float(args.gain_smoothing), 0.0, 0.999))

                if str(args.mask_mode) == "rms":
                    mask = build_rms_mask(
                        vad_segment,
                        in_sample_rate=in_sr,
                        out_sample_rate=out_sr,
                        out_len=hop_out,
                        frame_ms=float(args.mask_frame_ms),
                        threshold_db=float(args.mask_db),
                        smooth_ms=float(args.mask_smooth_ms),
                    )
                    out_hop = (out_hop * mask).astype(np.float32, copy=False)

                out_hop = smooth_boundary_inplace(out_hop, prev_last, fade_out)
                out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
                prev_last = float(out_hop[-1]) if len(out_hop) else prev_last

                out_io = _resample(out_hop, out_sr, io_sr)
                out_io = _normalize_len_end(out_io, hop_samples_io)
                out_io = apply_peak_limiter(out_io, peak_limit=float(args.peak_limit))
                out_buf.write(out_io)
                window_count += 1

                if window_count % max(1, int(2.5 / max(hop_seconds, 1e-6))) == 0 and timings:
                    mean_sec = float(np.mean(np.asarray(timings, dtype=np.float64)))
                    rtf = mean_sec / max(hop_seconds, 1e-9)
                    in_rms_mean = float(np.mean(np.asarray(in_rms, dtype=np.float64))) if in_rms else 0.0
                    print(
                        f"[classicvc_live_local] windows={window_count} mean_win_s={mean_sec:.3f} rtf={rtf:.2f} "
                        f"in_rms={in_rms_mean:.4f} out_buf_s={out_buf.size()/io_sr:.2f} underflows={out_buf.underflows} "
                        f"overflows={out_buf.overflows} in_q={input_q.qsize()}",
                        flush=True,
                    )
                    timings = timings[-50:]
                    in_rms = in_rms[-50:]

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    stream_in = sd.InputStream(
        samplerate=io_sr,
        channels=1,
        dtype="float32",
        blocksize=block_samples_io,
        device=input_device,
        callback=in_callback,
    )
    stream_out = sd.OutputStream(
        samplerate=io_sr,
        channels=1,
        dtype="float32",
        blocksize=block_samples_io,
        device=output_device,
        callback=out_callback,
    )

    print(
        f"[classicvc_live_local] io_sr={io_sr} in_sr={in_sr} out_sr={out_sr} window={args.window_ms}ms hop={args.hop_ms}ms "
        f"emit_align={args.emit_align} model_device={args.model_device} passthrough={bool(args.passthrough)}",
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


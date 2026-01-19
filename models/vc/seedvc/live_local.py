# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from evaluation.vc_quest.streaming_utils import apply_peak_limiter, is_silent_rms_db, is_voiced_webrtcvad
from models.vc.live_io import (
    OutputRingBuffer,
    normalize_len_end,
    parse_device_arg,
    print_device_help,
    resample_audio,
)


def _torch_sync(device) -> None:  # noqa: ANN001
    try:
        import torch
    except Exception:
        return
    if getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize()
    elif getattr(device, "type", "") == "mps":
        torch.mps.synchronize()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Seed-VC live VC (single-process, local inference).")
    parser.add_argument("--seedvc_dir", type=str, default="~/deps/Seed-VC", help="Path to Seed-VC repo checkout.")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, mps, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--checkpoint_path", type=str, default="", help="Optional checkpoint (else downloads xlsr-tiny).")
    parser.add_argument("--config_path", type=str, default="", help="Optional config (required if checkpoint_path set).")
    parser.add_argument("--hf_repo", type=str, default="Plachta/Seed-VC", help="HF repo for default checkpoint/config.")
    parser.add_argument(
        "--hf_checkpoint_name",
        type=str,
        default="DiT_uvit_tat_xlsr_ema.pth",
        help="HF checkpoint filename to download when checkpoint_path is empty.",
    )
    parser.add_argument(
        "--hf_config_name",
        type=str,
        default="config_dit_mel_seed_uvit_xlsr_tiny.yml",
        help="HF config filename to download when checkpoint_path is empty.",
    )

    parser.add_argument("--ref", type=str, default=None, help="Reference audio (required unless --passthrough).")

    parser.add_argument("--io_sample_rate", type=int, default=48000, help="Audio device sample rate.")
    parser.add_argument("--window_ms", type=int, default=300, help="Seed-VC block_time (must equal hop_ms).")
    parser.add_argument("--hop_ms", type=int, default=300, help="Seed-VC hop (must equal window_ms).")
    parser.add_argument("--crossfade_ms", type=int, default=40)
    parser.add_argument("--extra_time_ce_ms", type=int, default=2500)
    parser.add_argument("--extra_time_ms", type=int, default=500)
    parser.add_argument("--extra_time_right_ms", type=int, default=20)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--inference_cfg_rate", type=float, default=0.7)
    parser.add_argument("--max_prompt_length_sec", type=float, default=3.0)

    parser.add_argument("--vad_mode", type=str, default="rms", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)
    parser.add_argument("--peak_limit", type=float, default=0.99)

    parser.add_argument("--input_device", type=str, default=None)
    parser.add_argument("--output_device", type=str, default=None)
    parser.add_argument("--block_ms", type=int, default=20, help="Audio callback block size (ms).")
    parser.add_argument("--list_devices", action="store_true")
    parser.add_argument("--passthrough", action="store_true", help="Debug: bypass Seed-VC and play back mic audio.")
    args = parser.parse_args(argv)

    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: sounddevice. Install with `pip install sounddevice`.") from e

    if args.list_devices:
        print_device_help(sd)
        return 0

    if int(args.hop_ms) != int(args.window_ms):
        raise ValueError("Seed-VC streaming uses hop_ms == window_ms (block_time).")

    io_sr = int(args.io_sample_rate)
    block_samples_io = int(round(float(args.block_ms) / 1000.0 * float(io_sr)))
    if block_samples_io <= 0:
        block_samples_io = 256

    input_device = parse_device_arg(args.input_device)
    output_device = parse_device_arg(args.output_device)

    if args.passthrough:
        device = None
        engine = None
        sr = 16000
        block = int(round(float(args.window_ms) / 1000.0 * float(io_sr)))
        hop_seconds = float(args.window_ms) / 1000.0
        hop_samples_io = int(round(hop_seconds * float(io_sr)))
    else:
        if not args.ref:
            raise ValueError("--ref is required unless --passthrough")

        import torch

        device = (
            torch.device(args.device)
            if args.device
            else torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        )

        from evaluation.vc_quest.seedvc_convert import (  # local import to keep --list_devices fast
            SeedVCStreamingEngine,
            _load_audio_mono,
            _load_seedvc_models,
        )

        seedvc_dir = os.path.abspath(str(Path(args.seedvc_dir).expanduser()))
        if not os.path.isdir(seedvc_dir):
            raise FileNotFoundError(f"seedvc_dir not found: {seedvc_dir}")

        model_set = _load_seedvc_models(
            seedvc_dir=seedvc_dir,
            device=device,
            checkpoint_path=str(args.checkpoint_path),
            config_path=str(args.config_path),
            fp16=bool(args.fp16),
            hf_repo=str(args.hf_repo),
            hf_checkpoint_name=str(args.hf_checkpoint_name),
            hf_config_name=str(args.hf_config_name),
        )

        sr = int(model_set.sr)
        ref_wav, _ = _load_audio_mono(str(args.ref), sr=sr)

        engine = SeedVCStreamingEngine(
            model_set=model_set,
            device=device,
            reference_wav=ref_wav,
            max_prompt_length_sec=float(args.max_prompt_length_sec),
            fp16=bool(args.fp16),
            block_ms=int(args.window_ms),
            crossfade_ms=int(args.crossfade_ms),
            extra_time_ce_ms=int(args.extra_time_ce_ms),
            extra_time_ms=int(args.extra_time_ms),
            extra_time_right_ms=int(args.extra_time_right_ms),
            diffusion_steps=int(args.diffusion_steps),
            inference_cfg_rate=float(args.inference_cfg_rate),
            seed=int(args.seed),
        )

        block = int(engine.block_frame)
        hop_seconds = block / float(sr)
        hop_samples_io = int(round(hop_seconds * float(io_sr)))
        hop_samples_io = max(1, hop_samples_io)

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
        buf_io = np.zeros(0, dtype=np.float32)
        timings: list[float] = []

        hop_ms_eff = float(args.window_ms)
        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(hop_ms_eff, 1e-6)))
        hangover_left = 0

        window_count = 0

        # Approx algorithm delay (as per Seed-VC README). Prefill silence so the first audible
        # output doesn't start with repeated underflows.
        if (not args.passthrough) and engine is not None:
            algo_delay_sec = 2.0 * (float(args.window_ms) / 1000.0) + float(args.extra_time_right_ms) / 1000.0
            algo_delay_io = int(round(algo_delay_sec * float(io_sr)))
            if algo_delay_io > 0:
                out_buf.write(np.zeros(algo_delay_io, dtype=np.float32))

        while not stop.is_set():
            try:
                x_block = input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buf_io = np.concatenate([buf_io, x_block])
            while len(buf_io) >= hop_samples_io and not stop.is_set():
                hop_io = buf_io[:hop_samples_io]
                buf_io = buf_io[hop_samples_io:]

                if args.passthrough:
                    out_io = apply_peak_limiter(hop_io, peak_limit=float(args.peak_limit))
                    out_buf.write(out_io.astype(np.float32, copy=False))
                    window_count += 1
                    continue

                hop = resample_audio(hop_io, io_sr, sr)
                hop = normalize_len_end(hop, block)

                vad_mode = str(args.vad_mode)
                if vad_mode == "off":
                    voiced = True
                elif vad_mode == "rms":
                    voiced = not (
                        float(args.vad_db) > -200.0
                        and is_silent_rms_db(
                            hop,
                            sample_rate=sr,
                            frame_ms=float(args.vad_frame_ms),
                            silence_db=float(args.vad_db),
                        )
                    )
                elif vad_mode == "webrtc":
                    hop_16k = resample_audio(hop, sr, 16000)
                    voiced = is_voiced_webrtcvad(
                        hop_16k,
                        sample_rate=16000,
                        frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                        aggressiveness=int(args.vad_webrtc_aggressiveness),
                        min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                    )
                else:
                    raise ValueError(f"Unknown vad_mode: {vad_mode}")

                if not voiced and hangover_left > 0:
                    voiced = True
                    hangover_left -= 1
                elif voiced:
                    hangover_left = hangover_hops

                if not voiced:
                    out_block = np.zeros(block, dtype=np.float32)
                    engine.sola_buffer[:] = 0.0  # type: ignore[union-attr]
                else:
                    t0 = time.perf_counter()
                    out_block = engine.step(hop=hop, window_idx=window_count)  # type: ignore[union-attr]
                    _torch_sync(device)
                    timings.append(time.perf_counter() - t0)
                    if len(timings) > 200:
                        timings = timings[-200:]

                out_block = apply_peak_limiter(out_block, peak_limit=float(args.peak_limit))
                out_io = resample_audio(out_block, sr, io_sr)
                out_io = normalize_len_end(out_io, hop_samples_io)
                out_io = apply_peak_limiter(out_io, peak_limit=float(args.peak_limit))
                out_buf.write(out_io)
                window_count += 1

                if window_count % max(1, int(2.5 / max(hop_seconds, 1e-6))) == 0 and timings:
                    mean_sec = float(np.mean(np.asarray(timings, dtype=np.float64)))
                    rtf = mean_sec / max(hop_seconds, 1e-9)
                    print(
                        f"[seedvc_live_local] windows={window_count} mean_win_s={mean_sec:.3f} rtf={rtf:.2f} "
                        f"out_buf_s={out_buf.size()/io_sr:.2f} underflows={out_buf.underflows} "
                        f"overflows={out_buf.overflows} in_q={input_q.qsize()}",
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
        f"[seedvc_live_local] io_sr={io_sr} model_sr={sr} block={args.window_ms}ms fp16={bool(args.fp16)} "
        f"diffusion_steps={int(args.diffusion_steps)} passthrough={bool(args.passthrough)}",
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


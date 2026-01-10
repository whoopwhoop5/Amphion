# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.streaming_utils import apply_peak_limiter, is_silent_rms_db, is_voiced_webrtcvad


@dataclass(frozen=True)
class StreamConfig:
    block_size: int
    extra_size: int
    use_phase_vocoder: bool
    f0_estimation: str
    vad_mode: str
    vad_db: float
    vad_frame_ms: float
    vad_hangover_ms: float
    vad_webrtc_aggressiveness: int
    vad_webrtc_frame_ms: int
    vad_webrtc_min_voiced_ratio: float
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    tinyvc_dir: str
    encoder_path: str
    decoder_path: str
    device: str
    seed: int
    pitch_shift: float
    stream: Optional[StreamConfig] = None


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _resample_if_needed(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return wav
    import torchaudio
    from torchaudio.functional import resample

    t = torch.from_numpy(wav).unsqueeze(0)
    out = resample(t, orig_freq=int(src_sr), new_freq=int(dst_sr))
    return out.squeeze(0).detach().cpu().float().numpy().astype(np.float32, copy=False)


def _frame_rms(wav: np.ndarray, frame: int, *, eps: float = 1e-9) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if frame <= 0 or len(wav) < frame:
        return np.zeros(0, dtype=np.float32)
    n = len(wav) // frame
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + eps)
    return np.asarray(rms, dtype=np.float32).reshape(-1)


def _estimate_delay_samples_energy(
    src: np.ndarray,
    deg: np.ndarray,
    *,
    sample_rate: int,
    max_delay_s: float = 2.0,
    frame_ms: float = 10.0,
    eps: float = 1e-6,
) -> int:
    """Estimate a non-negative delay (src->deg) via RMS envelope correlation.

    Returns delay_samples such that deg[t] ~ src[t + delay_samples].
    """

    src = np.asarray(src, dtype=np.float32).reshape(-1)
    deg = np.asarray(deg, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or len(src) == 0 or len(deg) == 0:
        return 0

    frame = int(round(float(frame_ms) / 1000.0 * float(sample_rate)))
    frame = max(1, frame)

    src_r = _frame_rms(src, frame)
    deg_r = _frame_rms(deg, frame)
    if len(src_r) < 4 or len(deg_r) < 4:
        return 0

    max_delay_frames = int(round(float(max_delay_s) / max(float(frame_ms) / 1000.0, 1e-9)))
    max_delay_frames = max(0, min(max_delay_frames, len(src_r) - 1))

    # Use only the first part of deg to reduce ambiguity on long signals.
    deg_cap = min(len(deg_r), int(round(8.0 / max(float(frame_ms) / 1000.0, 1e-9))))
    deg_r = deg_r[:deg_cap]
    if len(deg_r) < 4:
        return 0

    best_lag = 0
    best_score = float("-inf")
    for lag in range(0, max_delay_frames + 1):
        n = min(len(deg_r), len(src_r) - lag)
        if n < 4:
            break
        a = src_r[lag : lag + n].astype(np.float32, copy=False)
        b = deg_r[:n].astype(np.float32, copy=False)

        a = (a - float(a.mean())) / float(a.std() + eps)
        b = (b - float(b.mean())) / float(b.std() + eps)
        score = float(np.mean(a * b))
        if score > best_score:
            best_score = score
            best_lag = lag

    return int(best_lag * frame)


@torch.no_grad()
def _load_tinyvc(
    *,
    tinyvc_dir: str,
    encoder_path: str,
    decoder_path: str,
    device: torch.device,
):
    _add_sys_path_first(tinyvc_dir)

    from module.infer import Generator  # type: ignore[import-not-found]
    from module.tinyvc import Decoder, Encoder  # type: ignore[import-not-found]

    encoder = Encoder().to(device).eval()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder = Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    gen = Generator(encoder, decoder).to(device).eval()
    return gen


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TinyVC runner (offline or streaming sim).")
    parser.add_argument("--tinyvc_dir", type=str, required=True, help="Path to tinyvc repo checkout.")
    parser.add_argument("--encoder_path", type=str, default="", help="Path to encoder.pt (default: tinyvc_dir/models/encoder.pt)")
    parser.add_argument("--decoder_path", type=str, default="", help="Path to decoder.pt (default: tinyvc_dir/models/decoder.pt)")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed (TinyVC is deterministic; stored for bookkeeping).")
    parser.add_argument("--pitch_shift", type=float, default=0.0, help="Pitch shift in semitones.")

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--stream", action="store_true", help="Run streaming simulation using TinyVC's SOLA wrapper.")
    parser.add_argument("--block_size", type=int, default=1920, help="Streaming block size in samples at 24kHz.")
    parser.add_argument("--extra_size", type=int, default=0, help="Optional extra context in samples.")
    parser.add_argument("--use_phase_vocoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--f0_estimation", type=str, default="harvest", choices=["harvest", "dio", "fcpe"])

    parser.add_argument("--vad_mode", type=str, default="rms", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    tinyvc_dir = os.path.abspath(str(args.tinyvc_dir))
    if not os.path.isdir(tinyvc_dir):
        raise FileNotFoundError(f"tinyvc_dir not found: {tinyvc_dir}")

    encoder_path = str(args.encoder_path).strip() or os.path.join(tinyvc_dir, "models", "encoder.pt")
    decoder_path = str(args.decoder_path).strip() or os.path.join(tinyvc_dir, "models", "decoder.pt")
    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"Missing encoder_path: {encoder_path}")
    if not os.path.isfile(decoder_path):
        raise FileNotFoundError(f"Missing decoder_path: {decoder_path}")

    torch.manual_seed(int(args.seed))

    sr = 24000
    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)
    ref_24k = _resample_if_needed(ref_wav, ref_sr, sr)
    src_24k = _resample_if_needed(src_wav, src_sr, sr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    gen = _load_tinyvc(
        tinyvc_dir=tinyvc_dir,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        device=device,
    )

    timings: list[float] = []
    delay_samples = 0

    # Reference -> target embedding.
    ref_t = torch.from_numpy(ref_24k).unsqueeze(0).to(device)
    tgt, _ = gen.encode(ref_t)

    if not args.stream:
        src_t = torch.from_numpy(src_24k).unsqueeze(0).to(device)
        t0 = time.time()
        out_t = gen.convert(
            src_t,
            tgt,
            float(args.pitch_shift),
            device=device,
            f0_estimation=str(args.f0_estimation),
        )
        timings.append(time.time() - t0)
        out = out_t.squeeze(0).detach().cpu().float().numpy().astype(np.float32, copy=False)
        out = apply_peak_limiter(out, peak_limit=float(args.peak_limit))
        sf.write(args.out, out, sr)
    else:
        from module.infer import StreamInfer  # type: ignore[import-not-found]

        block_size = int(args.block_size)
        if block_size <= 0:
            raise ValueError("block_size must be > 0")

        stream = StreamInfer(
            gen,
            target=tgt,
            pitch_shift=float(args.pitch_shift),
            device=device,
            block_size=block_size,
            extra_size=int(args.extra_size),
            use_phase_vocoder=bool(args.use_phase_vocoder),
            f0_estimation=str(args.f0_estimation),
        )
        stream.init_buffer()

        hangover_blocks = int(
            np.ceil(float(args.vad_hangover_ms) / max(1e-6, float(block_size) / float(sr) * 1000.0))
        )
        hangover_left = 0

        outs: list[np.ndarray] = []

        # For WebRTC VAD, resample each hop to 16k (supported sample-rate).
        vad_webrtc_sr = 16000

        for start in range(0, len(src_24k), block_size):
            block = src_24k[start : start + block_size]
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)), mode="constant")

            vad_mode = str(args.vad_mode)
            if vad_mode == "off":
                voiced = True
            elif vad_mode == "rms":
                voiced = not (
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        block,
                        sample_rate=sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )
            elif vad_mode == "webrtc":
                block_16k = _resample_if_needed(block, sr, vad_webrtc_sr)
                voiced = is_voiced_webrtcvad(
                    block_16k,
                    sample_rate=vad_webrtc_sr,
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
                hangover_left = hangover_blocks

            block_t = torch.from_numpy(block).to(device)
            t0 = time.time()
            out_t = stream.audio_callback(block_t)
            timings.append(time.time() - t0)

            out = out_t.detach().cpu().float().numpy().astype(np.float32, copy=False)
            if not voiced:
                out = np.zeros_like(out)
            out = apply_peak_limiter(out, peak_limit=float(args.peak_limit))
            outs.append(out)

        out_full = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        delay_samples = _estimate_delay_samples_energy(
            src_24k,
            out_full,
            sample_rate=sr,
        )
        sf.write(args.out, out_full, sr)

    if args.meta_json:
        meta_p = Path(args.meta_json)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        stream_cfg = (
            StreamConfig(
                block_size=int(args.block_size),
                extra_size=int(args.extra_size),
                use_phase_vocoder=bool(args.use_phase_vocoder),
                f0_estimation=str(args.f0_estimation),
                vad_mode=str(args.vad_mode),
                vad_db=float(args.vad_db),
                vad_frame_ms=float(args.vad_frame_ms),
                vad_hangover_ms=float(args.vad_hangover_ms),
                vad_webrtc_aggressiveness=int(args.vad_webrtc_aggressiveness),
                vad_webrtc_frame_ms=int(args.vad_webrtc_frame_ms),
                vad_webrtc_min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )

        cfg = RunConfig(
            tinyvc_dir=str(tinyvc_dir),
            encoder_path=str(encoder_path),
            decoder_path=str(decoder_path),
            device=str(device),
            seed=int(args.seed),
            pitch_shift=float(args.pitch_shift),
            stream=stream_cfg,
        )

        stats = {
            "delay_samples": int(delay_samples),
            "mean_window_sec": float(np.mean(np.asarray(timings, dtype=np.float64))) if timings else 0.0,
            "p95_window_sec": float(np.percentile(np.asarray(timings, dtype=np.float64), 95))
            if len(timings) >= 2
            else (float(timings[0]) if timings else 0.0),
            "windows": int(len(timings)),
        }
        meta_p.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2))

    print(f"Wrote: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


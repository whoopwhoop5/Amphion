# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    fade_ms: int
    normalize_align: str
    vad_db: float
    vad_frame_ms: float
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    ckpt_dir: str
    device: str
    tau: float
    ref_se_sec: float
    src_se_sec: float
    stream: Optional[StreamConfig] = None


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _resample_if_needed(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32, copy=False)


def _extract_se_from_audio(
    *,
    converter,
    wav: np.ndarray,
    sample_rate: int,
    max_sec: float,
    device: torch.device,
) -> torch.Tensor:
    from openvoice.mel_processing import spectrogram_torch

    hps = converter.hps
    target_sr = int(hps.data.sampling_rate)

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if max_sec > 0:
        wav = wav[: int(round(max_sec * sample_rate))]

    wav = _resample_if_needed(wav, sample_rate, target_sr)

    y = torch.from_numpy(wav).float().to(device).unsqueeze(0)
    spec = spectrogram_torch(
        y,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    ).to(device)
    with torch.no_grad():
        g = converter.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1).detach()
    return g


@torch.no_grad()
def _convert_audio_window(
    *,
    converter,
    window: np.ndarray,
    sample_rate: int,
    src_se: torch.Tensor,
    tgt_se: torch.Tensor,
    tau: float,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    from openvoice.mel_processing import spectrogram_torch

    hps = converter.hps
    target_sr = int(hps.data.sampling_rate)

    window = np.asarray(window, dtype=np.float32).reshape(-1)
    window = _resample_if_needed(window, sample_rate, target_sr)

    y = torch.from_numpy(window).float().to(device).unsqueeze(0)
    spec = spectrogram_torch(
        y,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    ).to(device)
    spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)

    out = (
        converter.model.voice_conversion(
            spec,
            spec_lengths,
            sid_src=src_se,
            sid_tgt=tgt_se,
            tau=float(tau),
        )[0][0, 0]
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    return np.asarray(out, dtype=np.float32).reshape(-1), target_sr


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenVoice tone-color VC runner (offline or streaming simulation)."
    )
    parser.add_argument("--ckpt_dir", type=str, required=True, help="e.g., OpenVoice/checkpoints_v2/converter")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--tau", type=float, default=0.3, help="OpenVoice voice_conversion tau")
    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--ref_se_sec", type=float, default=10.0, help="Seconds used to extract target embedding")
    parser.add_argument("--src_se_sec", type=float, default=10.0, help="Seconds used to extract source embedding")

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument("--window_ms", type=int, default=600)
    parser.add_argument("--hop_ms", type=int, default=600)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    # Import OpenVoice lazily so this script is importable without it.
    from openvoice.api import ToneColorConverter

    from models.vc.vevo.live_engine import (
        AudioRingBuffer,
        apply_peak_limiter,
        is_silent_rms_db,
        normalize_length,
        smooth_boundary_inplace,
    )

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    ckpt_dir = Path(args.ckpt_dir)
    config_path = ckpt_dir / "config.json"
    ckpt_path = ckpt_dir / "checkpoint.pth"
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing OpenVoice converter ckpt: {ckpt_dir}")

    converter = ToneColorConverter(str(config_path), device=str(device), enable_watermark=False)
    converter.load_ckpt(str(ckpt_path))

    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)

    tgt_se = _extract_se_from_audio(
        converter=converter,
        wav=ref_wav,
        sample_rate=ref_sr,
        max_sec=float(args.ref_se_sec),
        device=device,
    ).to(device)
    src_se = _extract_se_from_audio(
        converter=converter,
        wav=src_wav,
        sample_rate=src_sr,
        max_sec=float(args.src_se_sec),
        device=device,
    ).to(device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0

    if not args.stream:
        t0 = time.time()
        out, out_sr = _convert_audio_window(
            converter=converter,
            window=src_wav,
            sample_rate=src_sr,
            src_se=src_se,
            tgt_se=tgt_se,
            tau=float(args.tau),
            device=device,
        )
        timings.append(time.time() - t0)
        sf.write(args.out, out, out_sr)
    else:
        target_sr = int(converter.hps.data.sampling_rate)
        src_ov = _resample_if_needed(src_wav, src_sr, target_sr)
        src_ov = np.asarray(src_ov, dtype=np.float32).reshape(-1)

        window_samples = int(round(float(args.window_ms) / 1000.0 * float(target_sr)))
        hop_samples = int(round(float(args.hop_ms) / 1000.0 * float(target_sr)))
        fade_samples = int(round(float(args.fade_ms) / 1000.0 * float(target_sr)))
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("window_ms and hop_ms must be > 0")
        if hop_samples > window_samples:
            raise ValueError("hop_ms must be <= window_ms")

        ring = AudioRingBuffer(window_samples)
        prev_last: Optional[float] = None

        # Same “hearable” timeline convention as Vevo live_local:
        # start with one hop of output silence, then emit silence while warming up.
        outs: list[np.ndarray] = [np.zeros(hop_samples, dtype=np.float32)]
        window_count = 0

        for start in range(0, len(src_ov), hop_samples):
            hop = src_ov[start : start + hop_samples]
            if len(hop) < hop_samples:
                hop = np.pad(hop, (0, hop_samples - len(hop)), mode="constant")
            ring.write(hop)

            if ring.size < window_samples:
                prev_last = 0.0
                outs.append(np.zeros(hop_samples, dtype=np.float32))
                continue

            window = ring.read_last(window_samples)

            silent = float(args.vad_db) > -200.0 and is_silent_rms_db(
                hop,
                sample_rate=target_sr,
                frame_ms=float(args.vad_frame_ms),
                silence_db=float(args.vad_db),
            )
            if silent:
                out_window = np.zeros(window_samples, dtype=np.float32)
                out_sr = target_sr
            else:
                t0 = time.time()
                out_window, out_sr = _convert_audio_window(
                    converter=converter,
                    window=window,
                    sample_rate=target_sr,
                    src_se=src_se,
                    tgt_se=tgt_se,
                    tau=float(args.tau),
                    device=device,
                )
                timings.append(time.time() - t0)

            if out_sr != target_sr:
                # Should not happen, but keep behavior explicit.
                out_window = _resample_if_needed(out_window, out_sr, target_sr)

            out_window = normalize_length(out_window, window_samples, align=args.normalize_align)  # type: ignore[arg-type]
            out_hop = out_window[-hop_samples:].astype(np.float32, copy=False)
            out_hop = smooth_boundary_inplace(out_hop, prev_last, fade_samples)
            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
            prev_last = float(out_hop[-1]) if len(out_hop) else prev_last

            outs.append(out_hop)
            window_count += 1

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        # Account for the initial hop delay, matching vevo_live live_local behavior.
        out = out[: len(src_ov) + hop_samples]
        sf.write(args.out, out, target_sr)
        delay_samples = int(window_samples - hop_samples)

    if args.meta_json:
        meta_p = Path(args.meta_json)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        stream_cfg = (
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                fade_ms=int(args.fade_ms),
                normalize_align=str(args.normalize_align),
                vad_db=float(args.vad_db),
                vad_frame_ms=float(args.vad_frame_ms),
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )
        cfg = RunConfig(
            ckpt_dir=str(ckpt_dir),
            device=str(device),
            tau=float(args.tau),
            ref_se_sec=float(args.ref_se_sec),
            src_se_sec=float(args.src_se_sec),
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


# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.playlist import load_vc_playlist_manifest
from evaluation.vc_quest.seedvc_convert import (
    SeedVCStreamingEngine,
    _load_audio_mono,
    _load_seedvc_models,
    _offline_convert,
    _resample_np,
)
from evaluation.vc_quest.streaming_utils import (
    VadFrameMs,
    apply_peak_limiter,
    build_rms_mask,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    rms_db,
)


VadMode = Literal["rms", "webrtc", "off"]
GainMode = Literal["off", "match_src_rms"]
MaskMode = Literal["off", "rms"]


def _set_determinism(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _case_id(source_id: str, target_id: str) -> str:
    return f"{source_id}__to__{target_id}"


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    emit_align: str
    crossfade_ms: int
    extra_time_ce_ms: int
    extra_time_ms: int
    extra_time_right_ms: int
    drop_warmup_hops: bool
    diffusion_steps: int
    inference_cfg_rate: float
    max_prompt_length_sec: float
    vad_mode: VadMode
    vad_db: float
    vad_frame_ms: float
    vad_hangover_ms: float
    vad_webrtc_frame_ms: VadFrameMs
    vad_webrtc_aggressiveness: int
    vad_webrtc_min_voiced_ratio: float
    gain_mode: GainMode
    gain_target_delta_db: float
    gain_max_boost_db: float
    gain_smoothing: float
    mask_mode: MaskMode
    mask_db: float
    mask_frame_ms: float
    mask_smooth_ms: float
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    seedvc_dir: str
    device: str
    seed: int
    checkpoint_path: str
    config_path: str
    hf_repo: str
    hf_checkpoint_name: str
    hf_config_name: str
    fp16: bool
    length_adjust: float
    diffusion_steps: int
    inference_cfg_rate: float
    max_prompt_length_sec: float
    stream: Optional[StreamConfig]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Seed-VC over a playlist manifest (offline or streaming sim).")
    parser.add_argument("--manifest", type=str, required=True, help="Playlist manifest.json")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for wavs/meta (ignored by git).")

    parser.add_argument("--seedvc_dir", type=str, required=True, help="Path to seed-vc repo checkout.")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-case adds index).")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--checkpoint_path", type=str, default="", help="Optional checkpoint (else downloads from HF).")
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

    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--inference_cfg_rate", type=float, default=0.7)
    parser.add_argument("--length_adjust", type=float, default=1.0)
    parser.add_argument("--max_prompt_length_sec", type=float, default=3.0)

    parser.add_argument("--stream", action="store_true", help="Run streaming simulation (block-based + SOLA).")
    parser.add_argument("--window_ms", type=int, default=300, help="Chunk size (maps to Seed-VC block_time).")
    parser.add_argument("--hop_ms", type=int, default=300, help="Chunk hop (must equal window_ms for Seed-VC).")
    parser.add_argument("--emit_align", type=str, default="center", choices=["start", "center", "end"])
    parser.add_argument("--crossfade_ms", type=int, default=40)
    parser.add_argument("--extra_time_ce_ms", type=int, default=2500)
    parser.add_argument("--extra_time_ms", type=int, default=500)
    parser.add_argument("--extra_time_right_ms", type=int, default=20)
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop algorithmic delay so output aligns to source start (recommended for eval).",
    )
    parser.add_argument("--vad_mode", type=str, default="rms", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
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
    parser.add_argument("--max_pairs", type=int, default=0, help="If >0, limit number of pairs (smoke tests).")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip pairs whose output wav already exists.",
    )
    args = parser.parse_args(argv)

    manifest = load_vc_playlist_manifest(str(args.manifest))

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wavs"
    meta_dir = out_dir / "meta"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    seedvc_dir = os.path.abspath(str(args.seedvc_dir))
    if not os.path.isdir(seedvc_dir):
        raise FileNotFoundError(f"seedvc_dir not found: {seedvc_dir}")

    device = (
        torch.device(str(args.device))
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    )

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

    pairs = list(manifest.pairs)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]

    # Cache ref audio per target.
    ref_cache: dict[str, np.ndarray] = {}

    for pair_idx, pair in enumerate(pairs):
        s = manifest.sources[pair.source_id]
        t = manifest.targets[pair.target_id]
        cid = _case_id(pair.source_id, pair.target_id)

        out_wav = wav_dir / f"{cid}.wav"
        out_meta = meta_dir / f"{cid}.json"
        if bool(args.resume) and out_wav.exists() and out_meta.exists():
            continue

        if pair.target_id not in ref_cache:
            ref_cache[pair.target_id], _ = _load_audio_mono(t.wav_path, sr=sr)

        ref_wav = ref_cache[pair.target_id]
        src_wav, _ = _load_audio_mono(s.wav_path, sr=sr)

        timings: list[float] = []
        delay_samples = 0
        warmup_hops = 0

        case_seed = int(args.seed) + int(pair_idx)
        _set_determinism(case_seed)

        if not bool(args.stream):
            _torch_sync(device)
            t0 = time.perf_counter()
            out = _offline_convert(
                model_set=model_set,
                device=device,
                src_wav=src_wav,
                ref_wav=ref_wav,
                seed=case_seed,
                diffusion_steps=int(args.diffusion_steps),
                inference_cfg_rate=float(args.inference_cfg_rate),
                length_adjust=float(args.length_adjust),
                max_prompt_length_sec=float(args.max_prompt_length_sec),
            )
            _torch_sync(device)
            timings.append(time.perf_counter() - t0)
            sf.write(str(out_wav), out, sr)
        else:
            if int(args.hop_ms) != int(args.window_ms):
                raise ValueError("Seed-VC streaming uses hop_ms == window_ms (block_time).")
            block_ms = int(args.window_ms)
            hop_ms = int(args.hop_ms)

            engine = SeedVCStreamingEngine(
                model_set=model_set,
                device=device,
                reference_wav=ref_wav,
                max_prompt_length_sec=float(args.max_prompt_length_sec),
                fp16=bool(args.fp16),
                block_ms=block_ms,
                crossfade_ms=int(args.crossfade_ms),
                extra_time_ce_ms=int(args.extra_time_ce_ms),
                extra_time_ms=int(args.extra_time_ms),
                extra_time_right_ms=int(args.extra_time_right_ms),
                diffusion_steps=int(args.diffusion_steps),
                inference_cfg_rate=float(args.inference_cfg_rate),
                seed=case_seed,
            )

            block = int(engine.block_frame)

            # Approx algorithm delay (as per Seed-VC README).
            algo_delay_sec = 2.0 * (float(block_ms) / 1000.0) + float(args.extra_time_right_ms) / 1000.0
            algo_delay_samples = int(round(algo_delay_sec * float(sr)))

            # Run for extra time to flush tail.
            pad_len = int(algo_delay_samples + block)
            src_pad = np.pad(src_wav, (0, pad_len), mode="constant")

            outs: list[np.ndarray] = []
            hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(hop_ms), 1e-6)))
            hangover_left = 0
            gain_db_state = 0.0

            for i, start in enumerate(range(0, len(src_pad), block)):
                hop = src_pad[start : start + block]
                if len(hop) < block:
                    hop = np.pad(hop, (0, block - len(hop)), mode="constant")

                vad_segment = hop
                silent_rms = bool(
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        vad_segment,
                        sample_rate=sr,
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
                    hop_16k = _resample_np(hop, orig_sr=sr, target_sr=16000)
                    webrtc_voiced = is_voiced_webrtcvad(
                        hop_16k,
                        sample_rate=16000,
                        frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                        aggressiveness=int(args.vad_webrtc_aggressiveness),
                        min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                    )
                    voiced = bool(webrtc_voiced) and (not silent_rms)
                else:
                    raise ValueError(f"Unknown vad_mode: {vad_mode}")

                # Hangover is only allowed to override when we're not truly silent by RMS.
                if not voiced and hangover_left > 0 and (not silent_rms):
                    voiced = True
                    hangover_left -= 1
                elif voiced:
                    hangover_left = hangover_hops

                if not voiced:
                    out_block = np.zeros(block, dtype=np.float32)
                    engine.sola_buffer[:] = 0.0
                else:
                    _torch_sync(device)
                    t0 = time.perf_counter()
                    out_block = engine.step(hop=hop, window_idx=i)
                    _torch_sync(device)
                    timings.append(time.perf_counter() - t0)

                if voiced and str(args.gain_mode) == "match_src_rms":
                    alpha = float(np.clip(float(args.gain_smoothing), 0.0, 0.999))
                    src_db = rms_db(vad_segment, eps=1e-9)
                    out_db = rms_db(out_block, eps=1e-9)
                    desired_boost_db = (src_db - out_db) - float(args.gain_target_delta_db)
                    desired_boost_db = float(np.clip(desired_boost_db, 0.0, float(args.gain_max_boost_db)))
                    gain_db_state = alpha * gain_db_state + (1.0 - alpha) * desired_boost_db
                    gain = float(10.0 ** (gain_db_state / 20.0))
                    out_block = (out_block * gain).astype(np.float32, copy=False)
                elif not voiced:
                    gain_db_state *= float(np.clip(float(args.gain_smoothing), 0.0, 0.999))

                if str(args.mask_mode) == "rms":
                    mask = build_rms_mask(
                        vad_segment,
                        in_sample_rate=sr,
                        out_sample_rate=sr,
                        out_len=block,
                        frame_ms=float(args.mask_frame_ms),
                        threshold_db=float(args.mask_db),
                        smooth_ms=float(args.mask_smooth_ms),
                    )
                    out_block = (out_block * mask).astype(np.float32, copy=False)

                out_block = apply_peak_limiter(out_block, peak_limit=float(args.peak_limit))
                outs.append(out_block.astype(np.float32, copy=False))

            out_full = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
            if bool(args.drop_warmup_hops):
                delay_samples = 0
                start = int(max(0, algo_delay_samples))
                out_full = out_full[start : start + len(src_wav)]
            else:
                delay_samples = int(algo_delay_samples)
                out_full = out_full[: len(src_wav)]
            warmup_hops = int(np.ceil(float(algo_delay_samples) / float(block))) if algo_delay_samples > 0 else 0
            out_full = np.asarray(out_full, dtype=np.float32).reshape(-1)
            sf.write(str(out_wav), out_full, sr)

        stream_cfg = (
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                emit_align=str(args.emit_align),
                crossfade_ms=int(args.crossfade_ms),
                extra_time_ce_ms=int(args.extra_time_ce_ms),
                extra_time_ms=int(args.extra_time_ms),
                extra_time_right_ms=int(args.extra_time_right_ms),
                drop_warmup_hops=bool(args.drop_warmup_hops),
                diffusion_steps=int(args.diffusion_steps),
                inference_cfg_rate=float(args.inference_cfg_rate),
                max_prompt_length_sec=float(args.max_prompt_length_sec),
                vad_mode=str(args.vad_mode),  # type: ignore[arg-type]
                vad_db=float(args.vad_db),
                vad_frame_ms=float(args.vad_frame_ms),
                vad_hangover_ms=float(args.vad_hangover_ms),
                vad_webrtc_frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                vad_webrtc_aggressiveness=int(args.vad_webrtc_aggressiveness),
                vad_webrtc_min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                gain_mode=str(args.gain_mode),  # type: ignore[arg-type]
                gain_target_delta_db=float(args.gain_target_delta_db),
                gain_max_boost_db=float(args.gain_max_boost_db),
                gain_smoothing=float(args.gain_smoothing),
                mask_mode=str(args.mask_mode),  # type: ignore[arg-type]
                mask_db=float(args.mask_db),
                mask_frame_ms=float(args.mask_frame_ms),
                mask_smooth_ms=float(args.mask_smooth_ms),
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )
        cfg = RunConfig(
            seedvc_dir=str(seedvc_dir),
            device=str(device),
            seed=int(args.seed),
            checkpoint_path=str(args.checkpoint_path),
            config_path=str(args.config_path),
            hf_repo=str(args.hf_repo),
            hf_checkpoint_name=str(args.hf_checkpoint_name),
            hf_config_name=str(args.hf_config_name),
            fp16=bool(args.fp16),
            length_adjust=float(args.length_adjust),
            diffusion_steps=int(args.diffusion_steps),
            inference_cfg_rate=float(args.inference_cfg_rate),
            max_prompt_length_sec=float(args.max_prompt_length_sec),
            stream=stream_cfg,
        )

        stats = {
            "delay_samples": int(delay_samples),
            "warmup_hops": int(warmup_hops) if bool(args.stream) else 0,
            "mean_window_sec": float(np.mean(np.asarray(timings, dtype=np.float64))) if timings else 0.0,
            "p95_window_sec": float(np.percentile(np.asarray(timings, dtype=np.float64), 95))
            if len(timings) >= 2
            else (float(timings[0]) if timings else 0.0),
            "windows": int(len(timings)),
            "output_sample_rate": int(sr),
        }
        out_meta.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

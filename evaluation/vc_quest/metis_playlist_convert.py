# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.playlist import load_vc_playlist_manifest
from evaluation.vc_quest.streaming_utils import (
    AudioRingBuffer,
    apply_peak_limiter,
    build_rms_mask,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
    rms_db,
)


def _set_determinism(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr).astype(
        np.float32, copy=False
    )


def _trim_wav(wav: np.ndarray, sr: int, max_sec: float) -> np.ndarray:
    if float(max_sec) <= 0:
        return wav
    max_samples = int(round(float(max_sec) * float(sr)))
    if max_samples <= 0:
        return wav
    return np.asarray(wav[:max_samples], dtype=np.float32).reshape(-1)


@torch.no_grad()
def _infer_metis(
    *,
    metis,
    src_wav_16k: np.ndarray,
    prompt_wav_16k: np.ndarray,
    prompt_semantic_code,
    prompt_acoustic_code,
    n_timesteps: int,
    cfg: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    _set_determinism(seed)

    combine_semantic_code = metis.speech2semantic_w_prompt(
        speech=np.asarray(src_wav_16k, dtype=np.float32).reshape(-1),
        prompt_speech=np.asarray(prompt_wav_16k, dtype=np.float32).reshape(-1),
        prompt_semantic_code=prompt_semantic_code,
        steps=int(n_timesteps),
        cfg=float(cfg),
    )
    predict_acoustic_code = metis.semantic2acoustic(
        combine_semantic_code, prompt_acoustic_code
    )
    wav = metis.audio_tokenizer.code2wav(predict_acoustic_code)
    return np.asarray(wav, dtype=np.float32).reshape(-1), 24000


def _case_id(source_id: str, target_id: str) -> str:
    return f"{source_id}__to__{target_id}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Metis VC over a playlist manifest (offline or streaming sim)."
    )
    parser.add_argument("--manifest", type=str, required=True, help="Playlist manifest.json.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for wavs/meta.")

    parser.add_argument(
        "--variant",
        type=str,
        default="metis_vc",
        choices=["metis_vc", "metis_omni"],
        help="Which Metis checkpoint/config to use.",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")
    parser.add_argument("--n_timesteps", type=int, default=20, help="Metis Stage-1 diffusion timesteps.")
    parser.add_argument("--cfg", type=float, default=1.0, help="Metis Stage-1 guidance scale.")
    parser.add_argument("--ref_max_sec", type=float, default=10.0, help="Trim reference audio to N seconds (0 disables).")

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument("--window_ms", type=int, default=800)
    parser.add_argument("--hop_ms", type=int, default=400)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--emit_align", type=str, default="end", choices=["start", "center", "end"])
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop output until the first full window is available (recommended for eval).",
    )

    parser.add_argument("--vad_mode", type=str, default="rms", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)

    parser.add_argument(
        "--gain_mode",
        type=str,
        default="off",
        choices=["off", "match_src_rms"],
        help="Optional loudness compensation for streaming stability.",
    )
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

    device = str(args.device).strip() if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    from huggingface_hub import snapshot_download

    from models.tts.metis.metis import Metis
    from utils.util import load_config

    if str(args.variant) == "metis_vc":
        cfg_path = repo_root / "models" / "tts" / "metis" / "config" / "ft.json"
        ckpt_rel = "metis_vc/metis_vc.safetensors"
        allow_patterns = [ckpt_rel]
        model_type_init = "vc"
    elif str(args.variant) == "metis_omni":
        cfg_path = repo_root / "models" / "tts" / "metis" / "config" / "omni.json"
        ckpt_rel = "metis_omni/metis_omni.safetensors"
        allow_patterns = [ckpt_rel]
        model_type_init = "omni"
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    metis_cfg = load_config(str(cfg_path))
    ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir=str(repo_root / "models" / "tts" / "metis" / "ckpt"),
        allow_patterns=allow_patterns,
    )
    ckpt_path = str(Path(ckpt_dir) / ckpt_rel)

    metis = Metis(
        ckpt_path=ckpt_path,
        cfg=metis_cfg,
        device=str(device),
        model_type=str(model_type_init),
    )

    in_sr = 16000
    out_sr = 24000

    manifest = load_vc_playlist_manifest(args.manifest).resolve_paths(args.manifest)
    sources = manifest.sources_by_id()
    targets = manifest.targets_by_id()

    pairs = list(manifest.pairs)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]

    pairs_by_target: dict[str, list[tuple[int, str]]] = {}
    for pair_idx, pair in enumerate(pairs):
        pairs_by_target.setdefault(pair.target_id, []).append((pair_idx, pair.source_id))

    out_root = Path(args.out_dir)
    wav_dir = out_root / "wavs"
    meta_dir = out_root / "meta"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    prompt_cache: dict[str, tuple[np.ndarray, torch.Tensor, torch.Tensor]] = {}

    for target_id, items in pairs_by_target.items():
        if target_id not in prompt_cache:
            t = targets[target_id]
            ref_wav, ref_sr = _load_mono(t.wav_path)
            ref_wav = _trim_wav(ref_wav, ref_sr, float(args.ref_max_sec))
            prompt_16k = _resample_if_needed(ref_wav, ref_sr, in_sr)
            prompt_24k = _resample_if_needed(ref_wav, ref_sr, out_sr)
            prompt_semantic_code, _, prompt_acoustic_code = metis.audio_tokenizer(
                speech_16k=prompt_16k,
                speech=prompt_24k,
            )
            prompt_cache[target_id] = (
                np.asarray(prompt_16k, dtype=np.float32).reshape(-1),
                prompt_semantic_code,
                prompt_acoustic_code,
            )

        prompt_16k, prompt_semantic_code, prompt_acoustic_code = prompt_cache[target_id]

        for pair_idx, source_id in items:
            s = sources[source_id]
            cid = _case_id(source_id, target_id)

            out_wav = wav_dir / f"{cid}.wav"
            out_meta = meta_dir / f"{cid}.json"
            if bool(args.resume) and out_wav.exists() and out_meta.exists():
                continue

            src_wav, src_sr = _load_mono(s.wav_path)
            src_16k = _resample_if_needed(src_wav, src_sr, in_sr)

            timings: list[float] = []
            delay_samples = 0
            warmup_hops = 0

            if not bool(args.stream):
                t0 = time.time()
                out, _ = _infer_metis(
                    metis=metis,
                    src_wav_16k=src_16k,
                    prompt_wav_16k=prompt_16k,
                    prompt_semantic_code=prompt_semantic_code,
                    prompt_acoustic_code=prompt_acoustic_code,
                    n_timesteps=int(args.n_timesteps),
                    cfg=float(args.cfg),
                    seed=int(args.seed) + int(pair_idx),
                )
                timings.append(time.time() - t0)
                sf.write(str(out_wav), out, out_sr)
            else:
                window_in = int(round(float(args.window_ms) / 1000.0 * float(in_sr)))
                hop_in = int(round(float(args.hop_ms) / 1000.0 * float(in_sr)))
                if window_in <= 0 or hop_in <= 0:
                    raise ValueError("window_ms and hop_ms must be > 0")
                if hop_in > window_in:
                    raise ValueError("hop_ms must be <= window_ms")

                window_out = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
                hop_out = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
                fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))

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

                ring = AudioRingBuffer(window_in)
                prev_tail: Optional[np.ndarray] = None
                outs: list[np.ndarray] = []
                drop_warmup_hops = bool(args.drop_warmup_hops)
                window_count = 0

                hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6)))
                hangover_left = 0
                gain_db_state = 0.0

                for start in range(0, len(src_16k), hop_in):
                    hop = src_16k[start : start + hop_in]
                    if len(hop) < hop_in:
                        hop = np.pad(hop, (0, hop_in - len(hop)), mode="constant")
                    ring.write(hop)

                    if ring.size < window_in:
                        warmup_hops += 1
                        if fade_out > 0:
                            prev_tail = np.zeros(fade_out, dtype=np.float32)
                        if not drop_warmup_hops:
                            outs.append(np.zeros(hop_out, dtype=np.float32))
                        continue

                    window = ring.read_last(window_in)
                    vad_segment = window[emit_start_in : emit_start_in + hop_in]

                    vad_mode = str(args.vad_mode)
                    silent_rms = bool(
                        float(args.vad_db) > -200.0
                        and is_silent_rms_db(
                            vad_segment,
                            sample_rate=in_sr,
                            frame_ms=float(args.vad_frame_ms),
                            silence_db=float(args.vad_db),
                        )
                    )

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

                    if not voiced:
                        out_hop = np.zeros(hop_out, dtype=np.float32)
                    else:
                        t0 = time.time()
                        out_window, _ = _infer_metis(
                            metis=metis,
                            src_wav_16k=window,
                            prompt_wav_16k=prompt_16k,
                            prompt_semantic_code=prompt_semantic_code,
                            prompt_acoustic_code=prompt_acoustic_code,
                            n_timesteps=int(args.n_timesteps),
                            cfg=float(args.cfg),
                            seed=int(args.seed) + int(pair_idx) + int(window_count),
                        )
                        timings.append(time.time() - t0)
                        out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))
                        out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

                    gain_mode = str(args.gain_mode)
                    if voiced and gain_mode == "match_src_rms":
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

                    out_hop = crossfade_prefix_inplace(out_hop, prev_tail, fade_out)
                    out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
                    if fade_out > 0:
                        prev_tail = out_hop[-fade_out:].astype(np.float32, copy=True)

                    outs.append(out_hop)
                    window_count += 1

                out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
                sf.write(str(out_wav), out, out_sr)

                # Align output timeline to source for downstream scoring.
                if bool(args.drop_warmup_hops):
                    delay_samples = int(
                        int(warmup_hops) * int(hop_out)
                        + (int(hop_out) - int(window_out))
                        + int(emit_start_out)
                    )
                else:
                    delay_samples = int(emit_start_out)

            cfg = {
                "variant": str(args.variant),
                "device": str(device),
                "seed": int(args.seed),
                "n_timesteps": int(args.n_timesteps),
                "cfg": float(args.cfg),
                "ref_max_sec": float(args.ref_max_sec),
                "stream": {
                    "window_ms": int(args.window_ms),
                    "hop_ms": int(args.hop_ms),
                    "fade_ms": int(args.fade_ms),
                    "normalize_align": str(args.normalize_align),
                    "emit_align": str(args.emit_align),
                    "drop_warmup_hops": bool(args.drop_warmup_hops),
                    "vad_mode": str(args.vad_mode),
                    "vad_db": float(args.vad_db),
                    "vad_frame_ms": float(args.vad_frame_ms),
                    "vad_hangover_ms": float(args.vad_hangover_ms),
                    "vad_webrtc_aggressiveness": int(args.vad_webrtc_aggressiveness),
                    "vad_webrtc_frame_ms": int(args.vad_webrtc_frame_ms),
                    "vad_webrtc_min_voiced_ratio": float(args.vad_webrtc_min_voiced_ratio),
                    "gain_mode": str(args.gain_mode),
                    "gain_target_delta_db": float(args.gain_target_delta_db),
                    "gain_max_boost_db": float(args.gain_max_boost_db),
                    "gain_smoothing": float(args.gain_smoothing),
                    "mask_mode": str(args.mask_mode),
                    "mask_db": float(args.mask_db),
                    "mask_frame_ms": float(args.mask_frame_ms),
                    "mask_smooth_ms": float(args.mask_smooth_ms),
                    "peak_limit": float(args.peak_limit),
                }
                if bool(args.stream)
                else None,
            }

            stats = {
                "delay_samples": int(delay_samples),
                "warmup_hops": int(warmup_hops) if bool(args.stream) else 0,
                "mean_window_sec": float(np.mean(np.asarray(timings, dtype=np.float64))) if timings else 0.0,
                "p95_window_sec": float(np.percentile(np.asarray(timings, dtype=np.float64), 95))
                if len(timings) >= 2
                else (float(timings[0]) if timings else 0.0),
                "windows": int(len(timings)),
            }

            out_meta.write_text(json.dumps({"config": cfg, "stats": stats}, indent=2))

    meta = {
        "manifest": str(Path(args.manifest).resolve()),
        "model": "metis",
        "variant": str(args.variant),
        "stream": bool(args.stream),
        "out_sample_rate": int(out_sr),
        "out_dir": str(out_root.resolve()),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[metis_playlist] Wrote: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


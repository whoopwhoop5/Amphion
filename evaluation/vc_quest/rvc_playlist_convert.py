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
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from evaluation.vc_quest.playlist import load_vc_playlist_manifest
from evaluation.vc_quest.streaming_utils import (
    AudioRingBuffer,
    apply_peak_limiter,
    build_rms_mask,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
)


def _case_id(source_id: str, target_id: str) -> str:
    return f"{source_id}__to__{target_id}"


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32, copy=False)
    return wav, int(sr)


def _resample_if_needed(wav: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if int(in_sr) == int(out_sr):
        return wav.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(wav.astype(np.float32, copy=False), orig_sr=int(in_sr), target_sr=int(out_sr)).astype(
        np.float32,
        copy=False,
    )


def _prefer_rvc_modules(rvc_dir: str) -> None:
    rvc_dir = os.path.abspath(str(rvc_dir))
    if rvc_dir not in sys.path:
        sys.path.insert(0, rvc_dir)


def _load_rvc(model_name: str, *, rvc_dir: str, device: str, is_half: bool):
    # RVC's Config() parses argv; ensure our CLI flags do not leak into it.
    sys.argv = sys.argv[:1]

    os.chdir(rvc_dir)
    _prefer_rvc_modules(rvc_dir)

    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(rvc_dir, ".env"))
    except Exception:
        pass

    from configs.config import Config
    from infer.modules.vc.modules import VC
    from infer.modules.vc.utils import load_hubert

    cfg = Config()
    cfg.device = str(device)
    cfg.is_half = bool(is_half)

    vc = VC(cfg)
    if not str(model_name).endswith(".pth"):
        model_name = f"{model_name}.pth"
    vc.get_vc(str(model_name))
    if vc.hubert_model is None:
        vc.hubert_model = load_hubert(cfg)
    return vc


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run RVC WebUI (trained target voice) over a playlist manifest (offline or streaming sim)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Playlist manifest.json (see build_fleurs_playlist).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for wavs/meta (ignored by git).",
    )
    parser.add_argument("--rvc_dir", type=str, required=True, help="Path to RVC WebUI repo checkout.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model filename under RVC weight_root (e.g., myvoice.pth).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--f0method", type=str, default="rmvpe", choices=["rmvpe", "harvest", "pm", "dio", "crepe"])
    parser.add_argument("--index_path", type=str, default="", help="Optional faiss index path.")
    parser.add_argument(
        "--index_rate",
        type=float,
        default=0.0,
        help="Index blending ratio; set 0 to disable retrieval (recommended for streaming).",
    )
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--resample_sr", type=int, default=0)
    parser.add_argument("--rms_mix_rate", type=float, default=1.0)
    parser.add_argument("--protect", type=float, default=0.33)

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument("--window_ms", type=int, default=800)
    parser.add_argument("--hop_ms", type=int, default=400)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--emit_align", type=str, default="center", choices=["start", "center", "end"])
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

    parser.add_argument("--mask_mode", type=str, default="off", choices=["off", "rms"])
    parser.add_argument("--mask_db", type=float, default=-50.0)
    parser.add_argument("--mask_frame_ms", type=float, default=10.0)
    parser.add_argument("--mask_smooth_ms", type=float, default=10.0)
    parser.add_argument("--peak_limit", type=float, default=0.99)

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="If >0, limit number of pairs (smoke tests).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip pairs whose output wav+meta already exist.",
    )
    args = parser.parse_args(argv)

    rvc_dir = os.path.abspath(str(args.rvc_dir))
    if not os.path.isdir(rvc_dir):
        raise FileNotFoundError(f"rvc_dir not found: {rvc_dir}")

    vc = _load_rvc(
        str(args.model_name),
        rvc_dir=rvc_dir,
        device=str(args.device),
        is_half=bool(args.half),
    )

    in_sr = 16000
    out_sr = int(args.resample_sr) if int(args.resample_sr) >= 16000 else int(vc.tgt_sr)

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

    # RVC is trained per-target voice, but playlist pairs may include multiple targets.
    # We still iterate by target_id so callers can filter manifests externally and keep output stable.
    for target_id, items in pairs_by_target.items():
        _ = targets.get(target_id)
        for _, source_id in items:
            if source_id not in sources:
                raise KeyError(f"Missing source_id in manifest.sources: {source_id}")

        for pair_idx, source_id in items:
            cid = _case_id(source_id, target_id)

            out_wav = wav_dir / f"{cid}.wav"
            out_meta = meta_dir / f"{cid}.json"
            if bool(args.resume) and out_wav.exists() and out_meta.exists():
                continue

            s = sources[source_id]

            timings: list[float] = []
            delay_samples = 0
            warmup_hops = 0

            if not bool(args.stream):
                t0 = time.time()
                info, opt = vc.vc_single(
                    0,
                    str(s.wav_path),
                    int(args.f0up_key),
                    None,
                    str(args.f0method),
                    str(args.index_path),
                    None,
                    float(args.index_rate),
                    int(args.filter_radius),
                    int(args.resample_sr),
                    float(args.rms_mix_rate),
                    float(args.protect),
                )
                if opt[0] is None or opt[1] is None:
                    raise RuntimeError(f"RVC inference failed for {cid}: {info}")
                sr_out, audio_int16 = opt
                timings.append(time.time() - t0)
                audio = (audio_int16.astype(np.float32) / 32768.0).reshape(-1)
                sf.write(str(out_wav), audio, int(sr_out))
            else:
                src_wav, src_sr = _load_mono(s.wav_path)
                src_16k = _resample_if_needed(src_wav, src_sr, in_sr)

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

                hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6)))
                hangover_left = 0

                # Streaming sim: disable retrieval by default (index would reload per window in upstream Pipeline).
                index_path = str(args.index_path) if float(args.index_rate) > 0.0 else ""
                index_rate = float(args.index_rate) if index_path else 0.0

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

                    if not voiced:
                        out_hop = np.zeros(hop_out, dtype=np.float32)
                    else:
                        times_rvc = [0.0, 0.0, 0.0]
                        t0 = time.time()
                        out_int16 = vc.pipeline.pipeline(
                            vc.hubert_model,
                            vc.net_g,
                            0,
                            window.astype(np.float32, copy=False),
                            cid,
                            times_rvc,
                            int(args.f0up_key),
                            str(args.f0method),
                            index_path,
                            index_rate,
                            int(vc.if_f0 or 1),
                            int(args.filter_radius),
                            int(vc.tgt_sr),
                            int(args.resample_sr),
                            float(args.rms_mix_rate),
                            str(vc.version or "v1"),
                            float(args.protect),
                            None,
                        )
                        timings.append(time.time() - t0)
                        out_window = (out_int16.astype(np.float32) / 32768.0).reshape(-1)
                        out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))
                        out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

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

                out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
                sf.write(str(out_wav), out, int(out_sr))

                if bool(args.drop_warmup_hops):
                    delay_samples = int(
                        int(warmup_hops) * int(hop_out) + (int(hop_out) - int(window_out)) + int(emit_start_out)
                    )
                else:
                    delay_samples = int(emit_start_out)

            cfg = {
                "rvc_dir": str(rvc_dir),
                "model_name": str(args.model_name),
                "device": str(args.device),
                "half": bool(args.half),
                "f0up_key": int(args.f0up_key),
                "f0method": str(args.f0method),
                "index_path": str(args.index_path),
                "index_rate": float(args.index_rate),
                "filter_radius": int(args.filter_radius),
                "resample_sr": int(args.resample_sr),
                "rms_mix_rate": float(args.rms_mix_rate),
                "protect": float(args.protect),
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
        "model": "rvc",
        "stream": bool(args.stream),
        "out_sample_rate": int(out_sr),
        "out_dir": str(out_root.resolve()),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[rvc_playlist] Wrote: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


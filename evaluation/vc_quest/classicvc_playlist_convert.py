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

from evaluation.vc_quest.classicvc_convert import (
    _DummySoundControl,
    _compute_style_embedding,
    _import_mmcxli_audio_efx,
    _load_mono,
    _resample_if_needed,
    _resolve_weight,
    _trim_wav,
)
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


def _case_id(source_id: str, target_id: str) -> str:
    return f"{source_id}__to__{target_id}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ClassicVC/MMCXLI over a playlist manifest (offline or streaming sim)."
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

    parser.add_argument("--mmcxli_dir", type=str, required=True, help="Path to MMCXLI repo checkout.")
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX Runtime device selection via MMCXLI config (default: cuda).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (reserved).")
    parser.add_argument(
        "--ref_max_sec",
        type=float,
        default=10.0,
        help="Trim reference audio to this many seconds (0 disables).",
    )

    parser.add_argument(
        "--stream", action="store_true", help="Run window/hop streaming simulation."
    )
    parser.add_argument("--window_ms", type=int, default=800)
    parser.add_argument("--hop_ms", type=int, default=400)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument(
        "--normalize_align", type=str, default="end", choices=["start", "end"]
    )
    parser.add_argument(
        "--emit_align", type=str, default="center", choices=["start", "center", "end"]
    )
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop output until the first full window is available (recommended for eval).",
    )

    parser.add_argument(
        "--vad_mode", type=str, default="rms", choices=["rms", "webrtc", "off"]
    )
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
    parser.add_argument(
        "--gain_target_delta_db",
        type=float,
        default=10.0,
        help="When gain_mode=match_src_rms, aim for output RMS ≈ (input RMS - gain_target_delta_db).",
    )
    parser.add_argument("--gain_max_boost_db", type=float, default=18.0)
    parser.add_argument("--gain_smoothing", type=float, default=0.0)

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

    mmcxli_dir = os.path.abspath(str(args.mmcxli_dir))
    AudioEfx, load_make_vc_config = _import_mmcxli_audio_efx(mmcxli_dir)

    vc_cfg = load_make_vc_config("/tmp/mmcxli_vc_config.json", save=False)
    vc_cfg["model"]["model_device"] = str(args.model_device)
    vc_cfg["auto_encode"] = False
    vc_cfg["spec_rt_o"] = 2  # avoid GUI-only plotting paths

    wdir = Path(mmcxli_dir) / "weights"
    vc_cfg["model"]["harmof0_ckpt"] = _resolve_weight(wdir / "harmof0.onnx", "harmof0.onnx")
    vc_cfg["model"]["CE_ckpt"] = _resolve_weight(wdir / "hubert500.onnx", "hubert500.onnx")
    vc_cfg["model"]["SE_ckpt"] = _resolve_weight(wdir / "style_encoder_304.onnx", "style_encoder_304.onnx")
    vc_cfg["model"]["f0n_ckpt"] = _resolve_weight(
        wdir / "f0n_predictor_hubert500.onnx", "f0n_predictor_hubert500.onnx"
    )
    vc_cfg["model"]["decoder_ckpt"] = _resolve_weight(wdir / "decoder_24k.onnx", "decoder_24k.onnx")
    vc_cfg["model"]["style_compressor_ckpt"] = _resolve_weight(
        wdir / "pumap_encoder_2dim.onnx", "pumap_encoder_2dim.onnx"
    )
    vc_cfg["model"]["style_decoder_ckpt"] = _resolve_weight(
        wdir / "pumap_decoder_2dim.onnx", "pumap_decoder_2dim.onnx"
    )

    in_sr = 16000
    out_sr = int(vc_cfg["backend"]["sr_decode"])

    sc = _DummySoundControl(
        sr_out=out_sr,
        sr_proc=in_sr,
        block_roll_size=int(vc_cfg["backend"]["block_roll_size"]),
        content_expand_rate=float(vc_cfg["content_expand_rate"]),
    )
    audio_efx = AudioEfx(sc=sc, vc_config=vc_cfg, hop_size=160, dim_spec=352, ch_map=[0], bypass=False)

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

    for target_id, items in pairs_by_target.items():
        t = targets[target_id]
        ref_wav, ref_sr = _load_mono(t.wav_path)
        ref_wav = _trim_wav(ref_wav, ref_sr, float(args.ref_max_sec))
        ref_16k = _resample_if_needed(ref_wav, ref_sr, in_sr)
        style = _compute_style_embedding(audio_efx, ref_16k)
        sc.current_target_style = np.asarray(style, dtype=np.float32).reshape(1, -1)

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
                out = audio_efx.convert_offline(src_16k[None, :])
                timings.append(time.time() - t0)
                out = np.asarray(out, dtype=np.float32).reshape(-1)
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
                        out_window = audio_efx.convert_offline(window[None, :])
                        timings.append(time.time() - t0)
                        out_window = np.asarray(out_window, dtype=np.float32).reshape(-1)
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
                "mmcxli_dir": str(mmcxli_dir),
                "model_device": str(args.model_device),
                "seed": int(args.seed),
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
        "model": "classicvc",
        "stream": bool(args.stream),
        "out_sample_rate": int(out_sr),
        "out_dir": str(out_root.resolve()),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[classicvc_playlist] Wrote: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


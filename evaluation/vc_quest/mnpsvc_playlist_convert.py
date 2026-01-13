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

    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32, copy=False)


def _trim_to_max_sec(wav: np.ndarray, *, sample_rate: int, max_sec: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if max_sec <= 0 or sample_rate <= 0:
        return wav
    max_len = int(round(float(max_sec) * float(sample_rate)))
    if max_len <= 0 or len(wav) <= max_len:
        return wav
    return wav[:max_len]


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _prefer_mnpsvc_modules(mnpsvc_dir: str) -> None:
    """Ensure MNP-SVC's `modules/*` wins over Amphion's `modules/*`."""

    mnpsvc_dir = os.path.abspath(str(mnpsvc_dir))
    amphion_root = str(Path(__file__).resolve().parents[2])
    cwd_abs = os.path.abspath(os.getcwd())

    new_path: list[str] = []
    for p in sys.path:
        p_abs = os.path.abspath(p) if p else cwd_abs
        if p_abs == amphion_root:
            continue
        new_path.append(p)
    sys.path = new_path

    _add_sys_path_first(mnpsvc_dir)

    for k in list(sys.modules.keys()):
        if k == "modules" or k.startswith("modules."):
            del sys.modules[k]


def _case_id(source_id: str, target_id: str) -> str:
    return f"{source_id}__to__{target_id}"


def _build_response_mask(volume: np.ndarray, *, response_threshold_db: float) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32).reshape(-1)
    if len(vol) == 0:
        return np.zeros(0, dtype=np.float32)

    thr = float(10.0 ** (float(response_threshold_db) / 20.0))
    mask = (vol > thr).astype(np.float32, copy=False)
    mask = np.pad(mask, (4, 4), constant_values=(float(mask[0]), float(mask[-1])))
    mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)], dtype=np.float32)
    return mask


@torch.inference_mode()
def _infer_mnpsvc(
    *,
    model,
    units_encoder,
    f0_extractor,
    volume_extractor,
    upsample,
    src_wav: np.ndarray,
    sample_rate: int,
    hop_size: int,
    device: torch.device,
    response_threshold_db: float,
    spk_id: torch.Tensor,
    spk_mix: torch.Tensor,
) -> np.ndarray:
    src_wav = np.asarray(src_wav, dtype=np.float32).reshape(-1)
    if len(src_wav) == 0:
        return np.zeros(0, dtype=np.float32)

    wav_t = torch.from_numpy(src_wav).float().unsqueeze(0).to(device)
    units = units_encoder.encode(wav_t, sample_rate, hop_size)

    f0 = f0_extractor.extract(src_wav, uv_interp=True, device=str(device), silence_front=0)
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    volume = volume_extractor.extract(src_wav)
    volume = np.asarray(volume, dtype=np.float32).reshape(-1)

    f0_t = torch.from_numpy(f0).float().to(device).unsqueeze(0).unsqueeze(-1)
    vol_t = torch.from_numpy(volume).float().to(device).unsqueeze(0).unsqueeze(-1)

    out = model(units, f0_t, vol_t, spk_id=spk_id, spk_mix=spk_mix)
    out_t = out if isinstance(out, torch.Tensor) else torch.from_numpy(np.asarray(out))
    out_t = out_t.squeeze()

    mask_frames = _build_response_mask(volume, response_threshold_db=float(response_threshold_db))
    mask_t = torch.from_numpy(mask_frames).float().to(device).unsqueeze(0).unsqueeze(-1)
    mask_up = upsample(mask_t, int(hop_size)).squeeze(-1).reshape(-1)

    n = int(min(out_t.numel(), mask_up.numel()))
    out_t = out_t.reshape(-1)[:n] * mask_up[:n]
    out_np = out_t.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(out_np, dtype=np.float32).reshape(-1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run MNP-SVC over a playlist manifest (offline or streaming sim).")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mnpsvc_dir", type=str, required=True, help="Path to MNP-SVC repo checkout.")
    parser.add_argument("--model_path", type=str, default="", help="Path to model weights (.bin/.pt).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--pitch_extractor", type=str, default="rmvpe", choices=["rmvpe", "dio", "harvest", "crepe", "fcpe"])
    parser.add_argument("--f0_min", type=float, default=50.0)
    parser.add_argument("--f0_max", type=float, default=1200.0)
    parser.add_argument("--response_threshold_db", type=float, default=-60.0)
    parser.add_argument("--ref_max_sec", type=float, default=10.0)

    parser.add_argument("--stream", action="store_true")
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

    parser.add_argument("--gain_mode", type=str, default="off", choices=["off", "match_src_rms"])
    parser.add_argument("--gain_target_delta_db", type=float, default=10.0)
    parser.add_argument("--gain_max_boost_db", type=float, default=18.0)
    parser.add_argument("--gain_smoothing", type=float, default=0.0)

    parser.add_argument("--mask_mode", type=str, default="off", choices=["off", "rms"])
    parser.add_argument("--mask_db", type=float, default=-50.0)
    parser.add_argument("--mask_frame_ms", type=float, default=10.0)
    parser.add_argument("--mask_smooth_ms", type=float, default=10.0)
    parser.add_argument("--peak_limit", type=float, default=0.99)

    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip pairs whose output wav already exists.",
    )
    args = parser.parse_args(argv)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    mnpsvc_dir = os.path.abspath(str(args.mnpsvc_dir))
    if not os.path.isdir(mnpsvc_dir):
        raise FileNotFoundError(f"mnpsvc_dir not found: {mnpsvc_dir}")

    model_path = str(args.model_path).strip()
    if not model_path:
        model_path = os.path.join(mnpsvc_dir, "models", "pretrained", "mnp-svc", "vctk-full", "pytorch_model.bin")
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")

    _prefer_mnpsvc_modules(mnpsvc_dir)
    from modules.extractors import F0Extractor, SpeakerEmbedEncoder, UnitsEncoder, VolumeExtractor  # type: ignore[import-not-found]
    from modules.extractors.common import upsample  # type: ignore[import-not-found]
    from modules.vocoder import load_model  # type: ignore[import-not-found]

    _set_determinism(int(args.seed))
    model, cfg, _spk_info = load_model(model_path, device=str(device))

    model_sr = int(cfg.data.sampling_rate)
    block_size = int(cfg.data.block_size)
    if model_sr <= 0 or block_size <= 0:
        raise ValueError(f"Invalid model config: sampling_rate={model_sr} block_size={block_size}")

    units_encoder = UnitsEncoder(
        cfg.data.encoder,
        cfg.data.encoder_ckpt,
        cfg.data.encoder_sample_rate,
        cfg.data.encoder_hop_size,
        skip_frames=0 if cfg.data.get("units_skip_frames") is None else int(cfg.data.units_skip_frames),
        extract_layers=cfg.model.units_layers,
        device=str(device),
    )
    f0_extractor = F0Extractor(
        str(args.pitch_extractor),
        model_sr,
        block_size,
        float(args.f0_min),
        float(args.f0_max),
    )
    volume_extractor = VolumeExtractor(block_size, 1 if cfg.data.volume_window_size is None else int(cfg.data.volume_window_size))

    spk_encoder_ckpt = os.path.join(mnpsvc_dir, "models", "pretrained", "pyannote.audio", "wespeaker-voxceleb-resnet34-LM")
    spk_encoder = SpeakerEmbedEncoder(
        cfg.data.spk_embed_encoder,
        spk_encoder_ckpt,
        encoder_sample_rate=int(cfg.data.spk_embed_encoder_sample_rate),
        device=str(device),
    )

    manifest = load_vc_playlist_manifest(args.manifest).resolve_paths(args.manifest)
    sources = manifest.sources_by_id()
    targets = manifest.targets_by_id()
    pairs = list(manifest.pairs)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]

    out_root = Path(args.out_dir)
    wav_dir = out_root / "wavs"
    meta_dir = out_root / "meta"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    spk_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for pair_idx, pair in enumerate(pairs):
        s = sources[pair.source_id]
        t = targets[pair.target_id]
        cid = _case_id(pair.source_id, pair.target_id)

        out_wav = wav_dir / f"{cid}.wav"
        out_meta = meta_dir / f"{cid}.json"
        if bool(args.resume) and out_wav.exists() and out_meta.exists():
            continue

        ref_wav, ref_sr = _load_mono(t.wav_path)
        src_wav, src_sr = _load_mono(s.wav_path)
        ref_rs = _resample_if_needed(ref_wav, ref_sr, model_sr)
        src_rs = _resample_if_needed(src_wav, src_sr, model_sr)
        ref_rs = _trim_to_max_sec(ref_rs, sample_rate=model_sr, max_sec=float(args.ref_max_sec))

        if pair.target_id in spk_cache:
            spk_id, spk_mix = spk_cache[pair.target_id]
        else:
            ref_t = torch.from_numpy(ref_rs.reshape(1, -1)).float().to(device)
            spk_embed = spk_encoder.encode(ref_t, model_sr).detach().float().to(device).reshape(1, -1)
            spk_id = spk_embed.unsqueeze(0)  # (1, 1, C)
            spk_mix = torch.tensor([[[1.0]]], dtype=torch.float32, device=device)
            spk_cache[pair.target_id] = (spk_id, spk_mix)

        timings: list[float] = []
        delay_samples = 0
        warmup_hops = 0

        if not args.stream:
            t0 = time.time()
            out = _infer_mnpsvc(
                model=model,
                units_encoder=units_encoder,
                f0_extractor=f0_extractor,
                volume_extractor=volume_extractor,
                upsample=upsample,
                src_wav=src_rs,
                sample_rate=model_sr,
                hop_size=block_size,
                device=device,
                response_threshold_db=float(args.response_threshold_db),
                spk_id=spk_id,
                spk_mix=spk_mix,
            )
            timings.append(time.time() - t0)
            out = normalize_length(out, len(src_rs), align="start")
            sf.write(str(out_wav), out, model_sr)
        else:
            in_sr = model_sr
            out_sr = model_sr

            window_in = int(round(float(args.window_ms) / 1000.0 * float(in_sr)))
            hop_in = int(round(float(args.hop_ms) / 1000.0 * float(in_sr)))
            if window_in <= 0 or hop_in <= 0:
                raise ValueError("window_ms and hop_ms must be > 0")
            if hop_in > window_in:
                raise ValueError("hop_ms must be <= window_ms")

            window_out = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
            hop_out = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
            fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))

            ring = AudioRingBuffer(window_in)
            prev_tail: Optional[np.ndarray] = None

            drop_warmup_hops = bool(args.drop_warmup_hops)
            outs: list[np.ndarray] = []
            window_count = 0

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

            hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6)))
            hangover_left = 0
            gain_db_state = 0.0

            for start in range(0, len(src_rs), hop_in):
                hop = src_rs[start : start + hop_in]
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
                    webrtc_sr = in_sr if in_sr in (8000, 16000, 32000, 48000) else 16000
                    webrtc_segment = (
                        vad_segment
                        if webrtc_sr == in_sr
                        else _resample_if_needed(vad_segment, src_sr=in_sr, dst_sr=webrtc_sr)
                    )
                    webrtc_voiced = is_voiced_webrtcvad(
                        webrtc_segment,
                        sample_rate=webrtc_sr,
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
                    out_window = _infer_mnpsvc(
                        model=model,
                        units_encoder=units_encoder,
                        f0_extractor=f0_extractor,
                        volume_extractor=volume_extractor,
                        upsample=upsample,
                        src_wav=window,
                        sample_rate=model_sr,
                        hop_size=block_size,
                        device=device,
                        response_threshold_db=float(args.response_threshold_db),
                        spk_id=spk_id,
                        spk_mix=spk_mix,
                    )
                    timings.append(time.time() - t0)

                    out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))  # type: ignore[arg-type]
                    out_hop = out_window[
                        emit_start_out : emit_start_out + hop_out
                    ].astype(np.float32, copy=False)

                if voiced and str(args.gain_mode) == "match_src_rms":
                    alpha = float(np.clip(float(args.gain_smoothing), 0.0, 0.999))
                    src_db = rms_db(vad_segment, eps=1e-9)
                    out_db = rms_db(out_hop, eps=1e-9)
                    desired_boost_db = (src_db - out_db) - float(args.gain_target_delta_db)
                    desired_boost_db = float(
                        np.clip(desired_boost_db, 0.0, float(args.gain_max_boost_db))
                    )
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

            if bool(args.drop_warmup_hops):
                delay_samples = int(
                    int(warmup_hops) * int(hop_out)
                    + (int(hop_out) - int(window_out))
                    + int(emit_start_out)
                )
            else:
                delay_samples = int(emit_start_out)

        report = {
            "config": {
                "mnpsvc_dir": str(mnpsvc_dir),
                "model_path": str(model_path),
                "device": str(device),
                "seed": int(args.seed),
                "pitch_extractor": str(args.pitch_extractor),
                "f0_min": float(args.f0_min),
                "f0_max": float(args.f0_max),
                "response_threshold_db": float(args.response_threshold_db),
                "ref_max_sec": float(args.ref_max_sec),
                "stream": (
                    {
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
                    else None
                ),
            },
            "stats": {
                "delay_samples": int(delay_samples),
                "warmup_hops": int(warmup_hops) if bool(args.stream) else 0,
                "mean_window_sec": float(np.mean(np.asarray(timings, dtype=np.float64))) if timings else 0.0,
                "p95_window_sec": float(np.percentile(np.asarray(timings, dtype=np.float64), 95))
                if len(timings) >= 2
                else (float(timings[0]) if timings else 0.0),
                "windows": int(len(timings)),
            },
        }
        out_meta.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"[mnpsvc_playlist_convert] {pair_idx+1}/{len(pairs)} wrote {out_wav.name}", flush=True)

    print(f"[mnpsvc_playlist_convert] Done. Wrote: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.streaming_utils import (
    AudioRingBuffer,
    apply_peak_limiter,
    build_rms_mask,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
    rms_db,
    smooth_boundary_inplace,
)


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    fade_ms: int
    normalize_align: str
    emit_align: str
    drop_warmup_hops: bool
    vad_mode: str
    vad_db: float
    vad_frame_ms: float
    vad_hangover_ms: float
    vad_webrtc_aggressiveness: int
    vad_webrtc_frame_ms: int
    vad_webrtc_min_voiced_ratio: float
    gain_mode: str
    gain_target_delta_db: float
    gain_max_boost_db: float
    gain_smoothing: float
    mask_mode: str
    mask_db: float
    mask_frame_ms: float
    mask_smooth_ms: float
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    chatterbox_dir: str
    device: str
    seed: int
    cfm_timesteps: int
    watermark: bool
    stream: Optional[StreamConfig] = None


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


def _import_chatterbox_vc(chatterbox_dir: str):
    """Import chatterbox.vc without executing chatterbox/__init__.py.

    The upstream package imports TTS modules (transformers/diffusers/spacy) in __init__.py.
    For VC-only evaluation, we bypass that to keep the env lightweight and avoid torch pinning.
    """

    root = Path(chatterbox_dir).resolve()
    pkg_root = root / "src" / "chatterbox"
    if not pkg_root.is_dir():
        raise FileNotFoundError(f"Expected chatterbox sources under: {pkg_root}")

    # Replace any existing installed chatterbox module (if present).
    sys.modules.pop("chatterbox", None)
    sys.modules.pop("chatterbox.vc", None)

    pkg = types.ModuleType("chatterbox")
    pkg.__path__ = [str(pkg_root)]  # type: ignore[attr-defined]
    sys.modules["chatterbox"] = pkg

    vc_path = pkg_root / "vc.py"
    spec = importlib.util.spec_from_file_location("chatterbox.vc", str(vc_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load chatterbox.vc spec from: {vc_path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "chatterbox"
    sys.modules["chatterbox.vc"] = mod
    spec.loader.exec_module(mod)

    return mod.ChatterboxVC  # type: ignore[attr-defined]


@torch.inference_mode()
def _infer_chatterbox(
    *,
    model,
    src_wav_16k: np.ndarray,
    cfm_timesteps: int,
    watermark: bool,
    seed: int,
) -> tuple[np.ndarray, int]:
    _set_determinism(seed)
    x = torch.from_numpy(np.asarray(src_wav_16k, dtype=np.float32).reshape(-1)).to(model.device)
    x = x[None, :]

    s3_tokens, _ = model.s3gen.tokenizer(x)
    wav, _ = model.s3gen.inference(
        speech_tokens=s3_tokens,
        ref_dict=model.ref_dict,
        n_cfm_timesteps=int(cfm_timesteps),
    )
    wav_np = wav.squeeze(0).detach().cpu().float().numpy()

    if watermark:
        wav_np = model.watermarker.apply_watermark(wav_np, sample_rate=int(model.sr))

    return np.asarray(wav_np, dtype=np.float32).reshape(-1), int(model.sr)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ChatterboxVC one-shot VC runner (offline or streaming simulation)."
    )
    parser.add_argument("--chatterbox_dir", type=str, required=True, help="Path to chatterbox repo checkout.")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")
    parser.add_argument("--cfm_timesteps", type=int, default=10, help="S3Gen CFM timesteps (quality/speed).")
    parser.add_argument(
        "--watermark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the upstream PerTh watermark (default: true).",
    )

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument(
        "--window_ms",
        type=int,
        default=800,
        help="Streaming window size in ms. Use 0 to run one full-utterance window (equivalence test).",
    )
    parser.add_argument(
        "--hop_ms",
        type=int,
        default=400,
        help="Streaming hop size in ms. Use 0 to run one full-utterance hop (equivalence test).",
    )
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--emit_align", type=str, default="center", choices=["start", "center", "end"])
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop output until the first full window is available (recommended for eval).",
    )
    parser.add_argument("--vad_mode", type=str, default="webrtc", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)
    parser.add_argument(
        "--gain_mode",
        type=str,
        default="match_src_rms",
        choices=["off", "match_src_rms"],
        help="Voiced-only gain control to reduce low-level dropouts (default: match_src_rms).",
    )
    parser.add_argument(
        "--gain_target_delta_db",
        type=float,
        default=10.0,
        help="When gain_mode=match_src_rms, aim for output RMS ≈ (input RMS - gain_target_delta_db).",
    )
    parser.add_argument(
        "--gain_max_boost_db",
        type=float,
        default=18.0,
        help="When gain_mode=match_src_rms, cap the per-hop boost in dB (voiced only).",
    )
    parser.add_argument(
        "--gain_smoothing",
        type=float,
        default=0.9,
        help="EMA smoothing factor for gain (0=no smoothing, 0.9=slow).",
    )
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="rms",
        choices=["off", "rms"],
        help="Optional wrapper-level masking to suppress output during input silence (default: rms).",
    )
    parser.add_argument(
        "--mask_db",
        type=float,
        default=-50.0,
        help="When mask_mode=rms, frames below this dBFS threshold are suppressed.",
    )
    parser.add_argument(
        "--mask_frame_ms",
        type=float,
        default=10.0,
        help="When mask_mode=rms, frame size for RMS masking.",
    )
    parser.add_argument(
        "--mask_smooth_ms",
        type=float,
        default=10.0,
        help="When mask_mode=rms, smooth the mask with a moving average window (ms).",
    )
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    chatterbox_dir = os.path.abspath(str(args.chatterbox_dir))
    if not os.path.isdir(chatterbox_dir):
        raise FileNotFoundError(f"chatterbox_dir not found: {chatterbox_dir}")

    ChatterboxVC = _import_chatterbox_vc(chatterbox_dir)

    # ChatterboxVC expects device as a string.
    model = ChatterboxVC.from_pretrained(str(device))
    model.set_target_voice(str(args.ref))

    # Chatterbox tokenizes at 16kHz and synthesizes at 24kHz.
    in_sr = 16000

    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)
    # We do not feed reference audio here (ChatterboxVC prepares it internally), but keep SR for metadata.
    _ = _resample_if_needed(ref_wav, ref_sr, int(model.sr))
    src_16k = _resample_if_needed(src_wav, src_sr, in_sr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0
    warmup_hops = 0

    if not args.stream:
        t0 = time.time()
        out, out_sr = _infer_chatterbox(
            model=model,
            src_wav_16k=src_16k,
            cfm_timesteps=int(args.cfm_timesteps),
            watermark=bool(args.watermark),
            seed=int(args.seed),
        )
        timings.append(time.time() - t0)
        sf.write(args.out, out, out_sr)
    else:
        # window_ms/hop_ms allow 0 as a sentinel meaning "use the full utterance length"
        # (useful for streaming-vs-offline equivalence tests without padding).
        if float(args.window_ms) > 0:
            window_in = int(round(float(args.window_ms) / 1000.0 * float(in_sr)))
        else:
            window_in = int(len(src_16k))
        if float(args.hop_ms) > 0:
            hop_in = int(round(float(args.hop_ms) / 1000.0 * float(in_sr)))
        else:
            hop_in = int(window_in)
        if window_in <= 0 or hop_in <= 0:
            raise ValueError("window_ms and hop_ms must be > 0 (or 0 to use full utterance)")
        if hop_in > window_in:
            raise ValueError("hop_ms must be <= window_ms")

        out_sr = int(model.sr)
        if float(args.window_ms) > 0:
            window_out = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
        else:
            window_out = int(round(float(window_in) * float(out_sr) / float(in_sr)))
        if float(args.hop_ms) > 0:
            hop_out = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
        else:
            hop_out = int(round(float(hop_in) * float(out_sr) / float(in_sr)))
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
        prev_last: Optional[float] = None

        drop_warmup_hops = bool(args.drop_warmup_hops)
        outs: list[np.ndarray] = []
        window_count = 0

        hop_ms_eff = float(args.hop_ms) if float(args.hop_ms) > 0 else (1000.0 * float(hop_in) / float(in_sr))
        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(hop_ms_eff, 1e-6)))
        hangover_left = 0
        gain_db_state = 0.0

        for start in range(0, len(src_16k), hop_in):
            hop = src_16k[start : start + hop_in]
            if len(hop) < hop_in:
                hop = np.pad(hop, (0, hop_in - len(hop)), mode="constant")
            ring.write(hop)

            if ring.size < window_in:
                warmup_hops += 1
                prev_last = 0.0
                if not drop_warmup_hops:
                    outs.append(np.zeros(hop_out, dtype=np.float32))
                continue

            window = ring.read_last(window_in)

            # IMPORTANT: VAD should operate on the **region we emit**, not the current hop.
            # Otherwise emit_align=center/end can cause "silence leakage" (infer on silent frames)
            # and/or voiced dropouts (skip inference when emit region still contains speech).
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
                # Combine WebRTC VAD with an RMS silence gate to avoid false positives
                # turning into loud "silence noise" in the vocoder.
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

            # Hangover is only allowed to override when we're not truly silent by RMS.
            if not voiced and hangover_left > 0 and (not silent_rms):
                voiced = True
                hangover_left -= 1
            elif voiced:
                hangover_left = hangover_hops

            if not voiced:
                out_hop = np.zeros(hop_out, dtype=np.float32)
            else:
                t0 = time.time()
                out_window, got_sr = _infer_chatterbox(
                    model=model,
                    src_wav_16k=window,
                    cfm_timesteps=int(args.cfm_timesteps),
                    watermark=bool(args.watermark),
                    seed=int(args.seed) + int(window_count),
                )
                timings.append(time.time() - t0)
                if int(got_sr) != out_sr:
                    raise RuntimeError(f"Unexpected output sr={got_sr} (expected {out_sr})")

                out_window = normalize_length(
                    out_window, window_out, align=str(args.normalize_align)  # type: ignore[arg-type]
                )
                out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(
                    np.float32, copy=False
                )

            gain_mode = str(args.gain_mode)
            if voiced and gain_mode == "match_src_rms":
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

            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
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
            prev_last = float(out_hop[-1]) if len(out_hop) else prev_last

            outs.append(out_hop)
            window_count += 1

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        sf.write(args.out, out, out_sr)

        # Align output timeline to source for downstream scoring.
        # See FreeVC notes: when window_ms is not a multiple of hop_ms, warmup hops shift the
        # first emitted segment in time if we drop warmup hops.
        if bool(args.drop_warmup_hops):
            delay_samples = int(
                int(warmup_hops) * int(hop_out)
                + (int(hop_out) - int(window_out))
                + int(emit_start_out)
            )
        else:
            delay_samples = int(emit_start_out)

    if args.meta_json:
        meta_p = Path(args.meta_json)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        stream_cfg = (
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                fade_ms=int(args.fade_ms),
                normalize_align=str(args.normalize_align),
                emit_align=str(args.emit_align),
                drop_warmup_hops=bool(args.drop_warmup_hops),
                vad_mode=str(args.vad_mode),
                vad_db=float(args.vad_db),
                vad_frame_ms=float(args.vad_frame_ms),
                vad_hangover_ms=float(args.vad_hangover_ms),
                vad_webrtc_aggressiveness=int(args.vad_webrtc_aggressiveness),
                vad_webrtc_frame_ms=int(args.vad_webrtc_frame_ms),
                vad_webrtc_min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                gain_mode=str(args.gain_mode),
                gain_target_delta_db=float(args.gain_target_delta_db),
                gain_max_boost_db=float(args.gain_max_boost_db),
                gain_smoothing=float(args.gain_smoothing),
                mask_mode=str(args.mask_mode),
                mask_db=float(args.mask_db),
                mask_frame_ms=float(args.mask_frame_ms),
                mask_smooth_ms=float(args.mask_smooth_ms),
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )
        cfg = RunConfig(
            chatterbox_dir=str(chatterbox_dir),
            device=str(device),
            seed=int(args.seed),
            cfm_timesteps=int(args.cfm_timesteps),
            watermark=bool(args.watermark),
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
        }
        meta_p.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2))

    print(f"Wrote: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

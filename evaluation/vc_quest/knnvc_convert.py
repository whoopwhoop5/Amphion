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
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.streaming_utils import (
    AudioRingBuffer,
    apply_peak_limiter,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
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
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    knnvc_dir: str
    prematched: bool
    topk: int
    tgt_loudness_db: Optional[float]
    ref_trim_db: float
    ref_vad_trigger_level: float
    device: str
    seed: int
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


def _trim_ref(wav: np.ndarray, *, sample_rate: int, top_db: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == 0 or top_db <= 0:
        return wav
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=float(top_db))
    return np.asarray(trimmed, dtype=np.float32).reshape(-1)


def _parse_optional_float(x: str) -> Optional[float]:
    x = str(x).strip().lower()
    if x in ("", "none", "null", "off"):
        return None
    return float(x)


@torch.no_grad()
def _infer_knnvc(
    *,
    knn_vc,
    matching_set: torch.Tensor,
    src_wav_16k: np.ndarray,
    device: torch.device,
    topk: int,
    tgt_loudness_db: Optional[float],
    seed: int,
) -> np.ndarray:
    _set_determinism(seed)
    src_t = torch.from_numpy(np.asarray(src_wav_16k, dtype=np.float32).reshape(1, -1)).to(device)
    query_seq = knn_vc.get_features(src_t, vad_trigger_level=0)
    out_t = knn_vc.match(
        query_seq,
        matching_set,
        topk=int(topk),
        tgt_loudness_db=tgt_loudness_db,
        device=str(device),
    )
    return np.asarray(out_t.detach().cpu().float().numpy(), dtype=np.float32).reshape(-1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="kNN-VC runner (offline or streaming simulation).")
    parser.add_argument(
        "--knnvc_dir",
        type=str,
        required=True,
        help="Path to knn-vc repo checkout (used via torch.hub load, source=local).",
    )
    parser.add_argument(
        "--prematched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use prematched HiFiGAN weights (recommended).",
    )
    parser.add_argument("--topk", type=int, default=4, help="k in kNN matching.")
    parser.add_argument(
        "--tgt_loudness_db",
        type=str,
        default="none",
        help="Output loudness normalization in dBFS (e.g. -16). Use 'none' to disable (recommended for streaming).",
    )

    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--ref_trim_db", type=float, default=20.0, help="librosa.effects.trim top_db for reference")
    parser.add_argument(
        "--ref_vad_trigger_level",
        type=float,
        default=7.0,
        help="torchaudio.transforms.Vad trigger_level for matching set (0 disables).",
    )

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument("--window_ms", type=int, default=600)
    parser.add_argument("--hop_ms", type=int, default=600)
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
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    _set_determinism(int(args.seed))

    knnvc_dir = os.path.abspath(str(args.knnvc_dir))
    if not os.path.isdir(knnvc_dir):
        raise FileNotFoundError(f"knnvc_dir not found: {knnvc_dir}")

    # The torch.hub entrypoint is stable and avoids us re-implementing internal model wiring.
    knn_vc = torch.hub.load(
        knnvc_dir,
        "knn_vc",
        source="local",
        pretrained=True,
        prematched=bool(args.prematched),
        device=str(device),
    )

    in_sr = 16000
    out_sr = 16000

    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)
    ref_16k = _resample_if_needed(ref_wav, ref_sr, in_sr)
    src_16k = _resample_if_needed(src_wav, src_sr, in_sr)
    ref_16k = _trim_ref(ref_16k, sample_rate=in_sr, top_db=float(args.ref_trim_db))

    ref_t = torch.from_numpy(ref_16k.reshape(1, -1)).to(device)
    matching_set = knn_vc.get_matching_set(
        [ref_t],
        vad_trigger_level=float(args.ref_vad_trigger_level),
    ).to(device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tgt_loudness_db = _parse_optional_float(str(args.tgt_loudness_db))
    timings: list[float] = []
    delay_samples = 0

    if not bool(args.stream):
        t0 = time.time()
        out = _infer_knnvc(
            knn_vc=knn_vc,
            matching_set=matching_set,
            src_wav_16k=src_16k,
            device=device,
            topk=int(args.topk),
            tgt_loudness_db=tgt_loudness_db,
            seed=int(args.seed),
        )
        timings.append(time.time() - t0)
        sf.write(args.out, out, out_sr)
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

        ring = AudioRingBuffer(window_in)
        prev_tail: Optional[np.ndarray] = None

        drop_warmup_hops = bool(args.drop_warmup_hops)
        outs: list[np.ndarray] = []
        warmup_hops = 0
        window_count = 0

        if args.emit_align == "start":
            emit_start_out = 0
        elif args.emit_align == "center":
            emit_start_out = max(0, (window_out - hop_out) // 2)
        elif args.emit_align == "end":
            emit_start_out = max(0, window_out - hop_out)
        else:
            raise ValueError(f"Unknown emit_align: {args.emit_align}")

        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6)))
        hangover_left = 0

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

            vad_mode = str(args.vad_mode)
            if vad_mode == "off":
                voiced = True
            elif vad_mode == "rms":
                voiced = not (
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        hop,
                        sample_rate=in_sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )
            elif vad_mode == "webrtc":
                voiced = is_voiced_webrtcvad(
                    hop,
                    sample_rate=in_sr,
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
                out_hop = np.zeros(hop_out, dtype=np.float32)
            else:
                t0 = time.time()
                out_window = _infer_knnvc(
                    knn_vc=knn_vc,
                    matching_set=matching_set,
                    src_wav_16k=window,
                    device=device,
                    topk=int(args.topk),
                    tgt_loudness_db=tgt_loudness_db,
                    seed=int(args.seed) + int(window_count),
                )
                timings.append(time.time() - t0)

                out_window = normalize_length(
                    out_window, window_out, align=str(args.normalize_align)  # type: ignore[arg-type]
                )
                out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

            out_hop = crossfade_prefix_inplace(out_hop, prev_tail, fade_out)
            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
            if fade_out > 0:
                prev_tail = out_hop[-fade_out:].astype(np.float32, copy=True)

            outs.append(out_hop)
            window_count += 1

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        sf.write(args.out, out, out_sr)
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
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )
        cfg = RunConfig(
            knnvc_dir=str(knnvc_dir),
            prematched=bool(args.prematched),
            topk=int(args.topk),
            tgt_loudness_db=tgt_loudness_db,
            ref_trim_db=float(args.ref_trim_db),
            ref_vad_trigger_level=float(args.ref_vad_trigger_level),
            device=str(device),
            seed=int(args.seed),
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


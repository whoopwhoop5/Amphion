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
    build_rms_mask,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
    rms_db,
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
    variant: str
    device: str
    seed: int
    n_timesteps: int
    cfg: float
    ref_max_sec: float
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Metis VC runner (offline or streaming simulation)."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="metis_vc",
        choices=["metis_vc", "metis_omni"],
        help="Which Metis checkpoint/config to use.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed base (per-window adds index).",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=20,
        help="Metis Stage-1 diffusion timesteps (semantic token generation).",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
        help="Metis Stage-1 classifier-free guidance scale.",
    )
    parser.add_argument(
        "--ref_max_sec",
        type=float,
        default=10.0,
        help="Trim reference audio to this many seconds (0 disables).",
    )

    parser.add_argument(
        "--ref", type=str, required=True, help="Target/reference voice wav"
    )
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path")

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
        "--emit_align", type=str, default="end", choices=["start", "center", "end"]
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
    parser.add_argument(
        "--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30]
    )
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
    args = parser.parse_args(argv)

    device = (
        str(args.device).strip()
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    # Lazy imports to keep help/CLI fast.
    from huggingface_hub import snapshot_download

    from models.tts.metis.metis import Metis
    from utils.util import load_config

    if str(args.variant) == "metis_vc":
        cfg_path = repo_root / "models" / "tts" / "metis" / "config" / "ft.json"
        base_ckpt_rel = "metis_base/model.safetensors"
        lora_ckpt_rel = "metis_vc/metis_vc_lora_16.safetensors"
        adapter_ckpt_rel = "metis_vc/metis_vc_lora_16_adapter.safetensors"
        allow_patterns = [base_ckpt_rel, lora_ckpt_rel, adapter_ckpt_rel]
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

    if str(args.variant) == "metis_vc":
        base_ckpt_path = str(Path(ckpt_dir) / base_ckpt_rel)
        lora_ckpt_path = str(Path(ckpt_dir) / lora_ckpt_rel)
        adapter_ckpt_path = str(Path(ckpt_dir) / adapter_ckpt_rel)
        # The public Metis repo provides LoRA + adapter weights for VC (r=16).
        metis_cfg.model.t2s_model.use_lora = True
        metis_cfg.model.t2s_model.lora_r = 16
        metis_cfg.model.t2s_model.lora_alpha = 16
        metis = Metis(
            ckpt_path=None,
            base_ckpt_path=base_ckpt_path,
            lora_ckpt_path=lora_ckpt_path,
            adapter_ckpt_path=adapter_ckpt_path,
            cfg=metis_cfg,
            device=str(device),
            model_type=str(model_type_init),
        )
    else:
        ckpt_path = str(Path(ckpt_dir) / ckpt_rel)
        metis = Metis(
            ckpt_path=ckpt_path,
            cfg=metis_cfg,
            device=str(device),
            model_type=str(model_type_init),
        )

    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)
    ref_wav = _trim_wav(ref_wav, ref_sr, float(args.ref_max_sec))

    in_sr = 16000
    out_sr = 24000

    prompt_16k = _resample_if_needed(ref_wav, ref_sr, in_sr)
    prompt_24k = _resample_if_needed(ref_wav, ref_sr, out_sr)
    prompt_semantic_code, _, prompt_acoustic_code = metis.audio_tokenizer(
        speech_16k=prompt_16k, speech=prompt_24k
    )

    src_16k = _resample_if_needed(src_wav, src_sr, in_sr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0
    warmup_hops = 0

    if not bool(args.stream):
        t0 = time.time()
        out, out_sr2 = _infer_metis(
            metis=metis,
            src_wav_16k=src_16k,
            prompt_wav_16k=prompt_16k,
            prompt_semantic_code=prompt_semantic_code,
            prompt_acoustic_code=prompt_acoustic_code,
            n_timesteps=int(args.n_timesteps),
            cfg=float(args.cfg),
            seed=int(args.seed),
        )
        timings.append(time.time() - t0)
        sf.write(args.out, out, int(out_sr2))
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
                    seed=int(args.seed) + int(window_count),
                )
                timings.append(time.time() - t0)
                out_window = normalize_length(
                    out_window, window_out, align=str(args.normalize_align)
                )
                out_hop = out_window[
                    emit_start_out : emit_start_out + hop_out
                ].astype(np.float32, copy=False)

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
        sf.write(args.out, out, out_sr)

        # Align output timeline to source for downstream scoring.
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
            variant=str(args.variant),
            device=str(device),
            seed=int(args.seed),
            n_timesteps=int(args.n_timesteps),
            cfg=float(args.cfg),
            ref_max_sec=float(args.ref_max_sec),
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

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import math
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
    vc_model: str
    vocoder_model: str
    speaker_model: str
    device: str
    seed: int
    ref_trim_db: float
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


def _ensure_sinusoidal_pos_encoding_len(
    module,
    *,
    min_len: int,
) -> None:
    """Work around SpeechT5 max-length positional encoding limits.

    Some SpeechT5 checkpoints ship with a relatively small sinusoidal buffer which can
    cause `generate_speech()` to crash for longer sequences (off-by-one at the boundary).
    """

    if min_len <= 0:
        return
    pe = getattr(module, "pe", None)
    dim = int(getattr(module, "dim", 0))
    if pe is None or dim <= 0:
        return
    if pe.ndim != 3 or pe.shape[0] != 1:
        return

    cur_len = int(pe.shape[1])
    if cur_len >= int(min_len):
        return

    device = pe.device
    dtype = pe.dtype
    max_len = int(min_len)

    new_pe = torch.zeros((max_len, dim), device=device, dtype=dtype)
    position = torch.arange(0, max_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        (torch.arange(0, dim, 2, device=device, dtype=torch.float32) * -(math.log(10000.0) / dim))
    )
    new_pe[:, 0::2] = torch.sin(position * div_term)
    new_pe[:, 1::2] = torch.cos(position * div_term)
    new_pe = new_pe.unsqueeze(0).to(dtype)

    # `pe` is a (non-persistent) registered buffer in transformers; overwriting is OK.
    module.pe = new_pe


@torch.no_grad()
def _speaker_embedding_wavlm_xvector(
    *,
    feature_extractor,
    speaker_model,
    ref_wav_16k: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    inputs = feature_extractor(ref_wav_16k, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = speaker_model(**inputs)
    emb = out.embeddings
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb


@torch.no_grad()
def _infer_speecht5(
    *,
    processor,
    model,
    vocoder,
    speaker_embeddings: torch.Tensor,
    src_wav_16k: np.ndarray,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    _set_determinism(seed)
    inputs = processor(audio=src_wav_16k, sampling_rate=16000, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    wav_t = model.generate_speech(
        input_values=input_values,
        speaker_embeddings=speaker_embeddings,
        attention_mask=attention_mask,
        vocoder=vocoder,
    )
    wav = wav_t.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SpeechT5 VC runner (offline or streaming simulation).")
    parser.add_argument("--vc_model", type=str, default="microsoft/speecht5_vc")
    parser.add_argument("--vocoder_model", type=str, default="microsoft/speecht5_hifigan")
    parser.add_argument("--speaker_model", type=str, default="microsoft/wavlm-base-plus-sv")

    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--ref_trim_db", type=float, default=20.0, help="librosa.effects.trim top_db for reference")
    parser.add_argument(
        "--pos_max_len",
        type=int,
        default=20000,
        help="Ensure SpeechT5 sinusoidal positional encoding buffer is at least this long (workaround).",
    )

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument("--window_ms", type=int, default=600)
    parser.add_argument("--hop_ms", type=int, default=300)
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

    from transformers import (
        AutoFeatureExtractor,
        SpeechT5ForSpeechToSpeech,
        SpeechT5HifiGan,
        SpeechT5Processor,
        WavLMForXVector,
    )

    processor = SpeechT5Processor.from_pretrained(str(args.vc_model))
    model = SpeechT5ForSpeechToSpeech.from_pretrained(str(args.vc_model)).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(str(args.vocoder_model)).to(device).eval()

    try:
        _ensure_sinusoidal_pos_encoding_len(
            model.speecht5.decoder.prenet.pos_sinusoidal_embed,
            min_len=int(args.pos_max_len),
        )
    except Exception:
        # Best-effort: the underlying API may differ across transformer versions/checkpoints.
        pass

    spk_fe = AutoFeatureExtractor.from_pretrained(str(args.speaker_model))
    spk_model = WavLMForXVector.from_pretrained(str(args.speaker_model)).to(device).eval()

    in_sr = 16000
    out_sr = 16000

    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)
    ref_16k = _resample_if_needed(ref_wav, ref_sr, in_sr)
    src_16k = _resample_if_needed(src_wav, src_sr, in_sr)
    ref_16k = _trim_ref(ref_16k, sample_rate=in_sr, top_db=float(args.ref_trim_db))

    speaker_embeddings = _speaker_embedding_wavlm_xvector(
        feature_extractor=spk_fe,
        speaker_model=spk_model,
        ref_wav_16k=ref_16k,
        device=device,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0

    if not bool(args.stream):
        t0 = time.time()
        out = _infer_speecht5(
            processor=processor,
            model=model,
            vocoder=vocoder,
            speaker_embeddings=speaker_embeddings,
            src_wav_16k=src_16k,
            device=device,
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
                out_window = _infer_speecht5(
                    processor=processor,
                    model=model,
                    vocoder=vocoder,
                    speaker_embeddings=speaker_embeddings,
                    src_wav_16k=window,
                    device=device,
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
            vc_model=str(args.vc_model),
            vocoder_model=str(args.vocoder_model),
            speaker_model=str(args.speaker_model),
            device=str(device),
            seed=int(args.seed),
            ref_trim_db=float(args.ref_trim_db),
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

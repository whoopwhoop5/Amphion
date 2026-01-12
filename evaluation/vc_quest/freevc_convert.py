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
    freevc_dir: str
    variant: str
    device: str
    seed: int
    wavlm_model: str
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

    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr).astype(
        np.float32, copy=False
    )


def _trim_ref(wav: np.ndarray, *, sample_rate: int, top_db: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == 0:
        return wav
    if top_db <= 0:
        return wav
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=float(top_db))
    return np.asarray(trimmed, dtype=np.float32).reshape(-1)


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


@torch.no_grad()
def _infer_freevc(
    *,
    net_g,
    hps,
    cmodel,
    smodel,
    mel_spectrogram_torch,
    src_wav_16k: np.ndarray,
    ref_wav_16k: np.ndarray,
    out_sample_rate: int,
    seed: int,
    device: torch.device,
    ref_trim_db: float,
) -> tuple[np.ndarray, int]:
    # Reference -> speaker conditioning.
    ref_wav_16k = _trim_ref(ref_wav_16k, sample_rate=16000, top_db=ref_trim_db)
    if bool(hps.model.use_spk):
        g_tgt = smodel.embed_utterance(ref_wav_16k)
        g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
        mel_tgt = None
    else:
        ref_t = torch.from_numpy(ref_wav_16k).unsqueeze(0).to(device)
        mel_tgt = mel_spectrogram_torch(
            ref_t,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        g_tgt = None

    # Source -> content.
    src_t = torch.from_numpy(src_wav_16k).unsqueeze(0).to(device)
    c = cmodel(src_t).last_hidden_state.transpose(1, 2).to(device)

    _set_determinism(seed)
    if g_tgt is not None:
        audio = net_g.infer(c, g=g_tgt)
    else:
        assert mel_tgt is not None
        audio = net_g.infer(c, mel=mel_tgt)

    audio_np = audio[0][0].detach().cpu().float().numpy()
    return np.asarray(audio_np, dtype=np.float32).reshape(-1), int(out_sample_rate)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="FreeVC one-shot VC runner (offline or streaming simulation)."
    )
    parser.add_argument(
        "--freevc_dir", type=str, required=True, help="Path to FreeVC repo checkout."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="freevc",
        choices=["freevc", "freevc-s", "freevc-24"],
        help="Which FreeVC checkpoint/config to use.",
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
    parser.add_argument("--wavlm_model", type=str, default="microsoft/wavlm-large")

    parser.add_argument(
        "--ref", type=str, required=True, help="Target/reference voice wav"
    )
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument(
        "--meta_json",
        type=str,
        default="",
        help="Optional JSON path to write run metadata",
    )
    parser.add_argument(
        "--ref_trim_db",
        type=float,
        default=20.0,
        help="librosa.effects.trim top_db for reference",
    )

    parser.add_argument(
        "--stream", action="store_true", help="Run window/hop streaming simulation."
    )
    parser.add_argument("--window_ms", type=int, default=600)
    parser.add_argument("--hop_ms", type=int, default=600)
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
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    # Prevent FreeVC's `utils.py` from setting global DEBUG logging (which makes numba extremely noisy).
    import logging

    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    freevc_dir = os.path.abspath(str(args.freevc_dir))
    if not os.path.isdir(freevc_dir):
        raise FileNotFoundError(f"freevc_dir not found: {freevc_dir}")

    # Ensure FreeVC's top-level modules (models.py, utils.py, etc.) shadow Amphion's similarly named packages.
    _add_sys_path_first(freevc_dir)

    import utils as freevc_utils  # type: ignore[import-not-found]
    from mel_processing import mel_spectrogram_torch  # type: ignore[import-not-found]
    from models import SynthesizerTrn  # type: ignore[import-not-found]
    from speaker_encoder.voice_encoder import SpeakerEncoder  # type: ignore[import-not-found]
    from transformers import WavLMModel  # type: ignore[import-not-found]

    variant = str(args.variant)
    ckpt_name = {
        "freevc": "freevc.pth",
        "freevc-s": "freevc-s.pth",
        "freevc-24": "freevc-24.pth",
    }[variant]
    cfg_name = {
        "freevc": "freevc.json",
        "freevc-s": "freevc-s.json",
        "freevc-24": "freevc-24.json",
    }[variant]

    ckpt_path = Path(freevc_dir) / "checkpoints" / ckpt_name
    cfg_path = Path(freevc_dir) / "configs" / cfg_name
    spk_ckpt = (
        Path(freevc_dir) / "speaker_encoder" / "ckpt" / "pretrained_bak_5805000.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing FreeVC checkpoint: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing FreeVC config: {cfg_path}")
    if not spk_ckpt.exists():
        raise FileNotFoundError(f"Missing speaker encoder ckpt: {spk_ckpt}")

    print(f"[freevc] Loading {variant} from {cfg_path} + {ckpt_path}", flush=True)
    hps = freevc_utils.get_hparams_from_file(str(cfg_path))

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    net_g.eval()
    freevc_utils.load_checkpoint(str(ckpt_path), net_g, None)

    smodel = SpeakerEncoder(str(spk_ckpt))

    cmodel = WavLMModel.from_pretrained(str(args.wavlm_model)).to(device)
    cmodel.eval()

    # FreeVC uses WavLM at 16kHz for content + speaker encoder.
    content_sr = 16000
    wavlm_hop = 320  # WavLM's conv feature extractor hop @16kHz (20ms).
    upsample_rates = list(getattr(hps.model, "upsample_rates", []))
    if not upsample_rates:
        raise ValueError(
            "Missing hps.model.upsample_rates; cannot infer output sample rate."
        )
    prod_upsample = int(np.prod(np.asarray(upsample_rates, dtype=np.int64)))
    out_sr = int(round(float(content_sr) / float(wavlm_hop) * float(prod_upsample)))
    if out_sr <= 0:
        raise ValueError(f"Invalid inferred output sample rate: {out_sr}")

    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)
    ref_16k = _resample_if_needed(ref_wav, ref_sr, content_sr)
    src_16k = _resample_if_needed(src_wav, src_sr, content_sr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0

    if not args.stream:
        t0 = time.time()
        out, out_sr = _infer_freevc(
            net_g=net_g,
            hps=hps,
            cmodel=cmodel,
            smodel=smodel,
            mel_spectrogram_torch=mel_spectrogram_torch,
            src_wav_16k=src_16k,
            ref_wav_16k=ref_16k,
            out_sample_rate=int(out_sr),
            seed=int(args.seed),
            device=device,
            ref_trim_db=float(args.ref_trim_db),
        )
        timings.append(time.time() - t0)
        sf.write(args.out, out, out_sr)
    else:
        in_sr = content_sr
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
            emit_start_in = 0
        elif args.emit_align == "center":
            emit_start_out = max(0, (window_out - hop_out) // 2)
            emit_start_in = max(0, (window_in - hop_in) // 2)
        elif args.emit_align == "end":
            emit_start_out = max(0, window_out - hop_out)
            emit_start_in = max(0, window_in - hop_in)
        else:
            raise ValueError(f"Unknown emit_align: {args.emit_align}")

        hangover_hops = int(
            np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6))
        )
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

            # IMPORTANT: VAD should operate on the region we emit (emit_align aware).
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
                out_window, got_sr = _infer_freevc(
                    net_g=net_g,
                    hps=hps,
                    cmodel=cmodel,
                    smodel=smodel,
                    mel_spectrogram_torch=mel_spectrogram_torch,
                    src_wav_16k=window,
                    ref_wav_16k=ref_16k,
                    out_sample_rate=int(out_sr),
                    seed=int(args.seed) + int(window_count),
                    device=device,
                    ref_trim_db=float(args.ref_trim_db),
                )
                timings.append(time.time() - t0)

                out_window = normalize_length(
                    out_window, window_out, align=str(args.normalize_align)
                )  # type: ignore[arg-type]
                out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(
                    np.float32, copy=False
                )

            out_hop = crossfade_prefix_inplace(out_hop, prev_tail, fade_out)
            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
            if fade_out > 0:
                prev_tail = out_hop[-fade_out:].astype(np.float32, copy=True)

            outs.append(out_hop)
            window_count += 1

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        sf.write(args.out, out, out_sr)

        # Align output timeline to source for downstream scoring.
        #
        # If we drop warmup hops (recommended), the output file starts at the first emitted
        # segment, which is offset by the number of warmup hops. The emitted segment itself
        # can start inside the window depending on emit_align.
        #
        # This matters when window_ms is not an integer multiple of hop_ms.
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
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )
        cfg = RunConfig(
            freevc_dir=str(freevc_dir),
            variant=str(variant),
            device=str(device),
            seed=int(args.seed),
            wavlm_model=str(args.wavlm_model),
            ref_trim_db=float(args.ref_trim_db),
            stream=stream_cfg,
        )

        stats = {
            "delay_samples": int(delay_samples),
            "warmup_hops": int(warmup_hops) if bool(args.stream) else 0,
            "mean_window_sec": float(np.mean(np.asarray(timings, dtype=np.float64)))
            if timings
            else 0.0,
            "p95_window_sec": float(
                np.percentile(np.asarray(timings, dtype=np.float64), 95)
            )
            if len(timings) >= 2
            else (float(timings[0]) if timings else 0.0),
            "windows": int(len(timings)),
        }
        meta_p.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2))

    print(f"Wrote: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

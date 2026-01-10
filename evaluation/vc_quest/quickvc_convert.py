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
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.streaming_utils import (
    apply_peak_limiter,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
)

_MEL_BASIS: dict[str, torch.Tensor] = {}
_HANN_WINDOW: dict[str, torch.Tensor] = {}


EmitAlign = Literal["start", "center", "end"]
VadMode = Literal["off", "rms", "webrtc"]


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    fade_ms: int
    normalize_align: str
    emit_align: str
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
    quickvc_dir: str
    config_path: str
    ckpt_path: str
    device: str
    seed: int
    max_sec: float
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


def _patch_sys_path(repo_dir: str) -> None:
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # QuickVC uses top-level modules like `utils`, `modules`, `commons` which can collide with Amphion.
    for mod_name in (
        "utils",
        "modules",
        "commons",
        "attentions",
        "mel_processing",
        "models",
        "stft",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue

        mod_file = getattr(mod, "__file__", "") or ""
        if mod_file and not os.path.abspath(mod_file).startswith(repo_dir):
            sys.modules.pop(mod_name, None)


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _resample_if_needed(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return wav
    import resampy

    return resampy.resample(wav, src_sr, dst_sr).astype(np.float32, copy=False)


@torch.inference_mode()
def _build_engine(
    *,
    quickvc_dir: str,
    config_path: str,
    ckpt_path: str,
    device: torch.device,
):
    _patch_sys_path(quickvc_dir)

    # QuickVC imports `from scipy.signal import kaiser`, but recent SciPy versions moved/stop-export it.
    # Patch the attribute so QuickVC's import continues to work.
    try:
        import scipy.signal as _scipy_signal

        if not hasattr(_scipy_signal, "kaiser"):
            from scipy.signal.windows import kaiser as _kaiser  # type: ignore

            setattr(_scipy_signal, "kaiser", _kaiser)
    except Exception:
        pass

    import utils  # type: ignore[import-not-found]
    from models import SynthesizerTrn  # type: ignore[import-not-found]

    hps = utils.get_hparams_from_file(config_path)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    net_g.eval()
    _ = utils.load_checkpoint(ckpt_path, net_g, None)

    # HuBERT-Soft content encoder (torchhub).
    hubert_soft = torch.hub.load("bshall/hubert:main", "hubert_soft")
    hubert_soft = hubert_soft.to(device)
    hubert_soft.eval()

    return hps, net_g, hubert_soft


def _emit_start(
    *,
    emit_align: EmitAlign,
    window_out: int,
    hop_out: int,
) -> int:
    if hop_out > window_out:
        return 0
    if emit_align == "start":
        return 0
    if emit_align == "end":
        return int(window_out - hop_out)
    if emit_align == "center":
        return int((window_out - hop_out) // 2)
    raise ValueError(f"Unknown emit_align: {emit_align}")


def _mel_spectrogram_torch(
    y: torch.Tensor,
    *,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: Optional[float],
    center: bool = False,
) -> torch.Tensor:
    """QuickVC-compatible mel spectrogram (Torch STFT + librosa mel filterbank).

    QuickVC's upstream `mel_processing.py` uses a positional call to `librosa.filters.mel`,
    which breaks with modern librosa (keyword-only). We compute the same thing here.
    """

    import librosa

    if y.ndim != 2:
        raise ValueError(f"Expected y as [B, T], got shape={tuple(y.shape)}")

    fmax_val = float(sampling_rate) / 2.0 if fmax is None else float(fmax)

    dtype_device = f"{y.dtype}_{y.device}"
    fmax_key = f"{fmax_val}_{num_mels}_{n_fft}_{sampling_rate}_{dtype_device}"
    win_key = f"{win_size}_{dtype_device}"

    if fmax_key not in _MEL_BASIS:
        mel = librosa.filters.mel(
            sr=int(sampling_rate),
            n_fft=int(n_fft),
            n_mels=int(num_mels),
            fmin=float(fmin),
            fmax=float(fmax_val),
        )
        _MEL_BASIS[fmax_key] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if win_key not in _HANN_WINDOW:
        _HANN_WINDOW[win_key] = torch.hann_window(int(win_size)).to(dtype=y.dtype, device=y.device)

    pad = int((int(n_fft) - int(hop_size)) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(
        y,
        n_fft=int(n_fft),
        hop_length=int(hop_size),
        win_length=int(win_size),
        window=_HANN_WINDOW[win_key],
        center=bool(center),
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    mel_spec = torch.matmul(_MEL_BASIS[fmax_key], spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return mel_spec


@torch.inference_mode()
def _infer_quickvc(
    *,
    net_g,
    hps,
    hubert_soft,
    src_wav: np.ndarray,
    ref_mel: torch.Tensor,
    out_sample_rate: int,
    device: torch.device,
) -> np.ndarray:
    src_wav = np.asarray(src_wav, dtype=np.float32).reshape(-1)
    wav_src = torch.from_numpy(src_wav).to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    c = hubert_soft.units(wav_src)  # [1, T', 256]
    c = c.transpose(2, 1)  # [1, 256, T']
    audio = net_g.infer(c, mel=ref_mel)
    audio = audio[0][0].detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description="QuickVC offline + streaming-sim converter (vc_quest).")
    parser.add_argument("--quickvc_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sec", type=float, default=0.0)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--meta_json", type=str, default="")

    parser.add_argument("--ref_trim_db", type=float, default=20.0)

    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--window_ms", type=int, default=800)
    parser.add_argument("--hop_ms", type=int, default=400)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, choices=["start", "end"], default="end")
    parser.add_argument("--emit_align", type=str, choices=["start", "center", "end"], default="end")

    parser.add_argument("--vad_mode", type=str, choices=["off", "rms", "webrtc"], default="off")
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, choices=[10, 20, 30], default=30)
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)

    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args()

    _set_determinism(int(args.seed))
    device = torch.device(str(args.device))

    hps, net_g, hubert_soft = _build_engine(
        quickvc_dir=str(args.quickvc_dir),
        config_path=str(args.config),
        ckpt_path=str(args.ckpt),
        device=device,
    )

    out_sr = int(hps.data.sampling_rate)
    hop_len = int(hps.data.hop_length)

    # Reference mel (trim silence for more stable speaker embedding).
    import librosa

    ref_wav, _ = librosa.load(str(args.ref), sr=out_sr)
    if float(args.ref_trim_db) > 0:
        ref_wav, _ = librosa.effects.trim(ref_wav, top_db=float(args.ref_trim_db))
    ref_wav_t = torch.from_numpy(np.asarray(ref_wav, dtype=np.float32)).to(device).unsqueeze(0)
    ref_mel = _mel_spectrogram_torch(
        ref_wav_t,
        n_fft=int(hps.data.filter_length),
        num_mels=int(hps.data.n_mel_channels),
        sampling_rate=int(hps.data.sampling_rate),
        hop_size=int(hps.data.hop_length),
        win_size=int(hps.data.win_length),
        fmin=float(hps.data.mel_fmin),
        fmax=hps.data.mel_fmax,
    )

    src_wav, src_sr = _load_mono(str(args.src))
    src_wav = _resample_if_needed(src_wav, src_sr, out_sr)
    if float(args.max_sec) > 0:
        src_wav = src_wav[: int(round(float(args.max_sec) * float(out_sr)))]

    # Normalize input amplitude (mirrors librosa.util.normalize used in upstream scripts).
    peak = float(np.max(np.abs(src_wav))) if len(src_wav) else 0.0
    if np.isfinite(peak) and peak > 1e-6:
        src_wav = (src_wav / peak * 0.95).astype(np.float32, copy=False)

    fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))

    timings: list[float] = []
    delay_samples = 0

    if not bool(args.stream):
        t0 = time.time()
        out = _infer_quickvc(
            net_g=net_g,
            hps=hps,
            hubert_soft=hubert_soft,
            src_wav=src_wav,
            ref_mel=ref_mel,
            out_sample_rate=out_sr,
            device=device,
        )
        total_sec = time.time() - t0
        out = apply_peak_limiter(out, peak_limit=float(args.peak_limit))
        sf.write(str(args.out), out, out_sr)

        stats = {
            "delay_samples": 0,
            "total_sec": float(total_sec),
            "rtf_total": float(total_sec) / max(len(out) / float(out_sr), 1e-9),
        }
    else:
        window_in = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
        hop_in = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
        window_in = max(1, window_in)
        hop_in = max(1, hop_in)

        window_out = window_in
        hop_out = hop_in

        emit_start_out = _emit_start(
            emit_align=str(args.emit_align),  # type: ignore[arg-type]
            window_out=window_out,
            hop_out=hop_out,
        )
        delay_samples = int(emit_start_out)

        prev_tail: Optional[np.ndarray] = None
        outs: list[np.ndarray] = []

        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-9)))
        hangover_left = 0

        total_frames = int(np.ceil(len(src_wav) / float(hop_in)))
        first_emit_wall_sec: Optional[float] = None
        begin_time = time.time()

        for i in range(total_frames):
            pos = i * hop_in
            window = src_wav[pos : pos + window_in]
            if len(window) < window_in:
                window = np.pad(window, (0, window_in - len(window)), mode="constant")

            hop = window[emit_start_out : emit_start_out + hop_out]

            vad_mode = str(args.vad_mode)
            if vad_mode == "off":
                voiced = True
            elif vad_mode == "rms":
                voiced = not (
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        hop,
                        sample_rate=out_sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )
            elif vad_mode == "webrtc":
                voiced = is_voiced_webrtcvad(
                    hop,
                    sample_rate=out_sr,
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
                out_window = _infer_quickvc(
                    net_g=net_g,
                    hps=hps,
                    hubert_soft=hubert_soft,
                    src_wav=window,
                    ref_mel=ref_mel,
                    out_sample_rate=out_sr,
                    device=device,
                )
                timings.append(time.time() - t0)

                out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))  # type: ignore[arg-type]
                out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

            out_hop = crossfade_prefix_inplace(out_hop, prev_tail, fade_out)
            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
            if fade_out > 0:
                prev_tail = out_hop[-fade_out:].astype(np.float32, copy=True)

            outs.append(out_hop)
            if first_emit_wall_sec is None:
                first_emit_wall_sec = time.time() - begin_time

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        sf.write(str(args.out), out, out_sr)

        total_sec = time.time() - begin_time
        stats = {
            "delay_samples": int(delay_samples),
            "first_emit_wall_sec": float(first_emit_wall_sec or 0.0),
            "total_sec": float(total_sec),
            "rtf_total": float(total_sec) / max(len(out) / float(out_sr), 1e-9),
            "mean_window_sec": float(np.mean(np.asarray(timings, dtype=np.float64))) if timings else 0.0,
            "p95_window_sec": float(np.percentile(np.asarray(timings, dtype=np.float64), 95))
            if len(timings) >= 2
            else (float(timings[0]) if timings else 0.0),
            "windows": int(len(timings)),
        }

    if args.meta_json:
        meta_p = Path(str(args.meta_json))
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        stream_cfg = (
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                fade_ms=int(args.fade_ms),
                normalize_align=str(args.normalize_align),
                emit_align=str(args.emit_align),
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
            quickvc_dir=str(args.quickvc_dir),
            config_path=str(args.config),
            ckpt_path=str(args.ckpt),
            device=str(args.device),
            seed=int(args.seed),
            max_sec=float(args.max_sec),
            ref_trim_db=float(args.ref_trim_db),
            stream=stream_cfg,
        )
        meta_p.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2))

    print(f"Wrote: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

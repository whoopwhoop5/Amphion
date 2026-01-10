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
    apply_peak_limiter,
    crossfade_prefix_inplace,
    normalize_length,
)


@dataclass(frozen=True)
class StreamConfig:
    chunk_ms: int
    right_context: int
    model_context_frames: int
    vocoder_context_frames: int
    fade_ms: int
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    conan_dir: str
    config: str
    exp_name: str
    hparams_override: str
    device: str
    seed: int
    max_sec: float
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

    for mod_name in ("utils", "modules", "tasks"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue

        mod_file = getattr(mod, "__file__", "") or ""
        mod_path = getattr(mod, "__path__", None)

        if mod_file:
            if not os.path.abspath(mod_file).startswith(repo_dir):
                sys.modules.pop(mod_name, None)
            continue

        if mod_path is not None:
            paths = [os.path.abspath(str(p)) for p in list(mod_path)]
            if not any(p.startswith(repo_dir) for p in paths):
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


def _to_frames(wav: np.ndarray, *, sample_rate: int, hop_size: int) -> int:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if hop_size <= 0 or sample_rate <= 0:
        raise ValueError("hop_size and sample_rate must be > 0")
    return int(np.ceil(len(wav) / float(hop_size)))


def _mel_spectrogram_torch(
    wav_16k: np.ndarray,
    *,
    fft_size: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: Optional[float],
    device: torch.device,
) -> torch.Tensor:
    """Torch STFT mel (center=False), based on Conan's `inference/Conan_previous.py`."""

    import librosa

    mel_basis: dict[str, torch.Tensor] = {}
    hann_window: dict[str, torch.Tensor] = {}

    def _spec_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def _get_mel_basis_key() -> str:
        return f"{fmax}_{device}"

    y = torch.from_numpy(np.asarray(wav_16k, dtype=np.float32).reshape(-1)).to(device)
    if y.ndim != 1:
        y = y.reshape(-1)

    y = y.unsqueeze(0)  # [1, T]

    if torch.min(y) < -1.0 or torch.max(y) > 1.0:
        y = torch.clamp(y, -1.0, 1.0)

    key = _get_mel_basis_key()
    if key not in mel_basis:
        mel = librosa.filters.mel(
            sr=int(sampling_rate),
            n_fft=int(fft_size),
            n_mels=int(num_mels),
            fmin=float(fmin),
            fmax=float(sampling_rate / 2.0 if fmax is None else fmax),
        )
        mel_basis[key] = torch.from_numpy(mel).float().to(device)
        hann_window[str(device)] = torch.hann_window(int(win_size)).to(device)

    pad = int((int(fft_size) - int(hop_size)) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(
        y,
        n_fft=int(fft_size),
        hop_length=int(hop_size),
        win_length=int(win_size),
        window=hann_window[str(device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis[key], spec)
    mel_spec = _spec_norm(mel_spec)

    # [1, n_mels, frames] -> [1, frames, n_mels]
    return mel_spec.transpose(1, 2)


@torch.inference_mode()
def _build_engine(
    *,
    conan_dir: str,
    config_path: str,
    exp_name: str,
    hparams_override: str,
    device: torch.device,
):
    _patch_sys_path(conan_dir)

    from utils.commons.ckpt_utils import load_ckpt, load_ckpt_emformer  # type: ignore[import-not-found]
    from utils.commons.hparams import hparams, set_hparams  # type: ignore[import-not-found]
    from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls  # type: ignore[import-not-found]
    from modules.Conan.Conan import Conan  # type: ignore[import-not-found]
    from modules.Emformer.emformer import EmformerDistillModel  # type: ignore[import-not-found]

    prev_cwd = os.getcwd()
    try:
        os.chdir(conan_dir)
        set_hparams(
            config=config_path,
            exp_name=exp_name,
            hparams_str=hparams_override,
            print_hparams=False,
            global_hparams=True,
        )

        model = Conan(0, hparams)
        model.eval()
        load_ckpt(model, hparams["work_dir"], strict=False)
        model = model.to(device)

        vocoder_cls = get_vocoder_cls(hparams["vocoder"])
        if vocoder_cls is None:
            raise ValueError(f"Vocoder '{hparams['vocoder']}' is not registered.")
        vocoder = vocoder_cls()

        emformer = EmformerDistillModel(hparams, output_dim=100)
        load_ckpt_emformer(emformer, hparams["emformer_ckpt"], strict=False)
        emformer.eval()
        emformer = emformer.to(device)
    finally:
        os.chdir(prev_cwd)

    return hparams, model, vocoder, emformer


@torch.inference_mode()
def _run_offline(
    *,
    hparams,
    model,
    vocoder,
    emformer,
    ref_wav: str,
    src_wav: str,
    device: torch.device,
    max_sec: float,
    peak_limit: float,
) -> tuple[np.ndarray, int, dict]:
    from utils.audio import librosa_wav2spec  # type: ignore[import-not-found]

    out_sr = int(hparams["audio_sample_rate"])
    hop_size = int(hparams["hop_size"])

    # Reference mel (style / timbre).
    ref_mel_np = librosa_wav2spec(
        ref_wav,
        fft_size=hparams["fft_size"],
        hop_size=hparams["hop_size"],
        win_length=hparams["win_size"],
        num_mels=hparams["audio_num_mel_bins"],
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        sample_rate=hparams["audio_sample_rate"],
        loud_norm=hparams["loud_norm"],
    )["mel"]
    ref_mel_np = np.clip(ref_mel_np, float(hparams["mel_vmin"]), float(hparams["mel_vmax"]))
    ref_mel = torch.from_numpy(np.asarray(ref_mel_np, dtype=np.float32)).to(device)

    # Precompute speaker/style embedding once.
    style_embed = model.encode_spk_embed(ref_mel.unsqueeze(0).transpose(1, 2)).transpose(1, 2)

    # Source waveform -> 16k, trim to max_sec.
    src_wav_np, src_sr = _load_mono(src_wav)
    src_wav_np = _resample_if_needed(src_wav_np, src_sr, out_sr)
    if float(max_sec) > 0:
        src_wav_np = src_wav_np[: int(round(float(max_sec) * float(out_sr)))]

    # Normalize input amplitude to avoid clipping differences.
    peak = float(np.max(np.abs(src_wav_np))) if len(src_wav_np) else 0.0
    if np.isfinite(peak) and peak > 1e-6:
        src_wav_np = (src_wav_np / peak * 0.95).astype(np.float32, copy=False)

    src_mel = _mel_spectrogram_torch(
        src_wav_np,
        fft_size=int(hparams["fft_size"]),
        num_mels=int(hparams["audio_num_mel_bins"]),
        sampling_rate=int(out_sr),
        hop_size=int(hparams["hop_size"]),
        win_size=int(hparams["win_size"]),
        fmin=float(hparams.get("f0_min", 50.0)),
        fmax=None,
        device=device,
    )
    total_frames = int(src_mel.shape[1])

    # Emformer full-stream inference to tokens.
    logits = emformer.inference(src_mel)  # [1, T, 100]
    codes = torch.argmax(logits, dim=-1).to(device)  # [1, T]

    t0 = time.time()
    out = model(
        content=codes,
        spk_embed=style_embed,
        target=None,
        ref=None,
        f0=None,
        uv=None,
        infer=True,
        global_steps=200000,
    )
    mel_out = out["mel_out"][0]  # [T, 80]
    wav = vocoder.spec2wav(mel_out.detach().cpu().numpy())
    total_sec = time.time() - t0

    expected_len = int(total_frames) * int(hop_size)
    wav = normalize_length(wav, expected_len, align="start")
    wav = apply_peak_limiter(wav, peak_limit=float(peak_limit))

    stats = {
        "delay_samples": 0,
        "total_sec": float(total_sec),
        "rtf_total": float(total_sec) / max(expected_len / float(out_sr), 1e-9),
    }
    return wav, out_sr, stats


@torch.inference_mode()
def _run_streaming(
    *,
    hparams,
    model,
    vocoder,
    emformer,
    ref_wav: str,
    src_wav: str,
    device: torch.device,
    max_sec: float,
    chunk_ms: int,
    right_context: int,
    model_context_frames: int,
    vocoder_context_frames: int,
    fade_ms: int,
    peak_limit: float,
) -> tuple[np.ndarray, int, dict]:
    from utils.audio import librosa_wav2spec  # type: ignore[import-not-found]

    out_sr = int(hparams["audio_sample_rate"])
    hop_size = int(hparams["hop_size"])

    # Reference mel and speaker/style embed.
    ref_mel_np = librosa_wav2spec(
        ref_wav,
        fft_size=hparams["fft_size"],
        hop_size=hparams["hop_size"],
        win_length=hparams["win_size"],
        num_mels=hparams["audio_num_mel_bins"],
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        sample_rate=hparams["audio_sample_rate"],
        loud_norm=hparams["loud_norm"],
    )["mel"]
    ref_mel_np = np.clip(ref_mel_np, float(hparams["mel_vmin"]), float(hparams["mel_vmax"]))
    ref_mel = torch.from_numpy(np.asarray(ref_mel_np, dtype=np.float32)).to(device)
    style_embed = model.encode_spk_embed(ref_mel.unsqueeze(0).transpose(1, 2)).transpose(1, 2)

    # Source -> 16k -> mel.
    src_wav_np, src_sr = _load_mono(src_wav)
    src_wav_np = _resample_if_needed(src_wav_np, src_sr, out_sr)
    if float(max_sec) > 0:
        src_wav_np = src_wav_np[: int(round(float(max_sec) * float(out_sr)))]

    peak = float(np.max(np.abs(src_wav_np))) if len(src_wav_np) else 0.0
    if np.isfinite(peak) and peak > 1e-6:
        src_wav_np = (src_wav_np / peak * 0.95).astype(np.float32, copy=False)

    src_mel = _mel_spectrogram_torch(
        src_wav_np,
        fft_size=int(hparams["fft_size"]),
        num_mels=int(hparams["audio_num_mel_bins"]),
        sampling_rate=int(out_sr),
        hop_size=int(hparams["hop_size"]),
        win_size=int(hparams["win_size"]),
        fmin=float(hparams.get("f0_min", 50.0)),
        fmax=None,
        device=device,
    )
    total_frames = int(src_mel.shape[1])
    expected_len = int(total_frames) * int(hop_size)

    frames_per_chunk = max(1, int(round(int(chunk_ms) / 20)))
    rc = int(right_context)
    fade = int(round(float(fade_ms) / 1000.0 * float(out_sr)))

    codes_tail: Optional[torch.Tensor] = None  # [1, T_ctx]
    mel_tail: Optional[np.ndarray] = None  # [T_ctx, 80]
    prev_audio_tail: Optional[np.ndarray] = None

    state = None
    pos = 0
    timings: list[float] = []
    outs: list[np.ndarray] = []

    begin_time = time.time()
    first_emit_wall_sec: Optional[float] = None

    while pos < total_frames:
        emit = min(frames_per_chunk, total_frames - pos)
        look = min(rc, total_frames - (pos + emit))
        real_len = emit + look
        chunk = src_mel[:, pos : pos + real_len, :]  # [1, real_len, 80]
        need_pad = (frames_per_chunk + rc) - real_len
        if need_pad > 0:
            pad = chunk[:, -1:, :].expand(1, need_pad, int(chunk.shape[2]))
            chunk = torch.cat([chunk, pad], dim=1)

        lengths = torch.full((1,), int(chunk.shape[1]), dtype=torch.long, device=device)

        t0 = time.time()
        chunk_out, _, state = emformer.emformer.infer(chunk, lengths, state)
        chunk_out = emformer.proj(chunk_out)
        codes_new = torch.argmax(chunk_out, dim=-1)[:, :emit]  # [1, emit]

        if codes_tail is None:
            codes_window = codes_new
        else:
            codes_window = torch.cat([codes_tail, codes_new], dim=1)
        if int(model_context_frames) > 0 and int(codes_window.shape[1]) > int(model_context_frames):
            codes_window = codes_window[:, -int(model_context_frames) :]

        out = model(
            content=codes_window,
            spk_embed=style_embed,
            target=None,
            ref=None,
            f0=None,
            uv=None,
            infer=True,
            global_steps=200000,
        )
        mel_out = out["mel_out"][0]  # [T_win, 80]
        mel_new = mel_out[-emit:].detach().cpu().numpy().astype(np.float32, copy=False)  # [emit, 80]

        if mel_tail is None or int(vocoder_context_frames) <= 0:
            mel_in = mel_new
            context_frames = 0
        else:
            mel_in = np.concatenate([mel_tail, mel_new], axis=0)
            context_frames = int(mel_tail.shape[0])

        wav_in = vocoder.spec2wav(mel_in)
        context_samples = int(context_frames) * int(hop_size)
        emit_samples = int(emit) * int(hop_size)

        if len(wav_in) >= context_samples + emit_samples:
            wav_new = wav_in[context_samples : context_samples + emit_samples]
        else:
            wav_new = wav_in[-emit_samples:] if emit_samples > 0 else np.zeros(0, dtype=np.float32)
            if len(wav_new) < emit_samples:
                wav_new = np.pad(wav_new, (0, emit_samples - len(wav_new)), mode="constant")

        wav_new = crossfade_prefix_inplace(wav_new, prev_audio_tail, fade_len=int(fade))
        wav_new = apply_peak_limiter(wav_new, peak_limit=float(peak_limit))

        if fade > 0 and len(wav_new) >= fade:
            prev_audio_tail = wav_new[-fade:].astype(np.float32, copy=True)

        outs.append(np.asarray(wav_new, dtype=np.float32).reshape(-1))
        timings.append(time.time() - t0)
        if first_emit_wall_sec is None:
            first_emit_wall_sec = time.time() - begin_time

        # Update tails for next step.
        if int(model_context_frames) > 0:
            codes_tail = codes_window[:, -int(model_context_frames) :].detach()
        else:
            codes_tail = codes_window.detach()

        if int(vocoder_context_frames) > 0:
            mel_tail_full = (
                mel_new
                if mel_tail is None
                else np.concatenate([mel_tail, mel_new], axis=0).astype(np.float32, copy=False)
            )
            mel_tail = mel_tail_full[-int(vocoder_context_frames) :].astype(np.float32, copy=True)
        else:
            mel_tail = None

        pos += emit

    wav = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
    wav = normalize_length(wav, expected_len, align="start")
    total_sec = time.time() - begin_time

    stats = {
        "delay_samples": 0,
        "first_emit_wall_sec": float(first_emit_wall_sec) if first_emit_wall_sec is not None else float("nan"),
        "total_sec": float(total_sec),
        "rtf_total": float(total_sec) / max(expected_len / float(out_sr), 1e-9),
        "mean_chunk_sec": float(np.mean(np.asarray(timings, dtype=np.float64))) if timings else 0.0,
        "p95_chunk_sec": float(np.percentile(np.asarray(timings, dtype=np.float64), 95))
        if len(timings) >= 2
        else (float(timings[0]) if timings else 0.0),
        "chunks": int(len(timings)),
    }
    return wav, out_sr, stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Conan VC runner (offline or streaming simulation).")
    parser.add_argument("--conan_dir", type=str, required=True, help="Path to Conan repo checkout.")
    parser.add_argument("--config", type=str, required=True, help="Path to Conan yaml config (relative to conan_dir ok).")
    parser.add_argument("--exp_name", type=str, required=True, help="Checkpoint exp_name (maps to checkpoints/<exp_name>).")
    parser.add_argument(
        "--hparams_override",
        type=str,
        default="",
        help="Comma-separated overrides passed to Conan set_hparams(), e.g. emformer_ckpt=checkpoints/Emformer",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base.")
    parser.add_argument("--max_sec", type=float, default=0.0, help="If >0, trim source audio to this many seconds.")

    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--meta_json", type=str, default="")

    parser.add_argument("--stream", action="store_true", help="Run streaming simulation.")
    parser.add_argument("--chunk_ms", type=int, default=80)
    parser.add_argument("--right_context", type=int, default=2)
    parser.add_argument("--model_context_frames", type=int, default=64)
    parser.add_argument("--vocoder_context_frames", type=int, default=4)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    _set_determinism(int(args.seed))

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    conan_dir = os.path.abspath(str(args.conan_dir))
    if not os.path.isdir(conan_dir):
        raise FileNotFoundError(f"conan_dir not found: {conan_dir}")

    config_path = str(args.config)
    if not os.path.isabs(config_path):
        config_path = os.path.join(conan_dir, config_path)
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")

    hparams, model, vocoder, emformer = _build_engine(
        conan_dir=conan_dir,
        config_path=config_path,
        exp_name=str(args.exp_name),
        hparams_override=str(args.hparams_override),
        device=device,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if not bool(args.stream):
        wav, sr, stats = _run_offline(
            hparams=hparams,
            model=model,
            vocoder=vocoder,
            emformer=emformer,
            ref_wav=str(args.ref),
            src_wav=str(args.src),
            device=device,
            max_sec=float(args.max_sec),
            peak_limit=float(args.peak_limit),
        )
    else:
        wav, sr, stats = _run_streaming(
            hparams=hparams,
            model=model,
            vocoder=vocoder,
            emformer=emformer,
            ref_wav=str(args.ref),
            src_wav=str(args.src),
            device=device,
            max_sec=float(args.max_sec),
            chunk_ms=int(args.chunk_ms),
            right_context=int(args.right_context),
            model_context_frames=int(args.model_context_frames),
            vocoder_context_frames=int(args.vocoder_context_frames),
            fade_ms=int(args.fade_ms),
            peak_limit=float(args.peak_limit),
        )

    sf.write(args.out, wav, int(sr))

    if args.meta_json:
        stream_cfg = (
            StreamConfig(
                chunk_ms=int(args.chunk_ms),
                right_context=int(args.right_context),
                model_context_frames=int(args.model_context_frames),
                vocoder_context_frames=int(args.vocoder_context_frames),
                fade_ms=int(args.fade_ms),
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )
        cfg = RunConfig(
            conan_dir=str(conan_dir),
            config=str(config_path),
            exp_name=str(args.exp_name),
            hparams_override=str(args.hparams_override),
            device=str(device),
            seed=int(args.seed),
            max_sec=float(args.max_sec),
            stream=stream_cfg,
        )
        Path(args.meta_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta_json).write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2))

    print(f"Wrote: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


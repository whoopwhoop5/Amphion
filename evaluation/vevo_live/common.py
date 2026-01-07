# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import soundfile as sf
import torch
from torchmetrics import WordErrorRate

import whisper

from evaluation.metrics.similarity.speaker_similarity import extract_similarity
from models.vc.vevo.live_engine import AudioRingBuffer, crossfade_inplace, normalize_length
from models.vc.vevo.live_engine import VevoStreamingEngine
from models.vc.vevo.runner import VevoConverter


VevoKind = Literal["vevotimbre", "vevovoice"]


@dataclass(frozen=True)
class VevoInferenceConfig:
    kind: VevoKind = "vevotimbre"
    flow_matching_steps: int = 16
    diffusion_cfg: float = 1.0
    diffusion_rescale_cfg: float = 0.75
    seed: int = 1234

    # vevovoice-only
    ar_max_length: int = 2000
    ar_temperature: float = 0.8
    ar_top_k: int = 50
    ar_top_p: float = 0.9
    ar_repeat_penalty: float = 1.0
    ar_min_new_tokens: int = 50
    prepend_style_ref_to_input: bool = True


@dataclass(frozen=True)
class VevoStreamingConfig:
    window_ms: int = 1000
    hop_ms: int = 1000
    fade_ms: int = 20
    normalize_align: Literal["start", "end"] = "end"


@dataclass(frozen=True)
class EvalConfig:
    inference: VevoInferenceConfig
    streaming: VevoStreamingConfig

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def uid(self) -> str:
        h = hashlib.sha1(self.to_json().encode("utf-8")).hexdigest()
        return h[:10]


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav, int(sr)


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.astype(np.float32, copy=False), sr)


def compute_speaker_similarity(
    *,
    ref_wav_path: str,
    deg_dir: str,
    work_dir: str,
    model_name: Literal["wavlm", "rawnet", "resemblyzer"] = "wavlm",
) -> float:
    ref_dir = os.path.join(work_dir, "ref")
    Path(ref_dir).mkdir(parents=True, exist_ok=True)
    ref_dst = os.path.join(ref_dir, "ref.wav")
    if not os.path.exists(ref_dst):
        # Copy once for determinism.
        Path(ref_dst).write_bytes(Path(ref_wav_path).read_bytes())

    score = extract_similarity(
        ref_dir,
        deg_dir,
        kwargs={"model_name": model_name, "similarity_mode": "overall"},
    )
    return float(score)


def _normalize_text(text: str) -> str:
    for ch in [" ", ".", "'", "-", ",", "!", "?", "…", "，", "。", "！", "？"]:
        text = text.replace(ch, "")
    return text.lower()


def compute_wer_whisper(
    whisper_model,
    *,
    audio_ref_path: str,
    audio_deg_path: str,
) -> float:
    ref = whisper_model.transcribe(audio_ref_path, verbose=False)
    deg = whisper_model.transcribe(audio_deg_path, verbose=False)

    wer = WordErrorRate()
    if torch.cuda.is_available():
        wer = wer.to("cuda")

    ref_text = _normalize_text(ref["text"])
    deg_text = _normalize_text(deg["text"])
    return float(wer(deg_text, ref_text).detach().cpu().numpy().tolist())


def glitch_metrics(
    wav: np.ndarray,
    *,
    hop_samples: int,
) -> dict[str, float]:
    wav = wav.reshape(-1).astype(np.float32, copy=False)
    if hop_samples <= 0:
        return {"boundary_jump_ratio_mean": 0.0, "boundary_jump_ratio_p95": 0.0}

    diffs = np.abs(np.diff(wav))
    base = float(np.median(diffs)) + 1e-6

    jumps = []
    for idx in range(hop_samples, len(wav), hop_samples):
        if idx <= 0 or idx >= len(wav):
            continue
        jumps.append(abs(float(wav[idx]) - float(wav[idx - 1])) / base)

    if not jumps:
        return {"boundary_jump_ratio_mean": 0.0, "boundary_jump_ratio_p95": 0.0}

    jumps_np = np.asarray(jumps, dtype=np.float32)
    return {
        "boundary_jump_ratio_mean": float(np.mean(jumps_np)),
        "boundary_jump_ratio_p95": float(np.percentile(jumps_np, 95)),
    }


@torch.no_grad()
def simulate_streaming(
    converter: VevoConverter,
    *,
    reference_wav_path: str,
    source_wav_path: str,
    cfg: EvalConfig,
    max_hops: int = 0,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Run a deterministic, server-equivalent streaming simulation on a single file."""

    set_determinism(cfg.inference.seed)

    engine = VevoStreamingEngine(converter)
    engine.prepare_reference_bytes(Path(reference_wav_path).read_bytes())

    src, sr = load_mono(source_wav_path)
    if sr != engine.model_sr:
        raise ValueError(
            f"simulate_streaming expects source_wav at {engine.model_sr}Hz, got {sr}Hz: {source_wav_path}"
        )

    window_samples = int(round(cfg.streaming.window_ms / 1000 * engine.model_sr))
    hop_samples = int(round(cfg.streaming.hop_ms / 1000 * engine.model_sr))
    fade_samples = int(round(cfg.streaming.fade_ms / 1000 * engine.model_sr))

    ring = AudioRingBuffer(window_samples)
    prev_tail = None
    outs = []

    hop_count = 0
    timings = []

    for start in range(0, len(src), hop_samples):
        if max_hops and hop_count >= max_hops:
            break

        chunk = src[start : start + hop_samples]
        if len(chunk) < hop_samples:
            chunk = np.pad(chunk, (0, hop_samples - len(chunk)), mode="constant")
        ring.write(chunk)

        if ring.size < window_samples:
            outs.append(np.zeros(hop_samples, dtype=np.float32))
            hop_count += 1
            continue

        window = ring.read_last(window_samples)
        t0 = time.time()
        out_window = engine.convert_window(
            window,
            flow_matching_steps=cfg.inference.flow_matching_steps,
            diffusion_cfg=cfg.inference.diffusion_cfg,
            diffusion_rescale_cfg=cfg.inference.diffusion_rescale_cfg,
            seed=cfg.inference.seed + hop_count,
            ar_max_length=cfg.inference.ar_max_length,
            ar_temperature=cfg.inference.ar_temperature,
            ar_top_k=cfg.inference.ar_top_k,
            ar_top_p=cfg.inference.ar_top_p,
            ar_repeat_penalty=cfg.inference.ar_repeat_penalty,
            ar_min_new_tokens=cfg.inference.ar_min_new_tokens,
            prepend_style_ref_to_input=cfg.inference.prepend_style_ref_to_input,
        )
        timings.append(time.time() - t0)

        out_window = normalize_length(out_window, window_samples, align=cfg.streaming.normalize_align)
        hop = out_window[-hop_samples:].astype(np.float32, copy=False)
        hop = crossfade_inplace(hop, prev_tail, fade_samples)
        prev_tail = hop[-fade_samples:].copy() if fade_samples > 0 else None

        outs.append(hop)
        hop_count += 1

    out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    stats = {
        "hop_count": hop_count,
        "window_ms": cfg.streaming.window_ms,
        "hop_ms": cfg.streaming.hop_ms,
        "fade_ms": cfg.streaming.fade_ms,
        "mean_window_sec": float(np.mean(timings)) if timings else 0.0,
        "p95_window_sec": float(np.percentile(timings, 95)) if timings else 0.0,
    }
    return out, engine.model_sr, stats


def list_wavs(playlist_dir: str) -> list[str]:
    wavs = sorted(str(p) for p in Path(playlist_dir).glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in: {playlist_dir}")
    return wavs


def load_whisper(model_size: str = "base"):
    model = whisper.load_model(model_size)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


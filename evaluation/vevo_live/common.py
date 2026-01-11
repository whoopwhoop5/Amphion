# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch.nn.functional as F
import soundfile as sf
import torch
import torchaudio

if TYPE_CHECKING:
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
    window_ms: int = 2000
    hop_ms: int = 1000
    fade_ms: int = 10
    vad_db: float = -55.0
    vad_frame_ms: float = 10.0
    peak_limit: float = 0.99
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


def _encode_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32).reshape(-1), sr, format="WAV")
    return buf.getvalue()


def read_reference_wav_bytes(path: str, *, max_sec: float = 10.0) -> bytes:
    wav, sr = load_mono(path)
    if max_sec > 0:
        max_samples = int(round(max_sec * sr))
        if len(wav) > max_samples:
            wav = wav[:max_samples]
    return _encode_wav_bytes(wav, sr)


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.astype(np.float32, copy=False), sr)


class SpeakerSimilarityScorer:
    def __init__(
        self,
        *,
        model_name: Literal["wavlm", "resemblyzer"],
        ref_wav_path: str,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_name = model_name
        self.ref_wav_path = ref_wav_path
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._resemblyzer_encoder = None
        self._wavlm_feature_extractor = None
        self._wavlm_model = None

        self._ref_emb: Optional[torch.Tensor] = None

        if self.model_name == "resemblyzer":
            from resemblyzer import VoiceEncoder, preprocess_wav

            self._preprocess_wav = preprocess_wav
            self._resemblyzer_encoder = VoiceEncoder().to(self.device).eval()
            ref_wav = self._preprocess_wav(self.ref_wav_path)
            ref_emb = torch.from_numpy(
                self._resemblyzer_encoder.embed_utterance(ref_wav)
            ).to(self.device)
            self._ref_emb = F.normalize(ref_emb, dim=0)
        elif self.model_name == "wavlm":
            import librosa
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

            self._librosa = librosa
            self._wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus-sv"
            )
            self._wavlm_model = (
                WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
                .to(self.device)
                .eval()
            )

            ref_wav, _ = self._librosa.load(self.ref_wav_path, sr=16000)
            ref_inputs = self._wavlm_feature_extractor(
                [ref_wav], padding=True, return_tensors="pt", sampling_rate=16000
            )
            ref_inputs = {k: v.to(self.device) for k, v in ref_inputs.items()}
            with torch.no_grad():
                ref_emb = self._wavlm_model(**ref_inputs).embeddings[0]
                self._ref_emb = F.normalize(ref_emb, dim=0)
        else:
            raise ValueError(f"Unsupported similarity model: {self.model_name}")

    def score_dir(self, deg_dir: str) -> float:
        deg_paths = sorted(str(p) for p in Path(deg_dir).glob("*.wav"))
        if not deg_paths:
            raise FileNotFoundError(f"No .wav files found in: {deg_dir}")
        return self.score_paths(deg_paths)

    def score_paths(self, deg_paths: list[str]) -> float:
        assert self._ref_emb is not None
        scores: list[float] = []

        if self.model_name == "resemblyzer":
            assert self._resemblyzer_encoder is not None
            for p in deg_paths:
                wav = self._preprocess_wav(p)  # type: ignore[attr-defined]
                emb = torch.from_numpy(
                    self._resemblyzer_encoder.embed_utterance(wav)
                ).to(self.device)
                emb = F.normalize(emb, dim=0)
                scores.append(
                    float(
                        F.cosine_similarity(self._ref_emb, emb, dim=0)
                        .detach()
                        .cpu()
                        .item()
                    )
                )
            return float(np.mean(scores))

        if self.model_name == "wavlm":
            assert self._wavlm_feature_extractor is not None
            assert self._wavlm_model is not None
            for p in deg_paths:
                wav, _ = self._librosa.load(p, sr=16000)  # type: ignore[attr-defined]
                inputs = self._wavlm_feature_extractor(
                    [wav], padding=True, return_tensors="pt", sampling_rate=16000
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    emb = self._wavlm_model(**inputs).embeddings[0]
                    emb = F.normalize(emb, dim=0)
                    scores.append(
                        float(
                            F.cosine_similarity(self._ref_emb, emb, dim=0)
                            .detach()
                            .cpu()
                            .item()
                        )
                    )
            return float(np.mean(scores))

        raise ValueError(f"Unsupported similarity model: {self.model_name}")


@torch.no_grad()
def compute_content_similarity_hubert(
    converter: "VevoConverter",
    *,
    src_wav: np.ndarray,
    deg_wav: np.ndarray,
    sample_rate: int,
) -> float:
    """Content similarity using mean-pooled HuBERT features (cosine)."""

    src_wav = np.asarray(src_wav, dtype=np.float32).reshape(-1)
    deg_wav = np.asarray(deg_wav, dtype=np.float32).reshape(-1)
    if len(src_wav) == 0 or len(deg_wav) == 0:
        return float("nan")

    device = converter.device
    pipe = converter.pipeline

    src = torch.from_numpy(src_wav).unsqueeze(0).to(device)
    deg = torch.from_numpy(deg_wav).unsqueeze(0).to(device)

    if sample_rate != 16000:
        src_16k = torchaudio.functional.resample(src, sample_rate, 16000)
        deg_16k = torchaudio.functional.resample(deg, sample_rate, 16000)
    else:
        src_16k = src
        deg_16k = deg

    feats_src, len_src = pipe.extract_hubert_feature(src_16k, output_layer=18)
    feats_deg, len_deg = pipe.extract_hubert_feature(deg_16k, output_layer=18)

    ls = int(len_src[0].detach().cpu().item())
    ld = int(len_deg[0].detach().cpu().item())
    if ls <= 0 or ld <= 0:
        return float("nan")

    emb_src = feats_src[0, :ls].mean(dim=0)
    emb_deg = feats_deg[0, :ld].mean(dim=0)
    emb_src = F.normalize(emb_src, dim=0)
    emb_deg = F.normalize(emb_deg, dim=0)

    return float(F.cosine_similarity(emb_src, emb_deg, dim=0).detach().cpu().item())


def _normalize_text_for_wer(text: str) -> str:
    # Keep spaces (WER is word-based), remove common punctuation, lower-case.
    for ch in [
        ".",
        "'",
        "-",
        ",",
        "!",
        "?",
        "…",
        "，",
        "。",
        "！",
        "？",
        '"',
        "“",
        "”",
    ]:
        text = text.replace(ch, "")
    text = " ".join(text.strip().split())
    return text.lower()


def _word_error_rate(hyp_words: list[str], ref_words: list[str]) -> float:
    if not ref_words:
        return float("nan")

    # Classic Levenshtein distance on word tokens.
    n = len(ref_words)
    m = len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1, dtype=np.int32)
    dp[0, :] = np.arange(m + 1, dtype=np.int32)

    for i in range(1, n + 1):
        r = ref_words[i - 1]
        for j in range(1, m + 1):
            h = hyp_words[j - 1]
            cost = 0 if r == h else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,  # deletion
                dp[i, j - 1] + 1,  # insertion
                dp[i - 1, j - 1] + cost,  # substitution
            )

    return float(dp[n, m]) / float(n)


def compute_wer_whisper(
    whisper_model,
    *,
    audio_ref_path: str,
    audio_deg_path: str,
) -> float:
    ref = whisper_model.transcribe(audio_ref_path, verbose=False)
    lang = ref.get("language")
    deg = (
        whisper_model.transcribe(audio_deg_path, verbose=False, language=lang)
        if lang
        else whisper_model.transcribe(audio_deg_path, verbose=False)
    )

    ref_text = _normalize_text_for_wer(ref["text"])
    deg_text = _normalize_text_for_wer(deg["text"])
    if not ref_text:
        # Avoid divide-by-zero inside WER when reference has 0 words.
        return float("nan")
    return _word_error_rate(deg_text.split(), ref_text.split())


def glitch_metrics(
    wav: np.ndarray,
    *,
    hop_samples: int,
) -> dict[str, float]:
    wav = wav.reshape(-1).astype(np.float32, copy=False)
    if hop_samples <= 0:
        return {"boundary_jump_ratio_mean": 0.0, "boundary_jump_ratio_p95": 0.0}

    diffs = np.abs(np.diff(wav))
    base = max(float(np.median(diffs)), 1e-3)

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


def artifact_metrics_aligned(
    src_aligned: np.ndarray,
    deg_aligned: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: float = 10.0,
    silence_floor_db: float = -120.0,
    eps: float = 1e-9,
) -> dict[str, float]:
    """Metrics to detect streaming artifacts (noise leakage, dropouts, pumping).

    Expects `src_aligned` and `deg_aligned` to be time-aligned and roughly equal length.
    """

    src = np.asarray(src_aligned, dtype=np.float32).reshape(-1)
    deg = np.asarray(deg_aligned, dtype=np.float32).reshape(-1)
    n = int(min(len(src), len(deg)))
    if n <= 0 or sample_rate <= 0:
        return {
            "silence_in_db": float("nan"),
            "voiced_in_db": float("nan"),
            "silent_out_db_mean": float("nan"),
            "silent_out_db_p95": float("nan"),
            "dropout_frac_voiced": float("nan"),
            "delta_db_std_voiced": float("nan"),
            "delta_db_step_p95": float("nan"),
            "env_corr_voiced": float("nan"),
            "env_corr_nonsilent": float("nan"),
            "clip_frac": 0.0,
        }

    src = src[:n]
    deg = deg[:n]

    frame = int(round(float(frame_ms) / 1000.0 * float(sample_rate)))
    frame = max(1, frame)
    nf = n // frame
    if nf <= 0:
        return {
            "silence_in_db": float("nan"),
            "voiced_in_db": float("nan"),
            "silent_out_db_mean": float("nan"),
            "silent_out_db_p95": float("nan"),
            "dropout_frac_voiced": float("nan"),
            "delta_db_std_voiced": float("nan"),
            "delta_db_step_p95": float("nan"),
            "env_corr_voiced": float("nan"),
            "env_corr_nonsilent": float("nan"),
            "clip_frac": float(np.mean(np.abs(deg) >= 0.999)),
        }

    src_f = src[: nf * frame].reshape(nf, frame)
    deg_f = deg[: nf * frame].reshape(nf, frame)

    src_rms = np.sqrt(np.mean(src_f * src_f, axis=1) + eps)
    deg_rms = np.sqrt(np.mean(deg_f * deg_f, axis=1) + eps)

    src_db = 20.0 * np.log10(src_rms + eps)
    deg_db = 20.0 * np.log10(deg_rms + eps)
    src_db = np.maximum(src_db, float(silence_floor_db))
    deg_db = np.maximum(deg_db, float(silence_floor_db))

    in_p95 = float(np.percentile(src_db, 95))
    # Thresholds adapt to recording level but stay in sane dBFS ranges.
    silence_in_db = float(min(-50.0, in_p95 - 30.0))
    voiced_in_db = float(max(-45.0, in_p95 - 10.0))
    if not np.isfinite(silence_in_db) or not np.isfinite(voiced_in_db):
        silence_in_db = -60.0
        voiced_in_db = -40.0

    silent = src_db < silence_in_db
    voiced = src_db > voiced_in_db

    silent_out_db_mean = (
        float(np.mean(deg_db[silent])) if np.any(silent) else float("nan")
    )
    silent_out_db_p95 = (
        float(np.percentile(deg_db[silent], 95)) if np.any(silent) else float("nan")
    )

    # Dropouts: output far quieter than input on voiced frames.
    dropout = (deg_db < (src_db - 25.0)) & voiced
    dropout_frac_voiced = float(np.mean(dropout)) if np.any(voiced) else float("nan")

    delta_db = deg_db - src_db
    delta_db_std_voiced = (
        float(np.std(delta_db[voiced])) if np.any(voiced) else float("nan")
    )
    delta_db_step_p95 = (
        float(np.percentile(np.abs(np.diff(delta_db)), 95))
        if len(delta_db) > 1
        else 0.0
    )

    def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        nxy = int(min(len(x), len(y)))
        if nxy < 2:
            return float("nan")

        x = x[:nxy]
        y = y[:nxy]
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if not np.isfinite(denom) or denom <= float(eps):
            return float("nan")
        return float(np.sum(x * y) / denom)

    env_corr_voiced = (
        _pearson_corr(src_db[voiced], deg_db[voiced])
        if np.any(voiced)
        else float("nan")
    )
    nonsilent = ~silent
    env_corr_nonsilent = (
        _pearson_corr(src_db[nonsilent], deg_db[nonsilent])
        if np.any(nonsilent)
        else float("nan")
    )

    clip_frac = float(np.mean(np.abs(deg) >= 0.999))
    return {
        "silence_in_db": float(silence_in_db),
        "voiced_in_db": float(voiced_in_db),
        "silent_out_db_mean": float(silent_out_db_mean),
        "silent_out_db_p95": float(silent_out_db_p95),
        "dropout_frac_voiced": float(dropout_frac_voiced),
        "delta_db_std_voiced": float(delta_db_std_voiced),
        "delta_db_step_p95": float(delta_db_step_p95),
        "env_corr_voiced": float(env_corr_voiced),
        "env_corr_nonsilent": float(env_corr_nonsilent),
        "clip_frac": float(clip_frac),
    }


def pitch_metrics_aligned(
    src_aligned: np.ndarray,
    deg_aligned: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: float = 10.0,
    fmin: float = 50.0,
    fmax: float = 500.0,
    resample_to: int = 16000,
    eps: float = 1e-9,
) -> dict[str, float]:
    """Rough pitch preservation metrics (voiced frames only).

    Uses `librosa.yin` and an RMS-based voiced mask. Values are intended as relative signals
    for tuning (not absolute "ground truth").
    """

    src = np.asarray(src_aligned, dtype=np.float32).reshape(-1)
    deg = np.asarray(deg_aligned, dtype=np.float32).reshape(-1)
    n = int(min(len(src), len(deg)))
    if n <= 0 or int(sample_rate) <= 0:
        return {
            "f0_corr_voiced": float("nan"),
            "f0_mae_cents_voiced_p50": float("nan"),
            "f0_mae_cents_voiced_mean": float("nan"),
            "f0_frames_used": 0.0,
            "f0_frames_total": 0.0,
        }

    src = src[:n]
    deg = deg[:n]

    import librosa

    sr = int(resample_to) if int(resample_to) > 0 else int(sample_rate)
    if int(sample_rate) != sr:
        src = librosa.resample(src, orig_sr=int(sample_rate), target_sr=sr).astype(
            np.float32, copy=False
        )
        deg = librosa.resample(deg, orig_sr=int(sample_rate), target_sr=sr).astype(
            np.float32, copy=False
        )

    hop = int(round(float(frame_ms) / 1000.0 * float(sr)))
    hop = max(1, hop)
    frame_length = max(int(round(0.04 * float(sr))), 4 * hop)

    try:
        f0_src = librosa.yin(
            src,
            fmin=float(fmin),
            fmax=float(fmax),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop,
            center=False,
        )
        f0_deg = librosa.yin(
            deg,
            fmin=float(fmin),
            fmax=float(fmax),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop,
            center=False,
        )
    except Exception:
        return {
            "f0_corr_voiced": float("nan"),
            "f0_mae_cents_voiced_p50": float("nan"),
            "f0_mae_cents_voiced_mean": float("nan"),
            "f0_frames_used": 0.0,
            "f0_frames_total": 0.0,
        }

    rms_src = librosa.feature.rms(
        y=src, frame_length=frame_length, hop_length=hop, center=False
    )[0]
    src_db = 20.0 * np.log10(rms_src + float(eps))

    if src_db.size == 0:
        return {
            "f0_corr_voiced": float("nan"),
            "f0_mae_cents_voiced_p50": float("nan"),
            "f0_mae_cents_voiced_mean": float("nan"),
            "f0_frames_used": 0.0,
            "f0_frames_total": float(min(len(f0_src), len(f0_deg))),
        }

    in_p95 = float(np.percentile(src_db, 95))
    voiced_in_db = float(max(-45.0, in_p95 - 10.0))
    voiced = src_db > voiced_in_db

    nframes = int(min(len(f0_src), len(f0_deg), len(voiced)))
    f0_src = np.asarray(f0_src[:nframes], dtype=np.float64)
    f0_deg = np.asarray(f0_deg[:nframes], dtype=np.float64)
    voiced = np.asarray(voiced[:nframes], dtype=bool)

    valid = (
        voiced & np.isfinite(f0_src) & np.isfinite(f0_deg) & (f0_src > 0) & (f0_deg > 0)
    )
    used = int(np.sum(valid))

    if used < 3:
        return {
            "f0_corr_voiced": float("nan"),
            "f0_mae_cents_voiced_p50": float("nan"),
            "f0_mae_cents_voiced_mean": float("nan"),
            "f0_frames_used": float(used),
            "f0_frames_total": float(nframes),
        }

    cents_src = 1200.0 * np.log2(f0_src[valid])
    cents_deg = 1200.0 * np.log2(f0_deg[valid])

    d = cents_deg - cents_src
    mae_p50 = float(np.median(np.abs(d)))
    mae_mean = float(np.mean(np.abs(d)))

    corr = float("nan")
    if float(np.std(cents_src)) > 1e-6 and float(np.std(cents_deg)) > 1e-6:
        corr = float(np.corrcoef(cents_src, cents_deg)[0, 1])

    return {
        "f0_corr_voiced": float(corr),
        "f0_mae_cents_voiced_p50": float(mae_p50),
        "f0_mae_cents_voiced_mean": float(mae_mean),
        "f0_frames_used": float(used),
        "f0_frames_total": float(nframes),
    }


@torch.no_grad()
def simulate_streaming(
    converter: "VevoConverter",
    *,
    reference_wav_path: str,
    source_wav_path: str,
    cfg: EvalConfig,
    max_hops: int = 0,
    drop_warmup_hops: bool = True,
    reference_max_sec: float = 10.0,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Run a deterministic, server-equivalent streaming simulation on a single file."""

    set_determinism(cfg.inference.seed)

    from models.vc.vevo.live_engine import (
        AudioRingBuffer,
        VevoStreamingEngine,
        apply_peak_limiter,
        is_silent_rms_db,
        normalize_length,
        smooth_boundary_inplace,
    )

    engine = VevoStreamingEngine(converter)
    engine.prepare_reference_bytes(
        read_reference_wav_bytes(reference_wav_path, max_sec=reference_max_sec)
    )

    src, sr = load_mono(source_wav_path)
    if sr != engine.model_sr:
        raise ValueError(
            f"simulate_streaming expects source_wav at {engine.model_sr}Hz, got {sr}Hz: {source_wav_path}"
        )

    window_samples = int(round(cfg.streaming.window_ms / 1000 * engine.model_sr))
    hop_samples = int(round(cfg.streaming.hop_ms / 1000 * engine.model_sr))
    fade_samples = int(round(cfg.streaming.fade_ms / 1000 * engine.model_sr))

    ring = AudioRingBuffer(window_samples)
    prev_last: Optional[float] = None
    outs = []

    input_hops = 0
    warmup_hops = 0
    window_count = 0
    timings = []

    for start in range(0, len(src), hop_samples):
        if max_hops and window_count >= max_hops:
            break

        chunk = src[start : start + hop_samples]
        if len(chunk) < hop_samples:
            chunk = np.pad(chunk, (0, hop_samples - len(chunk)), mode="constant")
        ring.write(chunk)

        if ring.size < window_samples:
            warmup_hops += 1
            input_hops += 1
            prev_last = 0.0
            if not drop_warmup_hops:
                outs.append(np.zeros(hop_samples, dtype=np.float32))
            continue

        window = ring.read_last(window_samples)

        silent = float(cfg.streaming.vad_db) > -200.0 and is_silent_rms_db(
            chunk,
            sample_rate=engine.model_sr,
            frame_ms=float(cfg.streaming.vad_frame_ms),
            silence_db=float(cfg.streaming.vad_db),
        )
        if silent:
            out_window = np.zeros(window_samples, dtype=np.float32)
        else:
            t0 = time.time()
            out_window = engine.convert_window(
                window,
                flow_matching_steps=cfg.inference.flow_matching_steps,
                diffusion_cfg=cfg.inference.diffusion_cfg,
                diffusion_rescale_cfg=cfg.inference.diffusion_rescale_cfg,
                seed=cfg.inference.seed + window_count,
                ar_max_length=cfg.inference.ar_max_length,
                ar_temperature=cfg.inference.ar_temperature,
                ar_top_k=cfg.inference.ar_top_k,
                ar_top_p=cfg.inference.ar_top_p,
                ar_repeat_penalty=cfg.inference.ar_repeat_penalty,
                ar_min_new_tokens=cfg.inference.ar_min_new_tokens,
                prepend_style_ref_to_input=cfg.inference.prepend_style_ref_to_input,
            )
            timings.append(time.time() - t0)

        out_window = normalize_length(
            out_window, window_samples, align=cfg.streaming.normalize_align
        )
        hop = out_window[-hop_samples:].astype(np.float32, copy=False)
        hop = smooth_boundary_inplace(hop, prev_last, fade_samples)
        hop = apply_peak_limiter(hop, peak_limit=float(cfg.streaming.peak_limit))
        prev_last = float(hop[-1]) if len(hop) else prev_last

        outs.append(hop)
        window_count += 1
        input_hops += 1

    out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    stats = {
        "input_hops": input_hops,
        "warmup_hops": warmup_hops,
        "window_count": window_count,
        "window_ms": cfg.streaming.window_ms,
        "hop_ms": cfg.streaming.hop_ms,
        "fade_ms": cfg.streaming.fade_ms,
        "delay_samples": int(window_samples - hop_samples),
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
    try:
        import whisper  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'openai-whisper'. Install it to enable WER scoring "
            "(e.g., `pip install -U openai-whisper`)."
        ) from e

    model = whisper.load_model(model_size)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

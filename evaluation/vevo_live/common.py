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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._resemblyzer_encoder = None
        self._wavlm_feature_extractor = None
        self._wavlm_model = None

        self._ref_emb: Optional[torch.Tensor] = None

        if self.model_name == "resemblyzer":
            from resemblyzer import VoiceEncoder, preprocess_wav

            self._preprocess_wav = preprocess_wav
            self._resemblyzer_encoder = VoiceEncoder().to(self.device).eval()
            ref_wav = self._preprocess_wav(self.ref_wav_path)
            ref_emb = torch.from_numpy(self._resemblyzer_encoder.embed_utterance(ref_wav)).to(
                self.device
            )
            self._ref_emb = F.normalize(ref_emb, dim=0)
        elif self.model_name == "wavlm":
            import librosa
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

            self._librosa = librosa
            self._wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus-sv"
            )
            self._wavlm_model = WavLMForXVector.from_pretrained(
                "microsoft/wavlm-base-plus-sv"
            ).to(self.device).eval()

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
                emb = torch.from_numpy(self._resemblyzer_encoder.embed_utterance(wav)).to(
                    self.device
                )
                emb = F.normalize(emb, dim=0)
                scores.append(
                    float(F.cosine_similarity(self._ref_emb, emb, dim=0).detach().cpu().item())
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

    device = converter.device
    pipe = converter.pipeline

    src = torch.from_numpy(np.asarray(src_wav, dtype=np.float32)).unsqueeze(0).to(device)
    deg = torch.from_numpy(np.asarray(deg_wav, dtype=np.float32)).unsqueeze(0).to(device)

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
    for ch in [".", "'", "-", ",", "!", "?", "…", "，", "。", "！", "？", "\"", "“", "”"]:
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
        normalize_length,
        smooth_boundary_inplace,
    )

    engine = VevoStreamingEngine(converter)
    engine.prepare_reference_bytes(read_reference_wav_bytes(reference_wav_path, max_sec=reference_max_sec))

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

        out_window = normalize_length(out_window, window_samples, align=cfg.streaming.normalize_align)
        hop = out_window[-hop_samples:].astype(np.float32, copy=False)
        hop = smooth_boundary_inplace(hop, prev_last, fade_samples)
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

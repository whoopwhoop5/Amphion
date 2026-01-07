# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np


def normalize_text_for_wer(text: str) -> str:
    # Keep spaces (WER is word-based), remove common punctuation, lower-case.
    for ch in [".", "'", "-", ",", "!", "?", "…", "，", "。", "！", "？", "\"", "“", "”"]:
        text = text.replace(ch, "")
    text = " ".join(text.strip().split())
    return text.lower()


def word_error_rate(hyp_text: str, ref_text: str) -> float:
    ref = normalize_text_for_wer(ref_text).split()
    hyp = normalize_text_for_wer(hyp_text).split()
    if not ref:
        return float("nan")

    n = len(ref)
    m = len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1, dtype=np.int32)
    dp[0, :] = np.arange(m + 1, dtype=np.int32)

    for i in range(1, n + 1):
        r = ref[i - 1]
        for j in range(1, m + 1):
            h = hyp[j - 1]
            cost = 0 if r == h else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,  # deletion
                dp[i, j - 1] + 1,  # insertion
                dp[i - 1, j - 1] + cost,  # substitution
            )

    return float(dp[n, m]) / float(n)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a) + eps)
    nb = float(np.linalg.norm(b) + eps)
    return float(np.dot(a, b) / (na * nb))


class WhisperTranscriber:
    def __init__(self, *, model_size: str = "large-v3", device: Optional[str] = None) -> None:
        try:
            import whisper  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'openai-whisper'. Install it (e.g., `pip install -U openai-whisper`)."
            ) from e

        import torch

        self._torch = torch
        self._model = whisper.load_model(model_size)
        if device is None:
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
        else:
            self._model = self._model.to(device)

    def transcribe(self, audio_path: str) -> tuple[str, Optional[str]]:
        out = self._model.transcribe(audio_path, verbose=False)
        return str(out.get("text", "")), out.get("language")


class ResemblyzerEmbedder:
    def __init__(self, *, device: Optional[str] = None) -> None:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'resemblyzer'. Install it (e.g., `pip install -U resemblyzer`)."
            ) from e

        import torch

        self._torch = torch
        self._preprocess_wav = preprocess_wav
        enc = VoiceEncoder()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = enc.to(self._device).eval()

    def embed(self, wav_path: str) -> np.ndarray:
        wav = self._preprocess_wav(wav_path)
        emb = self._encoder.embed_utterance(wav)
        return np.asarray(emb, dtype=np.float32)


class SpeechBrainEmbedder:
    """Generic SpeechBrain embedding wrapper (used for accent embeddings in StyleStream metrics)."""

    def __init__(self, *, hf_repo_id: str, device: Optional[str] = None) -> None:
        try:
            from speechbrain.inference.classifiers import EncoderClassifier  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'speechbrain'. Install it (e.g., `pip install -U speechbrain`)."
            ) from e

        import torch
        import torchaudio

        self._torch = torch
        self._torchaudio = torchaudio
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = EncoderClassifier.from_hparams(
            source=hf_repo_id,
            run_opts={"device": self._device},
        )

    def embed(self, wav_path: str) -> np.ndarray:
        wav, sr = self._torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav[:1, :]
        if sr != 16000:
            wav = self._torchaudio.functional.resample(wav, sr, 16000)

        with self._torch.no_grad():
            emb = self._model.encode_batch(wav.to(self._device))
        emb_np = emb.detach().cpu().numpy()
        return emb_np.reshape(-1).astype(np.float32, copy=False)


class Emotion2VecEmbedder:
    """emotion2vec embedding extractor via ModelScope pipeline.

    StyleStream uses emotion2vec embeddings for E-SIM (cosine similarity). This wrapper
    attempts to match that behavior while keeping imports optional.
    """

    def __init__(self, *, model_id: str = "iic/emotion2vec_base", device: Optional[str] = None) -> None:
        try:
            from modelscope.pipelines import pipeline  # type: ignore[import-not-found]
            from modelscope.utils.constant import Tasks  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or "modelscope"
            raise ModuleNotFoundError(
                f"Missing dependency '{missing}' required for emotion2vec (modelscope). "
                "Install it (e.g., `pip install -U modelscope funasr addict`)."
            ) from e

        self._pipeline = pipeline(task=Tasks.emotion_recognition, model=model_id, device=device)

    def embed(self, wav_path: str) -> np.ndarray:
        out: Any = self._pipeline(
            wav_path,
            granularity="utterance",
            extract_embedding=True,
        )

        # ModelScope/FunASR commonly returns a list of dicts (one per utterance). Normalize to a dict.
        if isinstance(out, list):
            if not out:
                raise ValueError("emotion2vec output is an empty list")
            if len(out) != 1:
                # We only ever pass a single utterance path; pick the first entry deterministically.
                out = out[0]
            else:
                out = out[0]

        # Embeddings may be under different keys depending on version.
        if isinstance(out, dict):
            for k in ["embedding", "embeddings", "feats", "features"]:
                if k in out:
                    emb = out[k]
                    break
            else:
                raise KeyError(f"emotion2vec output missing embedding key: {list(out.keys())}")
        else:
            emb = out

        emb_np = np.asarray(emb, dtype=np.float32)
        if emb_np.ndim > 1:
            emb_np = emb_np.reshape(-1)
        return emb_np


@dataclass(frozen=True)
class PairwiseMetricModels:
    whisper: WhisperTranscriber
    speaker: ResemblyzerEmbedder
    accent: Optional[SpeechBrainEmbedder] = None
    emotion: Optional[Emotion2VecEmbedder] = None


def compute_pair_metrics(
    models: PairwiseMetricModels,
    *,
    src_transcript: str,
    target_wav_path: str,
    output_wav_path: str,
) -> dict[str, float]:
    pred_text, _lang = models.whisper.transcribe(output_wav_path)
    wer = word_error_rate(pred_text, src_transcript)

    spk_sim = _cosine(models.speaker.embed(target_wav_path), models.speaker.embed(output_wav_path))

    acc_sim = float("nan")
    if models.accent is not None:
        acc_sim = _cosine(models.accent.embed(target_wav_path), models.accent.embed(output_wav_path))

    emo_sim = float("nan")
    if models.emotion is not None:
        emo_sim = _cosine(models.emotion.embed(target_wav_path), models.emotion.embed(output_wav_path))

    return {
        "wer": float(wer) if math.isfinite(wer) else float("nan"),
        "s_sim": float(spk_sim) if math.isfinite(spk_sim) else float("nan"),
        "a_sim": float(acc_sim) if math.isfinite(acc_sim) else float("nan"),
        "e_sim": float(emo_sim) if math.isfinite(emo_sim) else float("nan"),
    }

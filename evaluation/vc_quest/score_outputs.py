# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from evaluation.vevo_live.common import (
    SpeakerSimilarityScorer,
    artifact_metrics_aligned,
    compute_wer_whisper,
    glitch_metrics,
    load_whisper,
    write_wav,
)


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


def _normalize_text_for_wer(text: str) -> str:
    for ch in [".", "'", "-", ",", "!", "?", "…", "，", "。", "！", "？", "\"", "“", "”"]:
        text = text.replace(ch, "")
    text = " ".join(text.strip().split())
    return text.lower()


def _word_error_rate(hyp_words: list[str], ref_words: list[str]) -> float:
    if not ref_words:
        return float("nan")

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


def _char_error_rate(hyp: str, ref: str) -> float:
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
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
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

    return float(dp[n, m]) / float(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a set of VC outputs (model-agnostic).")
    parser.add_argument("--ref_wav", type=str, required=True, help="Target/reference wav (for speaker similarity).")
    parser.add_argument("--src_wav", type=str, required=True, help="Source wav (for WER + artifacts).")
    parser.add_argument("--deg_wav", type=str, required=True, help="Converted wav to score.")
    parser.add_argument("--meta_json", type=str, default="", help="Optional meta JSON (to get delay_samples).")
    parser.add_argument("--out_json", type=str, required=True, help="Where to write report JSON.")
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--whisper_language", type=str, default="", help="Force Whisper language (e.g., fr).")
    parser.add_argument(
        "--src_text",
        type=str,
        default="",
        help="Optional transcript for src_wav. If provided, WER/CER are computed vs this text (single ASR pass).",
    )
    parser.add_argument("--similarity_model", type=str, default="wavlm", choices=["wavlm", "resemblyzer"])
    parser.add_argument("--similarity_device", type=str, default="", help="Optional torch device for similarity scorer.")
    args = parser.parse_args(argv)

    import torch

    if str(args.similarity_device).strip():
        sim_device = torch.device(str(args.similarity_device).strip())
    else:
        sim_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    whisper_model = load_whisper(args.whisper_model)
    speaker_scorer = SpeakerSimilarityScorer(
        model_name=args.similarity_model,  # type: ignore[arg-type]
        ref_wav_path=args.ref_wav,
        device=sim_device,
    )

    src, src_sr = _load_mono(args.src_wav)
    deg, deg_sr = _load_mono(args.deg_wav)

    delay_samples = 0
    if args.meta_json:
        meta = json.loads(Path(args.meta_json).read_text())
        delay_samples = int(meta.get("stats", {}).get("delay_samples", 0))
        stream_cfg = meta.get("config", {}).get("stream", {}) or {}
        hop_ms = int(stream_cfg.get("hop_ms") or 0)
    else:
        hop_ms = 0

    # Align on deg sample-rate for metrics that assume 1:1 time.
    src_rs = _resample_if_needed(src, src_sr, deg_sr)
    n = len(deg)
    src_trim = src_rs[delay_samples : delay_samples + n]
    if len(src_trim) < n:
        src_trim = np.pad(src_trim, (0, n - len(src_trim)), mode="constant")

    out_dir = Path(args.out_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_trim_path = out_dir / "src_aligned.wav"
    write_wav(str(ref_trim_path), src_trim, deg_sr)

    whisper_lang = str(args.whisper_language).strip() or None
    src_text = str(args.src_text).strip()
    wer = float("nan")
    cer = float("nan")
    deg_asr_text = ""
    wer_mode = "audio_ref"

    if src_text:
        wer_mode = "src_text"
        deg_asr = (
            whisper_model.transcribe(args.deg_wav, verbose=False, language=whisper_lang)
            if whisper_lang
            else whisper_model.transcribe(args.deg_wav, verbose=False)
        )
        deg_asr_text = _normalize_text_for_wer(deg_asr.get("text") or "")
        ref_text = _normalize_text_for_wer(src_text)
        wer = _word_error_rate(deg_asr_text.split(), ref_text.split())
        cer = _char_error_rate(deg_asr_text, ref_text)
    else:
        wer = compute_wer_whisper(
            whisper_model,
            audio_ref_path=str(ref_trim_path),
            audio_deg_path=args.deg_wav,
        )
    sim = speaker_scorer.score_paths([args.deg_wav])
    am = artifact_metrics_aligned(src_trim, deg, sample_rate=deg_sr)
    gm: dict[str, float] = {}
    if hop_ms > 0:
        hop_samples = int(round(float(hop_ms) / 1000.0 * float(deg_sr)))
        gm = glitch_metrics(deg, hop_samples=hop_samples)

    report: dict[str, Any] = {
        "speaker_similarity": float(sim),
        "wer_mode": str(wer_mode),
        "wer": float(wer) if np.isfinite(wer) else float("nan"),
        "cer": float(cer) if np.isfinite(cer) else float("nan"),
        "deg_asr_text": str(deg_asr_text),
        "delay_samples": int(delay_samples),
        "deg_sample_rate": int(deg_sr),
        **{f"artifact_{k}": float(v) for k, v in am.items()},
        **{f"glitch_{k}": float(v) for k, v in gm.items()},
        "paths": {
            "ref_wav": str(args.ref_wav),
            "src_wav": str(args.src_wav),
            "deg_wav": str(args.deg_wav),
            "src_aligned_wav": str(ref_trim_path),
            "meta_json": str(args.meta_json) if args.meta_json else "",
        },
    }

    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

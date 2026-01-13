#!/usr/bin/env python3
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

from evaluation.vevo_live.common import SpeakerSimilarityScorer, load_whisper


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _normalize_text_for_wer(text: str) -> str:
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
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
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


def _whisper_confidence_metrics(asr: dict[str, Any]) -> dict[str, float]:
    segs = asr.get("segments") or []
    if not isinstance(segs, list):
        segs = []

    def _collect(key: str) -> np.ndarray:
        vals: list[float] = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            try:
                v = float(seg.get(key))
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)
        return np.asarray(vals, dtype=np.float64)

    avg_logprob = _collect("avg_logprob")
    compression_ratio = _collect("compression_ratio")
    no_speech_prob = _collect("no_speech_prob")

    def _mean_or_nan(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float("nan")

    def _pct_or_nan(x: np.ndarray, q: float) -> float:
        return float(np.percentile(x, q)) if x.size else float("nan")

    return {
        "asr_num_segments": float(len(segs)),
        "asr_avg_logprob_mean": _mean_or_nan(avg_logprob),
        "asr_avg_logprob_p10": _pct_or_nan(avg_logprob, 10),
        "asr_compression_ratio_mean": _mean_or_nan(compression_ratio),
        "asr_compression_ratio_p90": _pct_or_nan(compression_ratio, 90),
        "asr_no_speech_prob_mean": _mean_or_nan(no_speech_prob),
        "asr_no_speech_prob_p90": _pct_or_nan(no_speech_prob, 90),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score non-aligned speech-to-speech outputs (e.g., ASR→TTS voice changer)."
    )
    parser.add_argument("--ref_wav", type=str, required=True)
    parser.add_argument("--src_wav", type=str, required=True)
    parser.add_argument("--deg_wav", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--whisper_language", type=str, default="")
    parser.add_argument(
        "--src_text",
        type=str,
        default="",
        help="Optional transcript for src_wav. If provided, WER/CER are computed vs this text (single ASR pass on deg).",
    )
    parser.add_argument(
        "--src_text_file",
        type=str,
        default="",
        help="Optional UTF-8 text file for src transcript (used if --src_text is empty).",
    )
    parser.add_argument("--similarity_model", type=str, default="wavlm", choices=["wavlm", "resemblyzer"])
    parser.add_argument("--similarity_device", type=str, default="")
    parser.add_argument("--meta_json", type=str, default="", help="Optional meta JSON to include in report.")
    args = parser.parse_args(argv)

    import torch

    whisper_model = load_whisper(str(args.whisper_model))
    whisper_lang = str(args.whisper_language).strip() or None

    # ASR
    deg_asr = (
        whisper_model.transcribe(str(args.deg_wav), verbose=False, language=whisper_lang)
        if whisper_lang
        else whisper_model.transcribe(str(args.deg_wav), verbose=False)
    )
    deg_text = _normalize_text_for_wer(str(deg_asr.get("text") or ""))

    src_text = str(args.src_text).strip()
    if not src_text:
        src_text_file = str(args.src_text_file).strip()
        if src_text_file:
            try:
                src_text = Path(src_text_file).read_text(encoding="utf-8").strip()
            except Exception:
                src_text = ""
    if src_text:
        ref_text = _normalize_text_for_wer(src_text)
    else:
        src_asr = (
            whisper_model.transcribe(str(args.src_wav), verbose=False, language=whisper_lang)
            if whisper_lang
            else whisper_model.transcribe(str(args.src_wav), verbose=False)
        )
        ref_text = _normalize_text_for_wer(str(src_asr.get("text") or ""))

    wer = _word_error_rate(deg_text.split(), ref_text.split())
    cer = _char_error_rate(deg_text, ref_text)
    asr_metrics = _whisper_confidence_metrics(deg_asr)

    # Speaker similarity
    if str(args.similarity_device).strip():
        device = torch.device(str(args.similarity_device).strip())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim_tgt = SpeakerSimilarityScorer(
        model_name=args.similarity_model,  # type: ignore[arg-type]
        ref_wav_path=str(args.ref_wav),
        device=device,
    ).score_paths([str(args.deg_wav)])

    sim_src = SpeakerSimilarityScorer(
        model_name=args.similarity_model,  # type: ignore[arg-type]
        ref_wav_path=str(args.src_wav),
        device=device,
    ).score_paths([str(args.deg_wav)])

    src_wav, src_sr = _load_mono(str(args.src_wav))
    deg_wav, deg_sr = _load_mono(str(args.deg_wav))
    src_dur = float(len(src_wav) / float(src_sr)) if src_sr > 0 else 0.0
    deg_dur = float(len(deg_wav) / float(deg_sr)) if deg_sr > 0 else 0.0
    dur_ratio = float(deg_dur / src_dur) if src_dur > 0 else float("nan")

    meta: dict[str, Any] = {}
    meta_path = str(args.meta_json).strip()
    if meta_path:
        try:
            meta = json.loads(Path(meta_path).read_text())
        except Exception:
            meta = {}

    report: dict[str, Any] = {
        **asr_metrics,
        "speaker_similarity_target": float(sim_tgt),
        "speaker_similarity_source": float(sim_src),
        "speaker_similarity_margin": float(sim_tgt - sim_src),
        "wer": float(wer) if np.isfinite(wer) else float("nan"),
        "cer": float(cer) if np.isfinite(cer) else float("nan"),
        "ref_text": str(ref_text),
        "deg_asr_text": str(deg_text),
        "src_dur_sec": float(src_dur),
        "deg_dur_sec": float(deg_dur),
        "dur_ratio": float(dur_ratio) if np.isfinite(dur_ratio) else float("nan"),
        "meta": meta,
        "paths": {
            "ref_wav": str(Path(args.ref_wav).resolve()),
            "src_wav": str(Path(args.src_wav).resolve()),
            "deg_wav": str(Path(args.deg_wav).resolve()),
            "meta_json": str(Path(meta_path).resolve()) if meta_path else "",
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

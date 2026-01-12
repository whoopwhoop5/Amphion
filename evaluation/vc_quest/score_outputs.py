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
    glitch_metrics,
    load_whisper,
    pitch_metrics_aligned,
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

    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr).astype(
        np.float32, copy=False
    )


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


def _clamp01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _score_lower_is_better(x: float, *, good: float, bad: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if good == bad:
        return 0.0
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return float((bad - x) / (bad - good))


def _score_higher_is_better(x: float, *, good: float, bad: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if good == bad:
        return 0.0
    if x >= good:
        return 1.0
    if x <= bad:
        return 0.0
    return float((x - bad) / (good - bad))


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


def _emit_start_ms(window_ms: int, hop_ms: int, emit_align: str) -> float:
    window_ms = int(window_ms)
    hop_ms = int(hop_ms)
    if window_ms <= 0 or hop_ms <= 0:
        return float("nan")

    align = str(emit_align)
    if align == "start":
        return 0.0
    if align == "center":
        return float(max(0, window_ms - hop_ms)) / 2.0
    if align == "end":
        return float(max(0, window_ms - hop_ms))
    return float("nan")


def _algo_delay_mid_ms(window_ms: int, hop_ms: int, emit_align: str) -> float:
    emit_start = _emit_start_ms(window_ms, hop_ms, emit_align)
    if not np.isfinite(emit_start):
        return float("nan")
    return float(window_ms) - (float(emit_start) + float(hop_ms) / 2.0)


def _call_score_v1(
    *,
    wer: float,
    speaker_similarity: float,
    silent_out_db_p95: float,
    dropout_frac_voiced: float,
    clip_frac: float,
    glitch_boundary_jump_ratio_p95: float,
    rtf_p95: float,
    latency_p95_ms: float,
) -> float:
    s_sim = _score_higher_is_better(speaker_similarity, good=0.97, bad=0.90)
    s_wer = _score_lower_is_better(wer, good=0.55, bad=1.05)

    s_silence = _score_lower_is_better(silent_out_db_p95, good=-40.0, bad=-25.0)
    s_dropout = _score_lower_is_better(dropout_frac_voiced, good=0.0, bad=0.01)
    s_clip = _score_lower_is_better(clip_frac, good=0.0, bad=0.001)

    s_glitch = _score_lower_is_better(glitch_boundary_jump_ratio_p95, good=2.0, bad=8.0)
    s_rtf = _score_lower_is_better(rtf_p95, good=0.85, bad=1.05)
    s_latency = _score_lower_is_better(latency_p95_ms, good=150.0, bad=500.0)

    quality = 0.55 * s_sim + 0.45 * s_wer
    stability = s_silence * s_dropout * s_clip

    return float(_clamp01(quality * stability * s_glitch * s_rtf * s_latency))


def _asr_confidence_score_v1(
    *,
    asr_avg_logprob_p10: float,
    asr_compression_ratio_p90: float,
) -> float:
    s_logprob = _score_higher_is_better(asr_avg_logprob_p10, good=-0.3, bad=-1.5)
    s_comp = _score_lower_is_better(asr_compression_ratio_p90, good=1.8, bad=2.4)
    return float(_clamp01(s_logprob * s_comp))


def _call_score_v2(
    *,
    wer: float,
    speaker_similarity: float,
    asr_avg_logprob_p10: float,
    asr_compression_ratio_p90: float,
    silent_out_db_p95: float,
    silence_leak_run_ms_p95: float,
    dropout_frac_voiced: float,
    dropout_run_ms_p95: float,
    clip_frac: float,
    glitch_boundary_jump_ratio_p95: float,
    rtf_p95: float,
    latency_p95_ms: float,
) -> float:
    s_sim = _score_higher_is_better(speaker_similarity, good=0.97, bad=0.90)
    s_wer = _score_lower_is_better(wer, good=0.55, bad=1.05)
    s_asr = _asr_confidence_score_v1(
        asr_avg_logprob_p10=asr_avg_logprob_p10,
        asr_compression_ratio_p90=asr_compression_ratio_p90,
    )

    s_silence = _score_lower_is_better(silent_out_db_p95, good=-40.0, bad=-25.0)
    s_leak_run = _score_lower_is_better(silence_leak_run_ms_p95, good=0.0, bad=250.0)
    s_dropout = _score_lower_is_better(dropout_frac_voiced, good=0.0, bad=0.01)
    s_dropout_run = _score_lower_is_better(dropout_run_ms_p95, good=0.0, bad=80.0)
    s_clip = _score_lower_is_better(clip_frac, good=0.0, bad=0.001)

    s_glitch = (
        _score_lower_is_better(glitch_boundary_jump_ratio_p95, good=2.0, bad=8.0)
        if np.isfinite(glitch_boundary_jump_ratio_p95)
        else 1.0
    )
    s_rtf = _score_lower_is_better(rtf_p95, good=0.85, bad=1.05)
    s_latency = _score_lower_is_better(latency_p95_ms, good=150.0, bad=1000.0)

    quality = 0.40 * s_sim + 0.35 * s_wer + 0.25 * s_asr
    stability = s_silence * s_leak_run * s_dropout * s_dropout_run * s_clip

    return float(_clamp01(quality * stability * s_glitch * s_rtf * s_latency))


def _ear_score_v2(
    *,
    wer: float,
    speaker_similarity: float,
    asr_avg_logprob_p10: float,
    asr_compression_ratio_p90: float,
    silent_out_db_p95: float,
    silence_leak_run_ms_p95: float,
    dropout_frac_voiced: float,
    dropout_run_ms_p95: float,
    clip_frac: float,
    glitch_boundary_jump_ratio_p95: float,
) -> float:
    s_sim = _score_higher_is_better(speaker_similarity, good=0.97, bad=0.90)
    s_wer = _score_lower_is_better(wer, good=0.55, bad=1.05)
    s_asr = _asr_confidence_score_v1(
        asr_avg_logprob_p10=asr_avg_logprob_p10,
        asr_compression_ratio_p90=asr_compression_ratio_p90,
    )

    s_silence = _score_lower_is_better(silent_out_db_p95, good=-40.0, bad=-25.0)
    s_leak_run = _score_lower_is_better(silence_leak_run_ms_p95, good=0.0, bad=250.0)
    s_dropout = _score_lower_is_better(dropout_frac_voiced, good=0.0, bad=0.01)
    s_dropout_run = _score_lower_is_better(dropout_run_ms_p95, good=0.0, bad=80.0)
    s_clip = _score_lower_is_better(clip_frac, good=0.0, bad=0.001)
    s_glitch = (
        _score_lower_is_better(glitch_boundary_jump_ratio_p95, good=2.0, bad=8.0)
        if np.isfinite(glitch_boundary_jump_ratio_p95)
        else 1.0
    )

    quality = 0.40 * s_sim + 0.35 * s_wer + 0.25 * s_asr
    stability = s_silence * s_leak_run * s_dropout * s_dropout_run * s_clip

    return float(_clamp01(quality * stability * s_glitch))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a set of VC outputs (model-agnostic)."
    )
    parser.add_argument(
        "--ref_wav",
        type=str,
        required=True,
        help="Target/reference wav (for speaker similarity).",
    )
    parser.add_argument(
        "--src_wav", type=str, required=True, help="Source wav (for WER + artifacts)."
    )
    parser.add_argument(
        "--deg_wav", type=str, required=True, help="Converted wav to score."
    )
    parser.add_argument(
        "--meta_json",
        type=str,
        default="",
        help="Optional meta JSON (to get delay_samples).",
    )
    parser.add_argument(
        "--out_json", type=str, required=True, help="Where to write report JSON."
    )
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument(
        "--whisper_language",
        type=str,
        default="",
        help="Force Whisper language (e.g., fr).",
    )
    parser.add_argument(
        "--src_text",
        type=str,
        default="",
        help="Optional transcript for src_wav. If provided, WER/CER are computed vs this text (single ASR pass).",
    )
    parser.add_argument(
        "--similarity_model",
        type=str,
        default="wavlm",
        choices=["wavlm", "resemblyzer"],
    )
    parser.add_argument(
        "--similarity_device",
        type=str,
        default="",
        help="Optional torch device for similarity scorer.",
    )
    parser.add_argument(
        "--pitch_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute extra pitch preservation metrics (slower).",
    )
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
    meta: dict[str, Any] = {}
    stream_cfg: dict[str, Any] = {}
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
    asr_metrics: dict[str, float] = {}
    wer_mode = "audio_ref"

    if src_text:
        wer_mode = "src_text"
        deg_asr = (
            whisper_model.transcribe(args.deg_wav, verbose=False, language=whisper_lang)
            if whisper_lang
            else whisper_model.transcribe(args.deg_wav, verbose=False)
        )
        deg_asr_text = _normalize_text_for_wer(str(deg_asr.get("text") or ""))
        ref_text = _normalize_text_for_wer(src_text)
        wer = _word_error_rate(deg_asr_text.split(), ref_text.split())
        cer = _char_error_rate(deg_asr_text, ref_text)
        asr_metrics = _whisper_confidence_metrics(deg_asr)
    else:
        ref_asr = (
            whisper_model.transcribe(
                str(ref_trim_path), verbose=False, language=whisper_lang
            )
            if whisper_lang
            else whisper_model.transcribe(str(ref_trim_path), verbose=False)
        )
        lang = ref_asr.get("language")
        deg_asr = (
            whisper_model.transcribe(args.deg_wav, verbose=False, language=lang)
            if lang
            else whisper_model.transcribe(args.deg_wav, verbose=False)
        )
        ref_text = _normalize_text_for_wer(str(ref_asr.get("text") or ""))
        deg_asr_text = _normalize_text_for_wer(str(deg_asr.get("text") or ""))
        wer = _word_error_rate(deg_asr_text.split(), ref_text.split())
        cer = _char_error_rate(deg_asr_text, ref_text)
        asr_metrics = _whisper_confidence_metrics(deg_asr)
    sim = speaker_scorer.score_paths([args.deg_wav])
    am = artifact_metrics_aligned(src_trim, deg, sample_rate=deg_sr)
    gm: dict[str, float] = {}
    if hop_ms > 0:
        hop_samples = int(round(float(hop_ms) / 1000.0 * float(deg_sr)))
        gm = glitch_metrics(deg, hop_samples=hop_samples)

    pm: dict[str, float] = {}
    if bool(args.pitch_metrics):
        pm = pitch_metrics_aligned(src_trim, deg, sample_rate=deg_sr)

    speed_mean_window_sec = float(
        meta.get("stats", {}).get("mean_window_sec", 0.0) or 0.0
    )
    speed_p95_window_sec = float(
        meta.get("stats", {}).get("p95_window_sec", 0.0) or 0.0
    )
    hop_sec = float(hop_ms) / 1000.0 if hop_ms > 0 else 0.0
    rtf_mean = float(speed_mean_window_sec / hop_sec) if hop_sec > 0 else float("nan")
    rtf_p95 = float(speed_p95_window_sec / hop_sec) if hop_sec > 0 else float("nan")

    stream_window_ms = int(stream_cfg.get("window_ms") or 0)
    stream_emit_align = str(stream_cfg.get("emit_align") or "")
    algo_delay_mid_ms = _algo_delay_mid_ms(stream_window_ms, hop_ms, stream_emit_align)
    latency_p95_ms = (
        float(algo_delay_mid_ms + 1000.0 * speed_p95_window_sec)
        if np.isfinite(algo_delay_mid_ms)
        else float("nan")
    )

    call_score_v1 = _call_score_v1(
        wer=float(wer) if np.isfinite(wer) else float("nan"),
        speaker_similarity=float(sim),
        silent_out_db_p95=float(am.get("silent_out_db_p95", float("nan"))),
        dropout_frac_voiced=float(am.get("dropout_frac_voiced", float("nan"))),
        clip_frac=float(am.get("clip_frac", float("nan"))),
        glitch_boundary_jump_ratio_p95=float(
            gm.get("boundary_jump_ratio_p95", float("nan"))
        ),
        rtf_p95=float(rtf_p95),
        latency_p95_ms=float(latency_p95_ms),
    )

    asr_confidence_v1 = _asr_confidence_score_v1(
        asr_avg_logprob_p10=float(asr_metrics.get("asr_avg_logprob_p10", float("nan"))),
        asr_compression_ratio_p90=float(
            asr_metrics.get("asr_compression_ratio_p90", float("nan"))
        ),
    )

    call_score_v2 = _call_score_v2(
        wer=float(wer) if np.isfinite(wer) else float("nan"),
        speaker_similarity=float(sim),
        asr_avg_logprob_p10=float(asr_metrics.get("asr_avg_logprob_p10", float("nan"))),
        asr_compression_ratio_p90=float(
            asr_metrics.get("asr_compression_ratio_p90", float("nan"))
        ),
        silent_out_db_p95=float(am.get("silent_out_db_p95", float("nan"))),
        silence_leak_run_ms_p95=float(
            am.get("silence_leak_run_ms_p95", float("nan"))
        ),
        dropout_frac_voiced=float(am.get("dropout_frac_voiced", float("nan"))),
        dropout_run_ms_p95=float(am.get("dropout_run_ms_p95", float("nan"))),
        clip_frac=float(am.get("clip_frac", float("nan"))),
        glitch_boundary_jump_ratio_p95=float(
            gm.get("boundary_jump_ratio_p95", float("nan"))
        ),
        rtf_p95=float(rtf_p95),
        latency_p95_ms=float(latency_p95_ms),
    )

    ear_score_v2 = _ear_score_v2(
        wer=float(wer) if np.isfinite(wer) else float("nan"),
        speaker_similarity=float(sim),
        asr_avg_logprob_p10=float(asr_metrics.get("asr_avg_logprob_p10", float("nan"))),
        asr_compression_ratio_p90=float(
            asr_metrics.get("asr_compression_ratio_p90", float("nan"))
        ),
        silent_out_db_p95=float(am.get("silent_out_db_p95", float("nan"))),
        silence_leak_run_ms_p95=float(
            am.get("silence_leak_run_ms_p95", float("nan"))
        ),
        dropout_frac_voiced=float(am.get("dropout_frac_voiced", float("nan"))),
        dropout_run_ms_p95=float(am.get("dropout_run_ms_p95", float("nan"))),
        clip_frac=float(am.get("clip_frac", float("nan"))),
        glitch_boundary_jump_ratio_p95=float(
            gm.get("boundary_jump_ratio_p95", float("nan"))
        ),
    )

    report: dict[str, Any] = {
        **asr_metrics,
        "speaker_similarity": float(sim),
        "wer_mode": str(wer_mode),
        "wer": float(wer) if np.isfinite(wer) else float("nan"),
        "cer": float(cer) if np.isfinite(cer) else float("nan"),
        "deg_asr_text": str(deg_asr_text),
        "delay_samples": int(delay_samples),
        "deg_sample_rate": int(deg_sr),
        "speed_mean_window_sec": float(speed_mean_window_sec),
        "speed_p95_window_sec": float(speed_p95_window_sec),
        "rtf_mean": float(rtf_mean) if np.isfinite(rtf_mean) else float("nan"),
        "rtf_p95": float(rtf_p95) if np.isfinite(rtf_p95) else float("nan"),
        "algo_delay_mid_ms": float(algo_delay_mid_ms)
        if np.isfinite(algo_delay_mid_ms)
        else float("nan"),
        "latency_p95_ms": float(latency_p95_ms)
        if np.isfinite(latency_p95_ms)
        else float("nan"),
        "call_score_v1": float(call_score_v1),
        "asr_confidence_v1": float(asr_confidence_v1),
        "call_score_v2": float(call_score_v2),
        "ear_score_v2": float(ear_score_v2),
        **{f"artifact_{k}": float(v) for k, v in am.items()},
        **{f"glitch_{k}": float(v) for k, v in gm.items()},
        **{f"pitch_{k}": float(v) for k, v in pm.items()},
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

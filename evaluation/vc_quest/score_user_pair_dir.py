# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from evaluation.vevo_live.common import (
    artifact_metrics_aligned,
    compute_wer_whisper,
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
    speaker_similarity_target: float,
    speaker_similarity_source: float,
    silent_out_db_p95: float,
    dropout_frac_voiced: float,
    clip_frac: float,
    glitch_boundary_jump_ratio_p95: float,
    rtf_p95: float,
    latency_p95_ms: float,
) -> float:
    # Speaker: we want high similarity to target, and also a margin vs source.
    s_sim_tgt = _score_higher_is_better(speaker_similarity_target, good=0.97, bad=0.90)
    s_sim_margin = _score_higher_is_better(
        float(speaker_similarity_target - speaker_similarity_source),
        good=0.05,
        bad=-0.05,
    )

    s_wer = _score_lower_is_better(wer, good=0.55, bad=1.05)

    s_silence = _score_lower_is_better(silent_out_db_p95, good=-40.0, bad=-25.0)
    s_dropout = _score_lower_is_better(dropout_frac_voiced, good=0.0, bad=0.01)
    s_clip = _score_lower_is_better(clip_frac, good=0.0, bad=0.001)

    s_glitch = _score_lower_is_better(glitch_boundary_jump_ratio_p95, good=2.0, bad=8.0)
    s_rtf = _score_lower_is_better(rtf_p95, good=0.85, bad=1.05)
    s_latency = _score_lower_is_better(latency_p95_ms, good=150.0, bad=500.0)

    # Content + target voice quality.
    quality = 0.40 * s_wer + 0.40 * s_sim_tgt + 0.20 * s_sim_margin

    # Any single "call killer" should dominate.
    stability = s_silence * s_dropout * s_clip

    return float(_clamp01(quality * stability * s_glitch * s_rtf * s_latency))


class _WavLMSpeakerEmbedder:
    def __init__(self, *, device: str):
        import torch
        import torch.nn.functional as F
        import librosa
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self._torch = torch
        self._F = F
        self._librosa = librosa

        self.device = str(device)
        self._fe = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self._model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        ).to(torch.device(self.device))
        self._model.eval()
        self._cache: dict[str, Any] = {}

    def embed_path(self, path: str):
        import torch

        p = str(Path(path).resolve())
        if p in self._cache:
            return self._cache[p]

        wav, _ = self._librosa.load(p, sr=16000)
        inputs = self._fe([wav], padding=True, return_tensors="pt", sampling_rate=16000)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self._model(**inputs).embeddings[0]
            emb = self._F.normalize(emb, dim=0)
        self._cache[p] = emb
        return emb

    def cosine(self, a, b) -> float:
        return float(self._F.cosine_similarity(a, b, dim=0).detach().cpu().item())


def _iter_pairs(run_dir: Path) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for p in run_dir.glob("*_stream_w*_h*.wav"):
        name = p.name
        # Expect ..._stream_w{w}_h{h}.wav
        try:
            tag = name.split("_stream_w", 1)[1]
            w_s, rest = tag.split("_h", 1)
            h_s = rest.split(".", 1)[0]
            w = int(w_s)
            h = int(h_s)
        except Exception:
            continue
        pairs.add((w, h))
    return sorted(pairs)


def _score_one(
    *,
    whisper_model,
    spk: _WavLMSpeakerEmbedder,
    ref_wav: str,
    src_wav: str,
    deg_wav: Path,
    meta_json: Path,
    out_json: Path,
    whisper_language: Optional[str],
    pitch_metrics: bool,
) -> dict[str, Any]:
    meta = json.loads(meta_json.read_text()) if meta_json.exists() else {}
    delay_samples = int(meta.get("stats", {}).get("delay_samples", 0))
    stream_cfg = meta.get("config", {}).get("stream", {}) or {}
    hop_ms = int(stream_cfg.get("hop_ms") or 0)
    window_ms = int(stream_cfg.get("window_ms") or 0)
    emit_align = str(stream_cfg.get("emit_align") or "")

    src, src_sr = _load_mono(src_wav)
    deg, deg_sr = _load_mono(str(deg_wav))

    src_rs = _resample_if_needed(src, src_sr, deg_sr)
    n = len(deg)
    src_trim = src_rs[delay_samples : delay_samples + n]
    if len(src_trim) < n:
        src_trim = np.pad(src_trim, (0, n - len(src_trim)), mode="constant")

    # Write aligned src segment for WER.
    out_json.parent.mkdir(parents=True, exist_ok=True)
    src_aligned_path = out_json.parent / f"{deg_wav.stem}.src_aligned.wav"
    write_wav(str(src_aligned_path), src_trim, deg_sr)

    # WER/CER using audio_ref (aligned src) vs deg.
    wer = compute_wer_whisper(
        whisper_model, audio_ref_path=str(src_aligned_path), audio_deg_path=str(deg_wav)
    )
    ref_text = ""
    hyp_text = ""
    cer = float("nan")
    try:
        ref_asr = (
            whisper_model.transcribe(
                str(src_aligned_path), verbose=False, language=whisper_language
            )
            if whisper_language
            else whisper_model.transcribe(str(src_aligned_path), verbose=False)
        )
        lang = ref_asr.get("language")
        deg_asr = (
            whisper_model.transcribe(str(deg_wav), verbose=False, language=lang)
            if lang
            else whisper_model.transcribe(str(deg_wav), verbose=False)
        )
        ref_text = _normalize_text_for_wer(str(ref_asr.get("text") or ""))
        hyp_text = _normalize_text_for_wer(str(deg_asr.get("text") or ""))
        cer = _char_error_rate(hyp_text, ref_text)
    except Exception:
        pass

    # Speaker similarity: to target and to source.
    emb_ref = spk.embed_path(ref_wav)
    emb_src = spk.embed_path(src_wav)
    emb_deg = spk.embed_path(str(deg_wav))
    sim_tgt = spk.cosine(emb_deg, emb_ref)
    sim_src = spk.cosine(emb_deg, emb_src)
    sim_margin = float(sim_tgt - sim_src)

    # Artifacts.
    am = artifact_metrics_aligned(src_trim, deg, sample_rate=deg_sr)
    gm: dict[str, float] = {}
    if hop_ms > 0:
        hop_samples = int(round(float(hop_ms) / 1000.0 * float(deg_sr)))
        gm = glitch_metrics(deg, hop_samples=hop_samples)

    pm: dict[str, float] = {}
    if pitch_metrics:
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

    algo_delay_mid_ms = _algo_delay_mid_ms(window_ms, hop_ms, emit_align)
    latency_p95_ms = (
        float(algo_delay_mid_ms + 1000.0 * speed_p95_window_sec)
        if np.isfinite(algo_delay_mid_ms)
        else float("nan")
    )

    call_score_v1 = _call_score_v1(
        wer=float(wer) if np.isfinite(wer) else float("nan"),
        speaker_similarity_target=float(sim_tgt),
        speaker_similarity_source=float(sim_src),
        silent_out_db_p95=float(am.get("silent_out_db_p95", float("nan"))),
        dropout_frac_voiced=float(am.get("dropout_frac_voiced", float("nan"))),
        clip_frac=float(am.get("clip_frac", float("nan"))),
        glitch_boundary_jump_ratio_p95=float(
            gm.get("boundary_jump_ratio_p95", float("nan"))
        ),
        rtf_p95=float(rtf_p95),
        latency_p95_ms=float(latency_p95_ms),
    )

    report: dict[str, Any] = {
        "speaker_similarity_target": float(sim_tgt),
        "speaker_similarity_source": float(sim_src),
        "speaker_similarity_margin": float(sim_margin),
        "wer": float(wer) if np.isfinite(wer) else float("nan"),
        "cer": float(cer) if np.isfinite(cer) else float("nan"),
        "ref_asr_text": str(ref_text),
        "deg_asr_text": str(hyp_text),
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
        **{f"artifact_{k}": float(v) for k, v in am.items()},
        **{f"glitch_{k}": float(v) for k, v in gm.items()},
        **{f"pitch_{k}": float(v) for k, v in pm.items()},
        "paths": {
            "ref_wav": str(ref_wav),
            "src_wav": str(src_wav),
            "deg_wav": str(deg_wav),
            "src_aligned_wav": str(src_aligned_path),
            "meta_json": str(meta_json),
        },
    }

    out_json.write_text(json.dumps(report, indent=2))
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a FreeVC/Chatterbox user-pair run dir in one process (fast)."
    )
    parser.add_argument("--run_dir", type=str, required=True)

    parser.add_argument(
        "--ref_v5",
        type=str,
        default="assets/vevo_live/user/ref_v5_24k_10s.wav",
        help="Target reference for fr_to_v5 direction",
    )
    parser.add_argument(
        "--src_fr",
        type=str,
        default="assets/vevo_live/user/src_fr_female_24k.wav",
        help="Source for fr_to_v5 direction",
    )
    parser.add_argument(
        "--ref_fr",
        type=str,
        default="assets/vevo_live/user/ref_fr_female_24k_10s.wav",
        help="Target reference for v5_to_fr direction",
    )
    parser.add_argument(
        "--src_v5",
        type=str,
        default="assets/vevo_live/user/src_v5_24k.wav",
        help="Source for v5_to_fr direction",
    )

    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument(
        "--whisper_language",
        type=str,
        default="",
        help="Force Whisper language (e.g., fr). Defaults to auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device for WavLM speaker embedder (default: cuda if avail).",
    )
    parser.add_argument(
        "--pitch_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute extra pitch preservation metrics (slower).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip reports that already exist.",
    )
    args = parser.parse_args(argv)

    import torch

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # Load models once.
    whisper_model = load_whisper(str(args.whisper_model))
    whisper_lang = str(args.whisper_language).strip() or None

    device = str(args.device).strip() or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    spk = _WavLMSpeakerEmbedder(device=device)

    # Offline (if present).
    for stem, ref_wav, src_wav in [
        ("v5_to_fr_offline", str(args.ref_fr), str(args.src_v5)),
        ("fr_to_v5_offline", str(args.ref_v5), str(args.src_fr)),
    ]:
        deg = run_dir / f"{stem}.wav"
        meta = run_dir / f"{stem}.meta.json"
        rep = run_dir / f"{stem}.report.json"
        if not deg.exists() or not meta.exists():
            continue
        if bool(args.resume) and rep.exists():
            continue
        _score_one(
            whisper_model=whisper_model,
            spk=spk,
            ref_wav=ref_wav,
            src_wav=src_wav,
            deg_wav=deg,
            meta_json=meta,
            out_json=rep,
            whisper_language=whisper_lang,
            pitch_metrics=bool(args.pitch_metrics),
        )

    # Streaming grid outputs.
    pairs = _iter_pairs(run_dir)
    for w, h in pairs:
        for prefix, ref_wav, src_wav in [
            ("v5_to_fr", str(args.ref_fr), str(args.src_v5)),
            ("fr_to_v5", str(args.ref_v5), str(args.src_fr)),
        ]:
            deg = run_dir / f"{prefix}_stream_w{w}_h{h}.wav"
            meta = run_dir / f"{prefix}_stream_w{w}_h{h}.meta.json"
            rep = run_dir / f"{prefix}_stream_w{w}_h{h}.report.json"
            if not deg.exists() or not meta.exists():
                continue
            if bool(args.resume) and rep.exists():
                continue
            _score_one(
                whisper_model=whisper_model,
                spk=spk,
                ref_wav=ref_wav,
                src_wav=src_wav,
                deg_wav=deg,
                meta_json=meta,
                out_json=rep,
                whisper_language=whisper_lang,
                pitch_metrics=bool(args.pitch_metrics),
            )

    # Select best config.
    from evaluation.vc_quest.select_best import main as select_best

    select_best(["--run_dir", str(run_dir)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

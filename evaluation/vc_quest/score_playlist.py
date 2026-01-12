# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from evaluation.vevo_live.common import (
    artifact_metrics_aligned,
    glitch_metrics,
    load_whisper,
    pitch_metrics_aligned,
)
from evaluation.vc_quest.playlist import VCPlaylistManifest, load_vc_playlist_manifest


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


def _brief_case(case: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "case_id",
        "call_score_v1",
        "wer",
        "speaker_similarity_target",
        "latency_p95_ms",
        "rtf_p95",
        "artifact_dropout_frac_voiced",
        "artifact_silent_out_db_p95",
        "glitch_boundary_jump_ratio_p95",
        "paths",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if k in case:
            out[k] = case.get(k)
    return out


def _topk_cases(
    per_case: list[dict[str, Any]],
    *,
    key: str,
    k: int,
    reverse: bool,
) -> list[dict[str, Any]]:
    k = int(k)
    if k <= 0:
        return []

    def sort_key(case: dict[str, Any]) -> tuple[int, float]:
        v = float(case.get(key, float("nan")) or float("nan"))
        is_bad = 0 if np.isfinite(v) else 1
        if not np.isfinite(v):
            v = 0.0
        return (is_bad, v)

    ordered = sorted(per_case, key=sort_key, reverse=bool(reverse))
    out: list[dict[str, Any]] = []
    for c in ordered:
        v = float(c.get(key, float("nan")) or float("nan"))
        if not np.isfinite(v):
            continue
        out.append(_brief_case(c))
        if len(out) >= k:
            break
    return out


def _safe_copy(src: str, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def _export_worst_cases(
    *,
    worst_cases: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for group, cases in worst_cases.items():
        group_dir = out_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(cases):
            cid = str(c.get("case_id") or f"case_{i:02d}")
            case_dir = group_dir / f"{i:02d}_{cid}"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "report.json").write_text(json.dumps(c, indent=2))

            paths = c.get("paths") or {}
            src_wav = str(paths.get("src_wav") or "")
            tgt_wav = str(paths.get("tgt_wav") or "")
            deg_wav = str(paths.get("deg_wav") or "")
            meta_json = str(paths.get("meta_json") or "")

            ok = True
            ok = ok and (not src_wav or _safe_copy(src_wav, case_dir / "src.wav"))
            ok = ok and (not tgt_wav or _safe_copy(tgt_wav, case_dir / "tgt.wav"))
            ok = ok and (not deg_wav or _safe_copy(deg_wav, case_dir / "deg.wav"))
            ok = ok and (not meta_json or _safe_copy(meta_json, case_dir / "meta.json"))

            entry = {"group": group, "case_id": cid, "dir": str(case_dir)}
            if ok:
                exported.append(entry)
            else:
                missing.append(entry)

    return {"out_dir": str(out_dir), "exported": exported, "missing": missing}


@dataclass
class _WavLMSpeakerEmbedder:
    device: str

    def __post_init__(self) -> None:
        import torch
        import torch.nn.functional as F
        import librosa
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self._torch = torch
        self._F = F
        self._librosa = librosa
        self._fe = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self._model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        ).to(torch.device(self.device))
        self._model.eval()
        self._cache: dict[str, Any] = {}

    def embed_path(self, path: str) -> Any:
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

    def cosine(self, a: Any, b: Any) -> float:
        return float(self._F.cosine_similarity(a, b, dim=0).detach().cpu().item())


def _case_id(source_id: str, target_id: str) -> str:
    return f"{source_id}__to__{target_id}"


def _aggregate(values: list[float]) -> dict[str, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
    }


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
    # Approximate delay of the emitted region midpoint relative to the window end.
    emit_start = _emit_start_ms(window_ms, hop_ms, emit_align)
    if not np.isfinite(emit_start):
        return float("nan")
    return float(window_ms) - (float(emit_start) + float(hop_ms) / 2.0)


def _call_score_v1(
    *,
    wer: float,
    speaker_similarity_target: float,
    silent_out_db_p95: float,
    dropout_frac_voiced: float,
    clip_frac: float,
    glitch_boundary_jump_ratio_p95: float,
    rtf_p95: float,
    latency_p95_ms: float,
) -> float:
    # Designed for live-call UX: strongly penalize noise, dropouts, clipping, lag.
    s_sim = _score_higher_is_better(speaker_similarity_target, good=0.97, bad=0.90)
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a playlist run (many pairs) with deterministic metrics."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Directory with wavs/ + meta/ from a playlist run.",
    )
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--whisper_language", type=str, default="fr")
    parser.add_argument(
        "--wer_mode",
        type=str,
        default="transcript",
        choices=["transcript", "audio_ref"],
        help=(
            "How to compute WER/CER: 'transcript' uses the dataset transcript; "
            "'audio_ref' transcribes the aligned source audio (more robust to streaming delay)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device for WavLM embedder (default: cuda if avail).",
    )
    parser.add_argument(
        "--write_case_reports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-case JSON reports under run_dir/reports/.",
    )
    parser.add_argument(
        "--worst_cases_k",
        type=int,
        default=10,
        help="Number of worst cases to surface per category (0 disables).",
    )
    parser.add_argument(
        "--export_worst_cases_dir",
        type=str,
        default="",
        help=(
            "Optional directory to export worst-case bundles (deg/src/tgt/meta/report) "
            "for quick listening. If relative, it is created under --run_dir."
        ),
    )
    parser.add_argument(
        "--pitch_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute extra pitch preservation metrics (slower).",
    )
    args = parser.parse_args(argv)

    import torch

    manifest: VCPlaylistManifest = load_vc_playlist_manifest(
        args.manifest
    ).resolve_paths(args.manifest)
    sources = manifest.sources_by_id()
    targets = manifest.targets_by_id()

    run_dir = Path(args.run_dir)
    wav_dir = run_dir / "wavs"
    meta_dir = run_dir / "meta"
    rep_dir = run_dir / "reports"
    if bool(args.write_case_reports):
        rep_dir.mkdir(parents=True, exist_ok=True)

    whisper_model = load_whisper(str(args.whisper_model))
    whisper_lang = str(args.whisper_language).strip() or None
    wer_mode = str(args.wer_mode)

    device = str(args.device).strip() or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    spk = _WavLMSpeakerEmbedder(device=device)

    asr_cache_dir = run_dir / "asr_cache"
    if wer_mode == "audio_ref":
        asr_cache_dir.mkdir(parents=True, exist_ok=True)
    src_asr_cache: dict[tuple[str, int, int, int], str] = {}

    per_case: list[dict[str, Any]] = []
    missing: list[str] = []

    for pair in manifest.pairs:
        s = sources[pair.source_id]
        t = targets[pair.target_id]
        cid = _case_id(pair.source_id, pair.target_id)

        deg_wav = wav_dir / f"{cid}.wav"
        meta_json = meta_dir / f"{cid}.json"
        if not deg_wav.exists() or not meta_json.exists():
            missing.append(cid)
            continue

        meta = json.loads(meta_json.read_text())
        delay_samples = int(meta.get("stats", {}).get("delay_samples", 0))
        stream_cfg = meta.get("config", {}).get("stream", {}) or {}
        hop_ms = int(stream_cfg.get("hop_ms") or 0)

        src, src_sr = _load_mono(s.wav_path)
        deg, deg_sr = _load_mono(str(deg_wav))

        src_rs = _resample_if_needed(src, src_sr, deg_sr)
        n = len(deg)
        src_trim = src_rs[delay_samples : delay_samples + n]
        if len(src_trim) < n:
            src_trim = np.pad(src_trim, (0, n - len(src_trim)), mode="constant")

        am = artifact_metrics_aligned(src_trim, deg, sample_rate=deg_sr)
        gm: dict[str, float] = {}
        if hop_ms > 0:
            hop_samples = int(round(float(hop_ms) / 1000.0 * float(deg_sr)))
            gm = glitch_metrics(deg, hop_samples=hop_samples)

        pm: dict[str, float] = {}
        if bool(args.pitch_metrics):
            pm = pitch_metrics_aligned(src_trim, deg, sample_rate=deg_sr)

        # Content: WER/CER (single ASR pass on deg).
        deg_asr = (
            whisper_model.transcribe(str(deg_wav), verbose=False, language=whisper_lang)
            if whisper_lang
            else whisper_model.transcribe(str(deg_wav), verbose=False)
        )
        hyp_text = _normalize_text_for_wer(str(deg_asr.get("text") or ""))

        if wer_mode == "audio_ref":
            cache_key = (str(pair.source_id), int(delay_samples), int(n), int(deg_sr))
            ref_text = src_asr_cache.get(cache_key, "")
            if not ref_text:
                wav_key = f"{pair.source_id}_d{delay_samples}_n{n}_sr{deg_sr}"
                ref_wav_path = asr_cache_dir / f"{wav_key}.wav"
                ref_txt_path = asr_cache_dir / f"{wav_key}.txt"
                if ref_txt_path.exists():
                    ref_text = ref_txt_path.read_text().strip()
                else:
                    from evaluation.vevo_live.common import write_wav

                    write_wav(str(ref_wav_path), src_trim, deg_sr)
                    ref_asr = (
                        whisper_model.transcribe(
                            str(ref_wav_path), verbose=False, language=whisper_lang
                        )
                        if whisper_lang
                        else whisper_model.transcribe(str(ref_wav_path), verbose=False)
                    )
                    ref_text = _normalize_text_for_wer(str(ref_asr.get("text") or ""))
                    ref_txt_path.write_text(ref_text)
                src_asr_cache[cache_key] = ref_text
        else:
            ref_text = _normalize_text_for_wer(str(s.transcript))

        wer = _word_error_rate(hyp_text.split(), ref_text.split())
        cer = _char_error_rate(hyp_text, ref_text)

        # Speaker: compare to target + to source, report margin.
        emb_tgt = spk.embed_path(t.wav_path)
        emb_src = spk.embed_path(s.wav_path)
        emb_deg = spk.embed_path(str(deg_wav))
        sim_tgt = spk.cosine(emb_deg, emb_tgt)
        sim_src = spk.cosine(emb_deg, emb_src)
        sim_margin = float(sim_tgt - sim_src)

        speed_mean_window_sec = float(
            meta.get("stats", {}).get("mean_window_sec", 0.0) or 0.0
        )
        speed_p95_window_sec = float(
            meta.get("stats", {}).get("p95_window_sec", 0.0) or 0.0
        )
        hop_sec = float(hop_ms) / 1000.0 if hop_ms > 0 else 0.0
        rtf_p95 = float(speed_p95_window_sec / hop_sec) if hop_sec > 0 else float("nan")

        stream_window_ms = int(stream_cfg.get("window_ms") or 0)
        stream_emit_align = str(stream_cfg.get("emit_align") or "")
        algo_delay_mid_ms = _algo_delay_mid_ms(
            stream_window_ms, hop_ms, stream_emit_align
        )
        latency_p95_ms = (
            float(algo_delay_mid_ms + 1000.0 * speed_p95_window_sec)
            if np.isfinite(algo_delay_mid_ms)
            else float("nan")
        )

        call_score_v1 = _call_score_v1(
            wer=float(wer) if np.isfinite(wer) else float("nan"),
            speaker_similarity_target=float(sim_tgt),
            silent_out_db_p95=float(am.get("silent_out_db_p95", float("nan"))),
            dropout_frac_voiced=float(am.get("dropout_frac_voiced", float("nan"))),
            clip_frac=float(am.get("clip_frac", float("nan"))),
            glitch_boundary_jump_ratio_p95=float(
                gm.get("boundary_jump_ratio_p95", float("nan"))
            ),
            rtf_p95=float(rtf_p95),
            latency_p95_ms=float(latency_p95_ms),
        )

        case = {
            "case_id": cid,
            "source_id": pair.source_id,
            "target_id": pair.target_id,
            "speaker_similarity_target": float(sim_tgt),
            "speaker_similarity_source": float(sim_src),
            "speaker_similarity_margin": float(sim_margin),
            "wer": float(wer) if np.isfinite(wer) else float("nan"),
            "cer": float(cer) if np.isfinite(cer) else float("nan"),
            "delay_samples": int(delay_samples),
            "deg_sample_rate": int(deg_sr),
            "speed_mean_window_sec": float(speed_mean_window_sec),
            "speed_p95_window_sec": float(speed_p95_window_sec),
            "rtf_mean": float(speed_mean_window_sec / hop_sec)
            if hop_sec > 0
            else float("nan"),
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
                "src_wav": str(s.wav_path),
                "tgt_wav": str(t.wav_path),
                "deg_wav": str(deg_wav),
                "meta_json": str(meta_json),
            },
        }
        per_case.append(case)

        if bool(args.write_case_reports):
            (rep_dir / f"{cid}.json").write_text(json.dumps(case, indent=2))

    # Aggregate.
    metrics: dict[str, list[float]] = {}
    for c in per_case:
        for k, v in c.items():
            if isinstance(v, (int, float)):
                metrics.setdefault(k, []).append(float(v))

    summary = {
        "meta": {
            "manifest": str(Path(args.manifest).resolve()),
            "run_dir": str(run_dir.resolve()),
            "whisper_model": str(args.whisper_model),
            "whisper_language": str(args.whisper_language),
            "wer_mode": wer_mode,
            "num_pairs_manifest": int(len(manifest.pairs)),
            "num_scored": int(len(per_case)),
            "num_missing": int(len(missing)),
            "missing_examples": missing[:10],
            **(manifest.meta or {}),
        },
        "aggregate": {k: _aggregate(vs) for k, vs in metrics.items()},
        "gates": {
            # Common gates for live-call suitability.
            "dropout_frac_voiced_gt_0p01": int(
                sum(
                    1
                    for c in per_case
                    if float(c.get("artifact_dropout_frac_voiced", 0.0) or 0.0) > 0.01
                )
            ),
            "silent_out_db_p95_gt_-25db": int(
                sum(
                    1
                    for c in per_case
                    if float(c.get("artifact_silent_out_db_p95", -200.0) or -200.0)
                    > -25.0
                )
            ),
            "clip_frac_gt_0p001": int(
                sum(
                    1
                    for c in per_case
                    if float(c.get("artifact_clip_frac", 0.0) or 0.0) > 0.001
                )
            ),
            "rtf_p95_gt_1": int(
                sum(1 for c in per_case if float(c.get("rtf_p95", 0.0) or 0.0) > 1.0)
            ),
            "latency_p95_ms_gt_300": int(
                sum(
                    1
                    for c in per_case
                    if float(c.get("latency_p95_ms", 0.0) or 0.0) > 300.0
                )
            ),
            "call_score_v1_lt_0p5": int(
                sum(
                    1
                    for c in per_case
                    if float(c.get("call_score_v1", 0.0) or 0.0) < 0.5
                )
            ),
        },
    }

    # Surface worst cases to speed up “human ear” spot-checking.
    worst_k = int(args.worst_cases_k)
    if worst_k > 0:
        worst_cases: dict[str, list[dict[str, Any]]] = {
            "lowest_call_score_v1": _topk_cases(
                per_case, key="call_score_v1", k=worst_k, reverse=False
            ),
            "highest_wer": _topk_cases(per_case, key="wer", k=worst_k, reverse=True),
            "lowest_speaker_similarity_target": _topk_cases(
                per_case, key="speaker_similarity_target", k=worst_k, reverse=False
            ),
            "highest_dropout_frac_voiced": _topk_cases(
                per_case, key="artifact_dropout_frac_voiced", k=worst_k, reverse=True
            ),
            "highest_silent_out_db_p95": _topk_cases(
                per_case, key="artifact_silent_out_db_p95", k=worst_k, reverse=True
            ),
            "highest_glitch_boundary_jump_ratio_p95": _topk_cases(
                per_case, key="glitch_boundary_jump_ratio_p95", k=worst_k, reverse=True
            ),
        }
        summary["worst_cases"] = worst_cases

        export_dir_s = str(args.export_worst_cases_dir).strip()
        if export_dir_s:
            export_dir = Path(export_dir_s)
            if not export_dir.is_absolute():
                export_dir = run_dir / export_dir
            summary["worst_cases_export"] = _export_worst_cases(
                worst_cases=worst_cases,
                out_dir=export_dir,
            )

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

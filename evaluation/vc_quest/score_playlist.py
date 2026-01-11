# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from evaluation.vevo_live.common import artifact_metrics_aligned, glitch_metrics, load_whisper
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
        self._fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self._model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(
            torch.device(self.device)
        )
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score a playlist run (many pairs) with deterministic metrics.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True, help="Directory with wavs/ + meta/ from a playlist run.")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--whisper_language", type=str, default="fr")
    parser.add_argument("--device", type=str, default="", help="Torch device for WavLM embedder (default: cuda if avail).")
    parser.add_argument(
        "--write_case_reports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-case JSON reports under run_dir/reports/.",
    )
    args = parser.parse_args(argv)

    import torch

    manifest: VCPlaylistManifest = load_vc_playlist_manifest(args.manifest).resolve_paths(args.manifest)
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

    device = str(args.device).strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
    spk = _WavLMSpeakerEmbedder(device=device)

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

        # Content: transcript-based WER/CER (single ASR pass on deg).
        ref_text = _normalize_text_for_wer(str(s.transcript))
        deg_asr = (
            whisper_model.transcribe(str(deg_wav), verbose=False, language=whisper_lang)
            if whisper_lang
            else whisper_model.transcribe(str(deg_wav), verbose=False)
        )
        hyp_text = _normalize_text_for_wer(str(deg_asr.get("text") or ""))
        wer = _word_error_rate(hyp_text.split(), ref_text.split())
        cer = _char_error_rate(hyp_text, ref_text)

        # Speaker: compare to target + to source, report margin.
        emb_tgt = spk.embed_path(t.wav_path)
        emb_src = spk.embed_path(s.wav_path)
        emb_deg = spk.embed_path(str(deg_wav))
        sim_tgt = spk.cosine(emb_deg, emb_tgt)
        sim_src = spk.cosine(emb_deg, emb_src)
        sim_margin = float(sim_tgt - sim_src)

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
            "speed_mean_window_sec": float(meta.get("stats", {}).get("mean_window_sec", 0.0) or 0.0),
            "speed_p95_window_sec": float(meta.get("stats", {}).get("p95_window_sec", 0.0) or 0.0),
            **{f"artifact_{k}": float(v) for k, v in am.items()},
            **{f"glitch_{k}": float(v) for k, v in gm.items()},
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
                sum(1 for c in per_case if float(c.get("artifact_dropout_frac_voiced", 0.0) or 0.0) > 0.01)
            ),
            "silent_out_db_p95_gt_-25db": int(
                sum(1 for c in per_case if float(c.get("artifact_silent_out_db_p95", -200.0) or -200.0) > -25.0)
            ),
            "clip_frac_gt_0p001": int(
                sum(1 for c in per_case if float(c.get("artifact_clip_frac", 0.0) or 0.0) > 0.001)
            ),
        },
    }

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


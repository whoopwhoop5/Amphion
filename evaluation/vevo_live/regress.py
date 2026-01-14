# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from evaluation.vevo_live.common import (
    EvalConfig,
    SpeakerSimilarityScorer,
    VevoInferenceConfig,
    VevoStreamingConfig,
    artifact_metrics_aligned,
    compute_content_similarity_hubert,
    compute_wer_whisper,
    glitch_metrics,
    list_wavs,
    load_whisper,
    simulate_streaming,
    write_wav,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vevo live VC regression suite.")
    parser.add_argument("--config_json", type=str, required=True, help="EvalConfig JSON (as produced by search).")
    parser.add_argument("--reference_wav", type=str, required=True)
    parser.add_argument(
        "--reference_max_sec",
        type=float,
        default=10.0,
        help="Trim reference audio to at most this many seconds (0 to disable).",
    )
    parser.add_argument("--playlist_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs/vevo_live/regress")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")

    parser.add_argument("--whisper_model", type=str, default="base")
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
        help="Optional torch device for speaker similarity scoring (e.g. cpu). Default: wavlm->converter device, resemblyzer->cpu.",
    )

    parser.add_argument(
        "--eval_seconds",
        type=float,
        default=4.0,
        help="Approx. evaluated output seconds per file (kept constant across hop sizes).",
    )
    parser.add_argument("--max_files", type=int, default=6, help="Max playlist wavs to evaluate.")

    # Defaults tuned for VC (Whisper WER can be unreliable on short, voice-converted chunks).
    parser.add_argument("--min_similarity", type=float, default=0.70)
    parser.add_argument("--min_content_hubert", type=float, default=0.80)
    parser.add_argument("--max_wer", type=float, default=0.90)
    parser.add_argument("--max_click_p95", type=float, default=5.0)
    parser.add_argument("--max_silent_out_db_p95", type=float, default=-35.0)
    parser.add_argument("--max_dropout_frac_voiced", type=float, default=0.02)
    parser.add_argument("--max_clip_frac", type=float, default=0.001)
    args = parser.parse_args(argv)

    cfg_raw = json.loads(Path(args.config_json).read_text())
    cfg = EvalConfig(
        inference=VevoInferenceConfig(**cfg_raw["inference"]),
        streaming=VevoStreamingConfig(**cfg_raw["streaming"]),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(args.playlist_dir)
    whisper_model = load_whisper(args.whisper_model)

    from models.vc.vevo.runner import VevoConverter

    converter = VevoConverter.from_pretrained(
        kind=cfg.inference.kind,  # type: ignore[arg-type]
        repo_cache_dir=args.repo_cache_dir,
    )
    import torch

    if str(args.similarity_device).strip():
        sim_device = torch.device(str(args.similarity_device).strip())
    else:
        sim_device = converter.device if args.similarity_model == "wavlm" else torch.device("cpu")
    speaker_scorer = SpeakerSimilarityScorer(
        model_name=args.similarity_model,  # type: ignore[arg-type]
        ref_wav_path=args.reference_wav,
        device=sim_device,
    )

    deg_dir = out_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)

    wers = []
    content_cos = []
    click_p95s = []
    silent_out_p95s = []
    dropout_fracs = []
    clip_fracs = []
    for wav in wavs[: args.max_files]:
        hop_sec = max(float(cfg.streaming.hop_ms) / 1000.0, 1e-9)
        max_hops = max(1, int(round(float(args.eval_seconds) / hop_sec)))
        out_wav, sr, stream_stats = simulate_streaming(
            converter,
            reference_wav_path=args.reference_wav,
            source_wav_path=wav,
            cfg=cfg,
            max_hops=max_hops,
            reference_max_sec=float(args.reference_max_sec),
        )
        out_path = deg_dir / (Path(wav).stem + ".wav")
        write_wav(str(out_path), out_wav, sr)

        # Trim reference to output duration for WER.
        src_wav, src_sr = sf.read(wav, dtype="float32")
        if src_wav.ndim > 1:
            src_wav = src_wav[:, 0]
        if int(src_sr) != sr:
            raise ValueError(f"Expected {sr}Hz wav in playlist, got {src_sr}Hz: {wav}")
        n = len(out_wav)
        delay_samples = int(stream_stats.get("delay_samples", 0))
        src_trim = np.asarray(src_wav).reshape(-1)[delay_samples : delay_samples + n]
        if len(src_trim) < n:
            src_trim = np.pad(src_trim, (0, n - len(src_trim)), mode="constant")
        ref_trim_path = out_dir / "ref_trim" / (Path(wav).stem + ".wav")
        write_wav(str(ref_trim_path), src_trim, sr)

        wer = compute_wer_whisper(
            whisper_model,
            audio_ref_path=str(ref_trim_path),
            audio_deg_path=str(out_path),
        )
        wers.append(wer)

        content_cos.append(
            compute_content_similarity_hubert(
                converter,
                src_wav=np.asarray(src_trim, dtype=np.float32).reshape(-1),
                deg_wav=np.asarray(out_wav, dtype=np.float32).reshape(-1),
                sample_rate=sr,
            )
        )

        gm = glitch_metrics(
            np.asarray(out_wav).reshape(-1),
            hop_samples=int(round(cfg.streaming.hop_ms / 1000 * sr)),
            sample_rate=int(sr),
        )
        click_p95s.append(gm["boundary_jump_ratio_p95"])

        am = artifact_metrics_aligned(
            np.asarray(src_trim, dtype=np.float32).reshape(-1),
            np.asarray(out_wav, dtype=np.float32).reshape(-1),
            sample_rate=sr,
        )
        if np.isfinite(am.get("silent_out_db_p95", float("nan"))):
            silent_out_p95s.append(float(am["silent_out_db_p95"]))
        if np.isfinite(am.get("dropout_frac_voiced", float("nan"))):
            dropout_fracs.append(float(am["dropout_frac_voiced"]))
        clip_fracs.append(float(am.get("clip_frac", 0.0)))

    sim = speaker_scorer.score_dir(str(deg_dir))
    wer = float(np.mean([w for w in wers if np.isfinite(w)])) if any(np.isfinite(w) for w in wers) else 1.0
    content = (
        float(np.mean([c for c in content_cos if np.isfinite(c)]))
        if any(np.isfinite(c) for c in content_cos)
        else float("nan")
    )
    click_p95 = float(np.mean(click_p95s)) if click_p95s else 0.0
    silent_out_db_p95 = float(np.mean(silent_out_p95s)) if silent_out_p95s else float("nan")
    dropout_frac_voiced = float(np.mean(dropout_fracs)) if dropout_fracs else float("nan")
    clip_frac = float(np.mean(clip_fracs)) if clip_fracs else 0.0

    report = {
        "similarity": sim,
        "content_hubert_cos": content,
        "wer": wer,
        "click_p95": click_p95,
        "artifact_silent_out_db_p95": silent_out_db_p95,
        "artifact_dropout_frac_voiced": dropout_frac_voiced,
        "artifact_clip_frac": clip_frac,
        "config": cfg_raw,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    ok = True
    if sim < args.min_similarity:
        ok = False
    if np.isfinite(content) and content < args.min_content_hubert:
        ok = False
    if wer > args.max_wer:
        ok = False
    if click_p95 > args.max_click_p95:
        ok = False
    if np.isfinite(silent_out_db_p95) and silent_out_db_p95 > args.max_silent_out_db_p95:
        ok = False
    if np.isfinite(dropout_frac_voiced) and dropout_frac_voiced > args.max_dropout_frac_voiced:
        ok = False
    if clip_frac > args.max_clip_frac:
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

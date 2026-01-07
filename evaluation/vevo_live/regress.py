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
    VevoInferenceConfig,
    VevoStreamingConfig,
    compute_speaker_similarity,
    compute_wer_whisper,
    glitch_metrics,
    list_wavs,
    load_whisper,
    simulate_streaming,
    write_wav,
)
from models.vc.vevo.runner import VevoConverter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vevo live VC regression suite.")
    parser.add_argument("--config_json", type=str, required=True, help="EvalConfig JSON (as produced by search).")
    parser.add_argument("--reference_wav", type=str, required=True)
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

    # Conservative defaults; tune once you have baseline numbers.
    parser.add_argument("--min_similarity", type=float, default=0.20)
    parser.add_argument("--max_wer", type=float, default=0.55)
    parser.add_argument("--max_click_p95", type=float, default=50.0)
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

    converter = VevoConverter.from_pretrained(
        kind=cfg.inference.kind,  # type: ignore[arg-type]
        repo_cache_dir=args.repo_cache_dir,
    )

    deg_dir = out_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)

    wers = []
    click_p95s = []
    for wav in wavs[:2]:
        out_wav, sr, _ = simulate_streaming(
            converter,
            reference_wav_path=args.reference_wav,
            source_wav_path=wav,
            cfg=cfg,
            max_hops=4,
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
        src_trim = np.asarray(src_wav).reshape(-1)[:n]
        if len(src_trim) < n:
            src_trim = np.pad(src_trim, (0, n - len(src_trim)), mode="constant")
        ref_trim_path = out_dir / "ref_trim" / (Path(wav).stem + ".wav")
        write_wav(str(ref_trim_path), src_trim, sr)

        wers.append(
            compute_wer_whisper(
                whisper_model,
                audio_ref_path=str(ref_trim_path),
                audio_deg_path=str(out_path),
            )
        )
        out_loaded, _ = sf.read(out_path, dtype="float32")
        gm = glitch_metrics(np.asarray(out_loaded).reshape(-1), hop_samples=int(round(cfg.streaming.hop_ms / 1000 * sr)))
        click_p95s.append(gm["boundary_jump_ratio_p95"])

    sim = compute_speaker_similarity(
        ref_wav_path=args.reference_wav,
        deg_dir=str(deg_dir),
        model_name=args.similarity_model,  # type: ignore[arg-type]
    )

    wer = float(np.mean([w for w in wers if np.isfinite(w)])) if any(np.isfinite(w) for w in wers) else 1.0
    click_p95 = float(np.mean(click_p95s)) if click_p95s else 0.0

    report = {"similarity": sim, "wer": wer, "click_p95": click_p95, "config": cfg_raw}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    ok = True
    if sim < args.min_similarity:
        ok = False
    if wer > args.max_wer:
        ok = False
    if click_p95 > args.max_click_p95:
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

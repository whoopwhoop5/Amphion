# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

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


def score_result(metrics: dict[str, Any]) -> float:
    # Higher is better.
    sim = float(metrics.get("speaker_similarity", 0.0))
    wer = float(metrics.get("wer", 1.0))
    click = float(metrics.get("glitch_boundary_jump_ratio_p95", 0.0))
    window_sec = float(metrics.get("mean_window_sec", 9e9))

    # Penalize instability + slow configs. Weights chosen to be conservative; adjust as needed.
    return sim - 0.5 * wer - 0.02 * click - 0.05 * window_sec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic Vevo live VC autotune.")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument("--reference_wav", type=str, required=True)
    parser.add_argument("--playlist_dir", type=str, required=True, help="Folder of 24000Hz mono wavs.")
    parser.add_argument("--out_dir", type=str, default="runs/vevo_live")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")

    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--similarity_model", type=str, default="wavlm", choices=["wavlm", "rawnet", "resemblyzer"])

    parser.add_argument("--max_files", type=int, default=2)
    parser.add_argument("--max_hops_per_file", type=int, default=4)

    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min_improve", type=float, default=1e-4)
    args = parser.parse_args(argv)

    out_root = Path(args.out_dir) / "search"
    out_root.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(args.playlist_dir)[: args.max_files]
    whisper_model = load_whisper(args.whisper_model)

    converter = VevoConverter.from_pretrained(
        kind=args.kind,  # type: ignore[arg-type]
        repo_cache_dir=args.repo_cache_dir,
    )

    # Deterministic grid (small by default).
    flow_steps_grid = [8, 12, 16, 24, 32]
    hop_ms_grid = [1000, 500, 250] if args.kind == "vevotimbre" else [1000]
    fade_ms_grid = [0, 10, 20]

    results = []
    best = None
    best_score = -1e9
    stale = 0

    for flow_steps in flow_steps_grid:
        for hop_ms in hop_ms_grid:
            for fade_ms in fade_ms_grid:
                cfg = EvalConfig(
                    inference=VevoInferenceConfig(
                        kind=args.kind,  # type: ignore[arg-type]
                        flow_matching_steps=flow_steps,
                        seed=1234,
                    ),
                    streaming=VevoStreamingConfig(window_ms=1000, hop_ms=hop_ms, fade_ms=fade_ms),
                )

                cfg_uid = cfg.uid()
                cfg_dir = out_root / cfg_uid
                cfg_dir.mkdir(parents=True, exist_ok=True)

                per_file = []
                for wav in wavs:
                    out_wav, sr, stream_stats = simulate_streaming(
                        converter,
                        reference_wav_path=args.reference_wav,
                        source_wav_path=wav,
                        cfg=cfg,
                        max_hops=args.max_hops_per_file,
                    )
                    deg_dir = cfg_dir / "deg"
                    deg_dir.mkdir(parents=True, exist_ok=True)
                    out_path = deg_dir / (Path(wav).stem + ".wav")
                    write_wav(str(out_path), out_wav, sr)

                    per_file.append({"wav": wav, "out_wav": str(out_path), **stream_stats})

                # Folder-level metrics
                deg_dir = str(cfg_dir / "deg")
                sim = compute_speaker_similarity(
                    ref_wav_path=args.reference_wav,
                    deg_dir=deg_dir,
                    work_dir=str(cfg_dir / "work"),
                    model_name=args.similarity_model,  # type: ignore[arg-type]
                )

                wers = []
                clicks = []
                mean_window_secs = []
                for row in per_file:
                    wers.append(
                        compute_wer_whisper(
                            whisper_model,
                            audio_ref_path=row["wav"],
                            audio_deg_path=row["out_wav"],
                        )
                    )

                    out_loaded, out_sr = sf.read(row["out_wav"], dtype="float32")
                    if out_loaded.ndim > 1:
                        out_loaded = out_loaded[:, 0]
                    gm = glitch_metrics(
                        np.asarray(out_loaded).reshape(-1),
                        hop_samples=int(round(cfg.streaming.hop_ms / 1000 * out_sr)),
                    )
                    clicks.append(gm["boundary_jump_ratio_p95"])
                    mean_window_secs.append(float(row["mean_window_sec"]))

                metrics = {
                    "cfg_uid": cfg_uid,
                    "speaker_similarity": sim,
                    "wer": float(np.mean(wers)) if wers else 1.0,
                    "glitch_boundary_jump_ratio_p95": float(np.mean(clicks)) if clicks else 0.0,
                    "mean_window_sec": float(np.mean(mean_window_secs)) if mean_window_secs else 0.0,
                }

                metrics["score"] = score_result(metrics)
                record = {"config": asdict(cfg), "metrics": metrics, "files": per_file}
                (cfg_dir / "result.json").write_text(json.dumps(record, indent=2))

                results.append(record)
                if metrics["score"] > best_score + args.min_improve:
                    best_score = float(metrics["score"])
                    best = record
                    stale = 0
                    # Snapshot best audio artifacts.
                    best_dir = out_root / "best"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    best_dir.mkdir(parents=True, exist_ok=True)
                    (best_dir / "best_config.json").write_text(json.dumps(best["config"], indent=2))
                    (best_dir / "best_metrics.json").write_text(json.dumps(best["metrics"], indent=2))
                    shutil.copytree(cfg_dir / "deg", best_dir / "deg")
                else:
                    stale += 1

                print(
                    f"[search] cfg={cfg_uid} flow={flow_steps} hop={hop_ms} fade={fade_ms} "
                    f"sim={sim:.3f} wer={metrics['wer']:.3f} click={metrics['glitch_boundary_jump_ratio_p95']:.2f} "
                    f"win_s={metrics['mean_window_sec']:.2f} score={metrics['score']:.3f} best={best_score:.3f}",
                    flush=True,
                )

                if stale >= args.patience:
                    print("[search] Early stop: plateau reached.", flush=True)
                    break
            if stale >= args.patience:
                break
        if stale >= args.patience:
            break

    # Write overall summary deterministically.
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps({"best_score": best_score, "best": best, "all": results}, indent=2))
    print(f"[search] Wrote: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

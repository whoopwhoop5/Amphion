# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from evaluation.vevo_live.common import (
    EvalConfig,
    SpeakerSimilarityScorer,
    VevoInferenceConfig,
    VevoStreamingConfig,
    compute_content_similarity_hubert,
    compute_wer_whisper,
    glitch_metrics,
    list_wavs,
    load_mono,
    load_whisper,
    read_reference_wav_bytes,
    set_determinism,
    write_wav,
)
from models.vc.vevo.live_engine import AudioRingBuffer, VevoStreamingEngine
from models.vc.vevo.runner import VevoConverter


def _parse_int_grid(s: str) -> list[int]:
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("Empty grid.")
    return vals


def _bench_configs(
    converter: VevoConverter,
    *,
    reference_wav: str,
    reference_max_sec: float,
    source_wav: str,
    cfgs: list[EvalConfig],
    warmup_windows: int,
    measured_windows: int,
) -> dict[str, dict[str, float]]:
    """Measure per-window inference time for each config (converter-only, no Whisper/SV)."""

    engine = VevoStreamingEngine(converter)
    engine.prepare_reference_bytes(read_reference_wav_bytes(reference_wav, max_sec=reference_max_sec))

    src, sr = load_mono(source_wav)
    if sr != engine.model_sr:
        raise ValueError(f"Expected {engine.model_sr}Hz source wav, got {sr}Hz: {source_wav}")
    src = np.asarray(src, dtype=np.float32).reshape(-1)

    bench: dict[str, dict[str, float]] = {}
    for cfg in cfgs:
        set_determinism(cfg.inference.seed)
        window_samples = int(round(cfg.streaming.window_ms / 1000 * engine.model_sr))
        hop_samples = int(round(cfg.streaming.hop_ms / 1000 * engine.model_sr))
        if hop_samples <= 0 or window_samples <= 0 or hop_samples > window_samples:
            continue

        cfg_uid = cfg.uid()
        ring = AudioRingBuffer(window_samples)
        timings: list[float] = []

        need = warmup_windows + measured_windows
        window_count = 0
        pos = 0
        while window_count < need:
            # Loop the source audio if it's too short for large hop sizes.
            if len(src) == 0:
                hop = np.zeros(hop_samples, dtype=np.float32)
            else:
                parts: list[np.ndarray] = []
                remaining = hop_samples
                while remaining > 0:
                    take = min(remaining, len(src) - pos)
                    if take <= 0:
                        pos = 0
                        continue
                    parts.append(src[pos : pos + take])
                    pos += take
                    remaining -= take
                    if pos >= len(src):
                        pos = 0
                hop = np.concatenate(parts).astype(np.float32, copy=False)
            ring.write(hop)
            if ring.size < window_samples:
                continue

            window = ring.read_last(window_samples)
            t0 = time.time()
            _ = engine.convert_window(
                window,
                flow_matching_steps=cfg.inference.flow_matching_steps,
                diffusion_cfg=cfg.inference.diffusion_cfg,
                diffusion_rescale_cfg=cfg.inference.diffusion_rescale_cfg,
                seed=cfg.inference.seed + window_count,
            )
            dt = time.time() - t0

            if window_count >= warmup_windows:
                timings.append(float(dt))
            window_count += 1

        if len(timings) < measured_windows:
            continue

        hop_sec = float(hop_samples) / float(engine.model_sr)
        arr = np.asarray(timings, dtype=np.float64)
        mean_sec = float(np.mean(arr))
        p50_sec = float(np.percentile(arr, 50))
        p95_sec = float(np.percentile(arr, 95))

        bench[cfg_uid] = {
            "hop_sec": float(hop_sec),
            "mean_window_sec": mean_sec,
            "p50_window_sec": p50_sec,
            "p95_window_sec": p95_sec,
            "rtf_mean": mean_sec / max(hop_sec, 1e-9),
            "rtf_p95": p95_sec / max(hop_sec, 1e-9),
            # A practical proxy for how delayed it feels (hop scheduling + compute).
            "delay_proxy_p95_sec": float(hop_sec + p95_sec),
        }

        print(
            f"[bench] cfg={cfg_uid} win={cfg.streaming.window_ms} hop={cfg.streaming.hop_ms} steps={cfg.inference.flow_matching_steps} "
            f"mean={mean_sec:.3f}s p95={p95_sec:.3f}s rtf_p95={bench[cfg_uid]['rtf_p95']:.2f} delay_p95={bench[cfg_uid]['delay_proxy_p95_sec']:.2f}s",
            flush=True,
        )

    return bench


def _quality_eval_one(
    converter: VevoConverter,
    speaker_scorer: SpeakerSimilarityScorer,
    whisper_model,
    *,
    reference_wav: str,
    reference_max_sec: float,
    wavs: list[str],
    cfg: EvalConfig,
    eval_seconds: float,
    out_dir: Path,
) -> dict[str, Any]:
    from evaluation.vevo_live.common import simulate_streaming

    cfg_uid = cfg.uid()
    cfg_dir = out_dir / cfg_uid
    if cfg_dir.exists():
        shutil.rmtree(cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    deg_dir = cfg_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)
    ref_trim_dir = cfg_dir / "ref_trim"
    ref_trim_dir.mkdir(parents=True, exist_ok=True)

    per_file = []
    wers = []
    contents = []
    clicks = []
    mean_window_secs = []
    p95_window_secs = []

    for wav in wavs:
        hop_sec = max(float(cfg.streaming.hop_ms) / 1000.0, 1e-9)
        max_hops = max(1, int(round(float(eval_seconds) / hop_sec)))

        out_wav, sr, stream_stats = simulate_streaming(
            converter,
            reference_wav_path=reference_wav,
            source_wav_path=wav,
            cfg=cfg,
            max_hops=max_hops,
            reference_max_sec=float(reference_max_sec),
        )

        out_path = deg_dir / (Path(wav).stem + ".wav")
        write_wav(str(out_path), out_wav, sr)

        src_wav, src_sr = sf.read(wav, dtype="float32")
        if src_wav.ndim > 1:
            src_wav = src_wav[:, 0]
        if int(src_sr) != int(sr):
            raise ValueError(f"Expected {sr}Hz wav in playlist, got {src_sr}Hz: {wav}")

        n = int(len(out_wav))
        delay_samples = int(stream_stats.get("delay_samples", 0))
        src_trim = np.asarray(src_wav).reshape(-1)[delay_samples : delay_samples + n]
        if len(src_trim) < n:
            src_trim = np.pad(src_trim, (0, n - len(src_trim)), mode="constant")
        ref_trim_path = ref_trim_dir / (Path(wav).stem + ".wav")
        write_wav(str(ref_trim_path), src_trim, sr)

        wer = compute_wer_whisper(
            whisper_model,
            audio_ref_path=str(ref_trim_path),
            audio_deg_path=str(out_path),
        )
        wers.append(float(wer) if np.isfinite(wer) else float("nan"))

        content = compute_content_similarity_hubert(
            converter,
            src_wav=np.asarray(src_trim, dtype=np.float32).reshape(-1),
            deg_wav=np.asarray(out_wav, dtype=np.float32).reshape(-1),
            sample_rate=int(sr),
        )
        contents.append(float(content) if np.isfinite(content) else float("nan"))

        gm = glitch_metrics(
            np.asarray(out_wav).reshape(-1),
            hop_samples=int(round(cfg.streaming.hop_ms / 1000 * sr)),
        )
        clicks.append(float(gm["boundary_jump_ratio_p95"]))

        mean_window_secs.append(float(stream_stats.get("mean_window_sec", 0.0)))
        p95_window_secs.append(float(stream_stats.get("p95_window_sec", 0.0)))

        per_file.append(
            {
                "wav": wav,
                "ref_trim_wav": str(ref_trim_path),
                "out_wav": str(out_path),
                "out_samples": int(n),
                "wer": wers[-1],
                "content_hubert_cos": contents[-1],
                "glitch_boundary_jump_ratio_p95": clicks[-1],
                **stream_stats,
            }
        )

    sim = speaker_scorer.score_dir(str(deg_dir))
    valid_wers = [w for w in wers if np.isfinite(w)]
    valid_contents = [c for c in contents if np.isfinite(c)]

    metrics = {
        "speaker_similarity": float(sim),
        "wer": float(np.mean(valid_wers)) if valid_wers else float("nan"),
        "wer_valid_frac": float(len(valid_wers) / max(1, len(wers))),
        "content_hubert_cos": float(np.mean(valid_contents)) if valid_contents else float("nan"),
        "content_hubert_cos_valid_frac": float(len(valid_contents) / max(1, len(contents))),
        "glitch_boundary_jump_ratio_p95": float(np.mean(clicks)) if clicks else 0.0,
        "mean_window_sec": float(np.mean(mean_window_secs)) if mean_window_secs else 0.0,
        "p95_window_sec": float(np.max(p95_window_secs)) if p95_window_secs else 0.0,
    }

    hop_sec = max(float(cfg.streaming.hop_ms) / 1000.0, 1e-9)
    metrics["hop_sec"] = float(hop_sec)
    metrics["rtf_mean"] = float(metrics["mean_window_sec"]) / hop_sec
    metrics["rtf_p95"] = float(metrics["p95_window_sec"]) / hop_sec
    metrics["delay_proxy_p95_sec"] = float(hop_sec + float(metrics["p95_window_sec"]))

    # Same base score as search.py (quality-weighted).
    score = (
        0.60 * float(metrics["speaker_similarity"])
        + 0.40 * float(metrics["content_hubert_cos"] if np.isfinite(metrics["content_hubert_cos"]) else 0.0)
        - 0.80 * float(metrics["wer"] if np.isfinite(metrics["wer"]) else 1.0)
        - 0.02 * float(metrics["glitch_boundary_jump_ratio_p95"])
        - 0.05 * float(metrics["delay_proxy_p95_sec"])
    )
    metrics["score"] = float(score)

    record = {"config": asdict(cfg), "metrics": metrics, "files": per_file}
    (cfg_dir / "result.json").write_text(json.dumps(record, indent=2))
    return record


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Two-stage macOS tuner for Vevo live streaming configs.")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument("--device", type=str, default=None, help="torch device for Vevo (default: auto)")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")

    parser.add_argument("--reference_wav", type=str, required=True)
    parser.add_argument("--reference_max_sec", type=float, default=10.0)
    parser.add_argument("--playlist_dir", type=str, required=True, help="Folder of 24000Hz mono wavs.")
    parser.add_argument("--out_dir", type=str, default="runs/vevo_live/tune_macos")

    parser.add_argument("--eval_seconds", type=float, default=6.0)
    parser.add_argument("--max_files", type=int, default=10)

    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--similarity_model", type=str, default="resemblyzer", choices=["wavlm", "resemblyzer"])
    parser.add_argument("--similarity_device", type=str, default="cpu")

    parser.add_argument("--flow_steps_grid", type=str, default="4,6,8,12")
    parser.add_argument("--window_ms_grid", type=str, default="1500,2000")
    parser.add_argument(
        "--hop_ms_grid",
        type=str,
        default="1000,1500,2000",
        help="Comma-separated hop sizes (ms). Include larger hops for MPS where compute is slower.",
    )
    parser.add_argument("--fade_ms_grid", type=str, default="0,10")

    parser.add_argument("--warmup_windows", type=int, default=2)
    parser.add_argument("--measured_windows", type=int, default=5)
    parser.add_argument("--max_rtf_p95", type=float, default=1.0)
    parser.add_argument(
        "--max_delay_proxy_p95",
        type=float,
        default=4.0,
        help="Stage-1 speed screen: hop_sec + p95_window_sec must be <= this value.",
    )
    parser.add_argument(
        "--max_speed_ok_configs",
        type=int,
        default=24,
        help="Cap configs evaluated in Stage 2 (sorted by lowest delay proxy). Use 0 to disable.",
    )

    parser.add_argument("--max_wer_gate", type=float, default=0.90)
    parser.add_argument("--min_content_gate", type=float, default=0.90)

    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(args.playlist_dir)[: args.max_files]
    if not wavs:
        raise FileNotFoundError(f"No wavs found in: {args.playlist_dir}")

    # Build config grid (deterministic).
    flow_steps_grid = _parse_int_grid(args.flow_steps_grid)
    window_ms_grid = _parse_int_grid(args.window_ms_grid)
    hop_ms_grid = _parse_int_grid(args.hop_ms_grid)
    fade_ms_grid = _parse_int_grid(args.fade_ms_grid)

    cfgs: list[EvalConfig] = []
    for flow_steps in flow_steps_grid:
        for window_ms in window_ms_grid:
            for hop_ms in hop_ms_grid:
                if hop_ms > window_ms:
                    continue
                for fade_ms in fade_ms_grid:
                    cfgs.append(
                        EvalConfig(
                            inference=VevoInferenceConfig(
                                kind=args.kind,  # type: ignore[arg-type]
                                flow_matching_steps=int(flow_steps),
                                seed=1234,
                            ),
                            streaming=VevoStreamingConfig(
                                window_ms=int(window_ms),
                                hop_ms=int(hop_ms),
                                fade_ms=int(fade_ms),
                            ),
                        )
                    )

    if not cfgs:
        raise RuntimeError("Empty config grid.")

    # Stage 1: speed screen (Vevo only).
    print(f"[tune_macos] Stage 1: speed screen ({len(cfgs)} configs)", flush=True)
    converter = VevoConverter.from_pretrained(
        kind=args.kind,  # type: ignore[arg-type]
        device=args.device,
        repo_cache_dir=args.repo_cache_dir,
    )
    bench = _bench_configs(
        converter,
        reference_wav=args.reference_wav,
        reference_max_sec=float(args.reference_max_sec),
        source_wav=wavs[0],
        cfgs=cfgs,
        warmup_windows=int(args.warmup_windows),
        measured_windows=int(args.measured_windows),
    )
    (out_dir / "bench.json").write_text(json.dumps(bench, indent=2))

    speed_ok: list[EvalConfig] = []
    for cfg in cfgs:
        b = bench.get(cfg.uid())
        if not b:
            continue
        if float(b.get("rtf_p95", 9e9)) > float(args.max_rtf_p95):
            continue
        if float(b.get("delay_proxy_p95_sec", 9e9)) > float(args.max_delay_proxy_p95):
            continue
        speed_ok.append(cfg)

    (out_dir / "speed_ok.json").write_text(
        json.dumps(
            {
                "count": len(speed_ok),
                "uids": [c.uid() for c in speed_ok],
                "grid": {
                    "flow_steps_grid": flow_steps_grid,
                    "window_ms_grid": window_ms_grid,
                    "hop_ms_grid": hop_ms_grid,
                    "fade_ms_grid": fade_ms_grid,
                },
                "constraints": {
                    "max_rtf_p95": float(args.max_rtf_p95),
                    "max_delay_proxy_p95": float(args.max_delay_proxy_p95),
                    "max_speed_ok_configs": int(args.max_speed_ok_configs),
                },
            },
            indent=2,
        )
    )

    if not speed_ok:
        print("[tune_macos] No configs passed speed constraints.", flush=True)
        return 2
    speed_ok_sorted = sorted(speed_ok, key=lambda c: float(bench.get(c.uid(), {}).get("delay_proxy_p95_sec", 9e9)))
    if int(args.max_speed_ok_configs) > 0:
        speed_ok_sorted = speed_ok_sorted[: int(args.max_speed_ok_configs)]
    print(f"[tune_macos] Stage 1 done: speed_ok={len(speed_ok)}/{len(cfgs)}", flush=True)

    # Stage 2: quality eval on speed-feasible configs.
    print(f"[tune_macos] Stage 2: quality eval ({len(speed_ok_sorted)} configs)", flush=True)
    whisper_model = load_whisper(args.whisper_model)
    import torch

    sim_device = torch.device(str(args.similarity_device))
    speaker_scorer = SpeakerSimilarityScorer(
        model_name=args.similarity_model,  # type: ignore[arg-type]
        ref_wav_path=args.reference_wav,
        device=sim_device,
    )

    quality_dir = out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for cfg in speed_ok_sorted:
        rec = _quality_eval_one(
            converter,
            speaker_scorer,
            whisper_model,
            reference_wav=args.reference_wav,
            reference_max_sec=float(args.reference_max_sec),
            wavs=wavs,
            cfg=cfg,
            eval_seconds=float(args.eval_seconds),
            out_dir=quality_dir,
        )

        uid = cfg.uid()
        b = bench.get(uid, {})
        rec["bench"] = b

        # Apply content/intelligibility gates.
        m = rec.get("metrics", {})
        gate_ok = True
        if float(args.max_wer_gate) > 0 and np.isfinite(m.get("wer", float("nan"))) and float(m["wer"]) > float(args.max_wer_gate):
            gate_ok = False
        if float(args.min_content_gate) > 0 and np.isfinite(m.get("content_hubert_cos", float("nan"))) and float(m["content_hubert_cos"]) < float(args.min_content_gate):
            gate_ok = False
        rec["metrics"]["gate_ok"] = bool(gate_ok)

        # Prefer bench-derived delay/rtf for selection.
        if b:
            rec["metrics"]["bench_rtf_p95"] = float(b.get("rtf_p95", float("nan")))
            rec["metrics"]["bench_delay_proxy_p95_sec"] = float(b.get("delay_proxy_p95_sec", float("nan")))

        records.append(rec)
        (quality_dir / uid / "record.json").write_text(json.dumps(rec, indent=2))

        m = rec["metrics"]
        print(
            f"[tune_macos] cfg={uid} win={cfg.streaming.window_ms} hop={cfg.streaming.hop_ms} steps={cfg.inference.flow_matching_steps} "
            f"wer={m.get('wer'):.3f} hubert={m.get('content_hubert_cos'):.3f} sim={m.get('speaker_similarity'):.3f} "
            f"bench_delay_p95={m.get('bench_delay_proxy_p95_sec', float('nan')):.2f}s gate={int(m.get('gate_ok', False))}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps({"bench": bench, "records": records}, indent=2))

    def _best_by_score(recs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        best_rec: Optional[dict[str, Any]] = None
        best_score = -1e18
        for r in recs:
            m = r.get("metrics", {})
            if not m.get("gate_ok"):
                continue
            s = float(m.get("score", -1e18))
            if s > best_score:
                best_score = s
                best_rec = r
        return best_rec

    def _best_by_delay(recs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        best_rec: Optional[dict[str, Any]] = None
        best_delay = 1e18
        best_score = -1e18
        for r in recs:
            m = r.get("metrics", {})
            if not m.get("gate_ok"):
                continue
            d = float(m.get("bench_delay_proxy_p95_sec", 1e18))
            s = float(m.get("score", -1e18))
            if d < best_delay or (d == best_delay and s > best_score):
                best_delay = d
                best_score = s
                best_rec = r
        return best_rec

    best_score = _best_by_score(records)
    best_low_delay = _best_by_delay(records)
    (out_dir / "best_score.json").write_text(json.dumps(best_score, indent=2))
    (out_dir / "best_low_delay.json").write_text(json.dumps(best_low_delay, indent=2))

    for name, best in [("best_score", best_score), ("best_low_delay", best_low_delay)]:
        if not best:
            continue
        (out_dir / f"{name}_config.json").write_text(json.dumps(best["config"], indent=2))

        best_artifacts = out_dir / f"{name}_deg"
        if best_artifacts.exists():
            shutil.rmtree(best_artifacts)
        best_uid = EvalConfig(
            inference=VevoInferenceConfig(**best["config"]["inference"]),
            streaming=VevoStreamingConfig(**best["config"]["streaming"]),
        ).uid()
        shutil.copytree(quality_dir / best_uid / "deg", best_artifacts)

    print(f"[tune_macos] wrote: {out_dir}/best_score.json and best_low_delay.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

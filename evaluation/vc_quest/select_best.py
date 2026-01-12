# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class PairMetrics:
    window_ms: int
    hop_ms: int
    mean_wer: float
    min_speaker_similarity: float
    max_silent_out_db_p95: float
    max_dropout_frac_voiced: float
    rtf_p95: float
    mean_call_score_v1: float
    mean_call_score_v2: float
    mean_ear_score_v2: float
    max_latency_p95_ms: float
    max_glitch_boundary_jump_ratio_p95: float


def _finite(x: float, default: float) -> float:
    if x is None:
        return default
    if isinstance(x, bool):
        return default
    try:
        xf = float(x)
    except Exception:
        return default
    if not math.isfinite(xf):
        return default
    return xf


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _target_similarity(rep: dict) -> float:
    if "speaker_similarity_target" in rep:
        return _finite(rep.get("speaker_similarity_target", float("nan")), 0.0)
    return _finite(rep.get("speaker_similarity", float("nan")), 0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select best streaming config from vc_quest reports."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument(
        "--score_key",
        type=str,
        default="call_score_v1",
        choices=["call_score_v1", "call_score_v2", "ear_score_v2"],
        help="Primary ranking metric (higher is better).",
    )
    parser.add_argument("--require_rtf_p95", type=float, default=1.0)
    parser.add_argument(
        "--require_min_speaker_similarity",
        type=float,
        default=0.85,
        help="Hard-ish constraint; relaxed if nothing matches.",
    )
    parser.add_argument(
        "--require_max_silent_out_db_p95",
        type=float,
        default=-25.0,
        help="Less is better (more negative). Hard-ish constraint; relaxed if nothing matches.",
    )
    parser.add_argument(
        "--require_max_dropout_frac_voiced",
        type=float,
        default=0.01,
        help="Hard-ish constraint; relaxed if nothing matches.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # Expect filenames like: v5_to_fr_stream_w600_h200.report.json
    pat = re.compile(r"^(?P<pair>.+)_stream_w(?P<w>\d+)_h(?P<h>\d+)\.report\.json$")

    buckets: dict[tuple[int, int], dict[str, Path]] = {}
    for p in run_dir.glob("*_stream_w*_h*.report.json"):
        m = pat.match(p.name)
        if not m:
            continue
        w = int(m.group("w"))
        h = int(m.group("h"))
        pair = str(m.group("pair"))
        buckets.setdefault((w, h), {})[pair] = p

    if not buckets:
        raise FileNotFoundError(f"No stream reports found under: {run_dir}")

    rows: list[PairMetrics] = []
    for (w, h), pair_map in sorted(buckets.items()):
        if "v5_to_fr" not in pair_map or "fr_to_v5" not in pair_map:
            continue

        rep_a = _load_json(pair_map["v5_to_fr"])
        rep_b = _load_json(pair_map["fr_to_v5"])

        # Grab matching meta JSONs for speed.
        meta_a = run_dir / f"v5_to_fr_stream_w{w}_h{h}.meta.json"
        meta_b = run_dir / f"fr_to_v5_stream_w{w}_h{h}.meta.json"
        if not meta_a.exists() or not meta_b.exists():
            continue
        ma = _load_json(meta_a)
        mb = _load_json(meta_b)

        wer_a = _finite(rep_a.get("wer", float("nan")), 1.0)
        wer_b = _finite(rep_b.get("wer", float("nan")), 1.0)
        mean_wer = 0.5 * (wer_a + wer_b)

        sim_a = _target_similarity(rep_a)
        sim_b = _target_similarity(rep_b)
        min_sim = min(sim_a, sim_b)

        leak_a = _finite(rep_a.get("artifact_silent_out_db_p95", float("nan")), 0.0)
        leak_b = _finite(rep_b.get("artifact_silent_out_db_p95", float("nan")), 0.0)
        max_leak = max(leak_a, leak_b)

        drop_a = _finite(rep_a.get("artifact_dropout_frac_voiced", float("nan")), 1.0)
        drop_b = _finite(rep_b.get("artifact_dropout_frac_voiced", float("nan")), 1.0)
        max_drop = max(drop_a, drop_b)

        score_a = _finite(rep_a.get("call_score_v1", float("nan")), 0.0)
        score_b = _finite(rep_b.get("call_score_v1", float("nan")), 0.0)
        mean_call_score_v1 = 0.5 * (score_a + score_b)

        score2_a = _finite(rep_a.get("call_score_v2", float("nan")), 0.0)
        score2_b = _finite(rep_b.get("call_score_v2", float("nan")), 0.0)
        mean_call_score_v2 = 0.5 * (score2_a + score2_b)

        ear_a = _finite(rep_a.get("ear_score_v2", float("nan")), 0.0)
        ear_b = _finite(rep_b.get("ear_score_v2", float("nan")), 0.0)
        mean_ear_score_v2 = 0.5 * (ear_a + ear_b)

        lat_a = _finite(rep_a.get("latency_p95_ms", float("nan")), 1e9)
        lat_b = _finite(rep_b.get("latency_p95_ms", float("nan")), 1e9)
        max_latency_p95_ms = max(lat_a, lat_b)

        glitch_a = _finite(
            rep_a.get("glitch_boundary_jump_ratio_p95", float("nan")), 0.0
        )
        glitch_b = _finite(
            rep_b.get("glitch_boundary_jump_ratio_p95", float("nan")), 0.0
        )
        max_glitch_boundary_jump_ratio_p95 = max(glitch_a, glitch_b)

        hop_sec = float(h) / 1000.0
        p95_a = _finite(ma.get("stats", {}).get("p95_window_sec", 0.0), 0.0)
        p95_b = _finite(mb.get("stats", {}).get("p95_window_sec", 0.0), 0.0)
        rtf_p95 = max(p95_a / max(hop_sec, 1e-6), p95_b / max(hop_sec, 1e-6))

        rows.append(
            PairMetrics(
                window_ms=w,
                hop_ms=h,
                mean_wer=mean_wer,
                min_speaker_similarity=min_sim,
                max_silent_out_db_p95=max_leak,
                max_dropout_frac_voiced=max_drop,
                rtf_p95=rtf_p95,
                mean_call_score_v1=float(mean_call_score_v1),
                mean_call_score_v2=float(mean_call_score_v2),
                mean_ear_score_v2=float(mean_ear_score_v2),
                max_latency_p95_ms=float(max_latency_p95_ms),
                max_glitch_boundary_jump_ratio_p95=float(
                    max_glitch_boundary_jump_ratio_p95
                ),
            )
        )

    if not rows:
        raise RuntimeError("No complete configs found (need both directions + meta).")

    # Primary constraint: real-time.
    realtime = [r for r in rows if r.rtf_p95 <= float(args.require_rtf_p95)]
    base = realtime if realtime else rows

    min_sim_req = float(args.require_min_speaker_similarity)
    max_leak_req = float(args.require_max_silent_out_db_p95)
    max_drop_req = float(args.require_max_dropout_frac_voiced)

    tiers: list[tuple[str, Callable[[PairMetrics], bool]]] = [
        (
            "quality",
            lambda r: (
                r.min_speaker_similarity >= min_sim_req
                and r.max_silent_out_db_p95 <= max_leak_req
                and r.max_dropout_frac_voiced <= max_drop_req
            ),
        ),
        (
            "no_leak_constraint",
            lambda r: (
                r.min_speaker_similarity >= min_sim_req
                and r.max_dropout_frac_voiced <= max_drop_req
            ),
        ),
        (
            "no_dropout_constraint",
            lambda r: (
                r.min_speaker_similarity >= min_sim_req
                and r.max_silent_out_db_p95 <= max_leak_req
            ),
        ),
        (
            "no_similarity_constraint",
            lambda r: (
                r.max_silent_out_db_p95 <= max_leak_req
                and r.max_dropout_frac_voiced <= max_drop_req
            ),
        ),
        ("rtf_only", lambda r: True),
    ]

    tier_name = "rtf_only"
    eligible = base
    for name, pred in tiers:
        subset = [r for r in base if pred(r)]
        if subset:
            tier_name = name
            eligible = subset
            break

    score_key = str(args.score_key)

    def _primary_score(r: PairMetrics) -> float:
        if score_key == "call_score_v2":
            return float(r.mean_call_score_v2)
        if score_key == "ear_score_v2":
            return float(r.mean_ear_score_v2)
        return float(r.mean_call_score_v1)

    # Sort for live-call UX: higher score is better, then lower latency, then lower WER.
    eligible.sort(
        key=lambda r: (
            -_primary_score(r),
            r.max_latency_p95_ms,
            r.mean_wer,
            -r.min_speaker_similarity,
            r.max_silent_out_db_p95,
            r.max_dropout_frac_voiced,
            r.max_glitch_boundary_jump_ratio_p95,
            r.rtf_p95,
        )
    )
    best = eligible[0]

    print("window_ms hop_ms score_key score call_v1 call_v2 ear_v2 lat_p95_ms mean_wer min_sim leak_p95_db drop_voiced glitch_p95 rtf_p95")
    print(f"[select_best] tier={tier_name} (eligible={len(eligible)}/{len(base)})")
    for r in eligible[:20]:
        print(
            f"{r.window_ms:8d} {r.hop_ms:6d} {score_key:8s} {_primary_score(r):5.3f} {r.mean_call_score_v1:7.3f} {r.mean_call_score_v2:7.3f} {r.mean_ear_score_v2:7.3f} {r.max_latency_p95_ms:10.1f} "
            f"{r.mean_wer:7.3f} {r.min_speaker_similarity:7.3f} {r.max_silent_out_db_p95:10.2f} "
            f"{r.max_dropout_frac_voiced:10.3f} {r.max_glitch_boundary_jump_ratio_p95:9.3f} {r.rtf_p95:7.3f}"
        )

    out = {
        "selection": {
            "tier": str(tier_name),
            "score_key": str(score_key),
            "require_rtf_p95": float(args.require_rtf_p95),
            "require_min_speaker_similarity": float(min_sim_req),
            "require_max_silent_out_db_p95": float(max_leak_req),
            "require_max_dropout_frac_voiced": float(max_drop_req),
        },
        "best": {
            "window_ms": int(best.window_ms),
            "hop_ms": int(best.hop_ms),
            "mean_call_score_v1": float(best.mean_call_score_v1),
            "mean_call_score_v2": float(best.mean_call_score_v2),
            "mean_ear_score_v2": float(best.mean_ear_score_v2),
            "mean_score": float(_primary_score(best)),
            "max_latency_p95_ms": float(best.max_latency_p95_ms),
            "max_glitch_boundary_jump_ratio_p95": float(
                best.max_glitch_boundary_jump_ratio_p95
            ),
            "mean_wer": float(best.mean_wer),
            "min_speaker_similarity": float(best.min_speaker_similarity),
            "max_silent_out_db_p95": float(best.max_silent_out_db_p95),
            "max_dropout_frac_voiced": float(best.max_dropout_frac_voiced),
            "rtf_p95": float(best.rtf_p95),
        },
        "candidates": [r.__dict__ for r in rows],
    }

    out_path = (
        Path(args.out_json) if args.out_json else (run_dir / "best_streaming.json")
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

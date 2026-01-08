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


@dataclass(frozen=True)
class PairMetrics:
    window_ms: int
    hop_ms: int
    mean_wer: float
    min_speaker_similarity: float
    max_silent_out_db_p95: float
    max_dropout_frac_voiced: float
    rtf_p95: float


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select best streaming config from vc_quest reports.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
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

        sim_a = _finite(rep_a.get("speaker_similarity", float("nan")), 0.0)
        sim_b = _finite(rep_b.get("speaker_similarity", float("nan")), 0.0)
        min_sim = min(sim_a, sim_b)

        leak_a = _finite(rep_a.get("artifact_silent_out_db_p95", float("nan")), 0.0)
        leak_b = _finite(rep_b.get("artifact_silent_out_db_p95", float("nan")), 0.0)
        max_leak = max(leak_a, leak_b)

        drop_a = _finite(rep_a.get("artifact_dropout_frac_voiced", float("nan")), 1.0)
        drop_b = _finite(rep_b.get("artifact_dropout_frac_voiced", float("nan")), 1.0)
        max_drop = max(drop_a, drop_b)

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

    tiers: list[tuple[str, callable[[PairMetrics], bool]]] = [
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
            lambda r: (r.min_speaker_similarity >= min_sim_req and r.max_dropout_frac_voiced <= max_drop_req),
        ),
        (
            "no_dropout_constraint",
            lambda r: (r.min_speaker_similarity >= min_sim_req and r.max_silent_out_db_p95 <= max_leak_req),
        ),
        (
            "no_similarity_constraint",
            lambda r: (
                r.max_silent_out_db_p95 <= max_leak_req and r.max_dropout_frac_voiced <= max_drop_req
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

    # Sort: lower WER is better, higher min speaker sim is better, lower leak (more negative) is better.
    eligible.sort(
        key=lambda r: (
            r.mean_wer,
            -r.min_speaker_similarity,
            r.max_silent_out_db_p95,
            r.max_dropout_frac_voiced,
            r.rtf_p95,
        )
    )
    best = eligible[0]

    print("window_ms hop_ms mean_wer min_sim leak_p95_db drop_voiced rtf_p95")
    print(f"[select_best] tier={tier_name} (eligible={len(eligible)}/{len(base)})")
    for r in eligible[:20]:
        print(
            f"{r.window_ms:8d} {r.hop_ms:6d} {r.mean_wer:7.3f} {r.min_speaker_similarity:7.3f} "
            f"{r.max_silent_out_db_p95:10.2f} {r.max_dropout_frac_voiced:10.3f} {r.rtf_p95:7.3f}"
        )

    out = {
        "selection": {
            "tier": str(tier_name),
            "require_rtf_p95": float(args.require_rtf_p95),
            "require_min_speaker_similarity": float(min_sim_req),
            "require_max_silent_out_db_p95": float(max_leak_req),
            "require_max_dropout_frac_voiced": float(max_drop_req),
        },
        "best": {
            "window_ms": int(best.window_ms),
            "hop_ms": int(best.hop_ms),
            "mean_wer": float(best.mean_wer),
            "min_speaker_similarity": float(best.min_speaker_similarity),
            "max_silent_out_db_p95": float(best.max_silent_out_db_p95),
            "max_dropout_frac_voiced": float(best.max_dropout_frac_voiced),
            "rtf_p95": float(best.rtf_p95),
        },
        "candidates": [r.__dict__ for r in rows],
    }

    out_path = Path(args.out_json) if args.out_json else (run_dir / "best_streaming.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

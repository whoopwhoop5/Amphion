# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from models.vc.vevo.live_engine import AudioRingBuffer, VevoStreamingEngine
from models.vc.vevo.runner import VevoConverter


@dataclass(frozen=True)
class BenchResult:
    kind: str
    device: str
    window_ms: int
    hop_ms: int
    flow_matching_steps: int
    warmup_windows: int
    measured_windows: int
    load_sec: float
    ref_prep_sec: float
    mean_window_sec: float
    p50_window_sec: float
    p95_window_sec: float

    @property
    def hop_sec(self) -> float:
        return self.hop_ms / 1000.0

    @property
    def rtf_mean(self) -> float:
        # <1.0 means faster than real-time for the hop budget.
        return self.mean_window_sec / max(self.hop_sec, 1e-9)

    @property
    def throughput_x(self) -> float:
        return self.hop_sec / max(self.mean_window_sec, 1e-9)


def _load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return np.asarray(x, dtype=np.float32).reshape(-1), int(sr)


def bench(
    *,
    kind: str,
    device: Optional[str],
    reference_wav: str,
    source_wav: str,
    repo_cache_dir: str,
    window_ms: int,
    hop_ms: int,
    flow_matching_steps: int,
    seed: int,
    warmup_windows: int,
    measured_windows: int,
) -> BenchResult:
    if warmup_windows < 0 or measured_windows <= 0:
        raise ValueError("warmup_windows must be >= 0 and measured_windows must be > 0")

    t0 = time.time()
    converter = VevoConverter.from_pretrained(
        kind=kind,  # type: ignore[arg-type]
        device=device,
        repo_cache_dir=repo_cache_dir,
    )
    load_sec = time.time() - t0

    engine = VevoStreamingEngine(converter)

    t0 = time.time()
    engine.prepare_reference_bytes(Path(reference_wav).read_bytes())
    ref_prep_sec = time.time() - t0

    src, sr = _load_wav_mono(source_wav)
    if sr != engine.model_sr:
        raise ValueError(f"Expected source_wav at {engine.model_sr}Hz, got {sr}Hz: {source_wav}")

    window_samples = int(round(window_ms / 1000 * engine.model_sr))
    hop_samples = int(round(hop_ms / 1000 * engine.model_sr))
    if hop_samples <= 0 or window_samples <= 0:
        raise ValueError("window_ms and hop_ms must be > 0")
    if hop_samples > window_samples:
        raise ValueError("hop_ms must be <= window_ms")

    ring = AudioRingBuffer(window_samples)
    timings: list[float] = []

    need_windows = warmup_windows + measured_windows
    window_count = 0
    hop_count = 0

    for start in range(0, len(src), hop_samples):
        hop = src[start : start + hop_samples]
        if len(hop) < hop_samples:
            hop = np.pad(hop, (0, hop_samples - len(hop)), mode="constant")
        ring.write(hop)
        hop_count += 1

        if ring.size < window_samples:
            continue

        window = ring.read_last(window_samples)

        t0 = time.time()
        _ = engine.convert_window(
            window,
            flow_matching_steps=flow_matching_steps,
            seed=seed + window_count,
        )
        dt = time.time() - t0

        if window_count >= warmup_windows:
            timings.append(dt)

        window_count += 1
        if window_count >= need_windows:
            break

    if len(timings) != measured_windows:
        raise RuntimeError(
            f"Not enough audio to benchmark: got {len(timings)} measured windows "
            f"(wanted {measured_windows}). hops={hop_count} windows={window_count}"
        )

    arr = np.asarray(timings, dtype=np.float64)
    return BenchResult(
        kind=kind,
        device=str(converter.device),
        window_ms=window_ms,
        hop_ms=hop_ms,
        flow_matching_steps=flow_matching_steps,
        warmup_windows=warmup_windows,
        measured_windows=measured_windows,
        load_sec=float(load_sec),
        ref_prep_sec=float(ref_prep_sec),
        mean_window_sec=float(np.mean(arr)),
        p50_window_sec=float(np.percentile(arr, 50)),
        p95_window_sec=float(np.percentile(arr, 95)),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Vevo window inference on the local machine.")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. mps, cpu, cuda")
    parser.add_argument("--reference_wav", type=str, default="assets/vevo_live/target_ref.wav")
    parser.add_argument("--source_wav", type=str, default="assets/vevo_live/playlist/source_clip_00.wav")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")

    parser.add_argument("--window_ms", type=int, default=1000)
    parser.add_argument("--hop_ms", type=int, default=500)
    parser.add_argument("--flow_matching_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup_windows", type=int, default=2)
    parser.add_argument("--measured_windows", type=int, default=5)
    args = parser.parse_args(argv)

    result = bench(
        kind=args.kind,
        device=args.device,
        reference_wav=args.reference_wav,
        source_wav=args.source_wav,
        repo_cache_dir=args.repo_cache_dir,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        flow_matching_steps=args.flow_matching_steps,
        seed=args.seed,
        warmup_windows=args.warmup_windows,
        measured_windows=args.measured_windows,
    )

    print(
        "\n".join(
            [
                "[vevo_bench]",
                f" kind={result.kind}",
                f" device={result.device}",
                f" window_ms={result.window_ms} hop_ms={result.hop_ms} steps={result.flow_matching_steps}",
                f" load_sec={result.load_sec:.2f} ref_prep_sec={result.ref_prep_sec:.2f}",
                f" window_sec mean={result.mean_window_sec:.3f} p50={result.p50_window_sec:.3f} p95={result.p95_window_sec:.3f}",
                f" rtf_mean={result.rtf_mean:.2f} throughput_x={result.throughput_x:.2f}",
            ]
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


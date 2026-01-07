# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from models.vc.vevo.live_engine import (
    AudioRingBuffer,
    crossfade_inplace,
    normalize_length,
)
from models.vc.vevo.runner import VevoConverter
from models.vc.vevo.live_engine import VevoStreamingEngine


@dataclass
class SessionConfig:
    kind: str
    sample_rate: int
    window_samples: int
    hop_samples: int
    fade_samples: int
    flow_matching_steps: int
    diffusion_cfg: float
    diffusion_rescale_cfg: float
    seed: Optional[int]

    ar_max_length: int
    ar_temperature: float
    ar_top_k: int
    ar_top_p: float
    ar_repeat_penalty: float
    ar_min_new_tokens: int
    prepend_style_ref_to_input: bool

    normalize_align: str


def _parse_init(msg: dict[str, Any]) -> tuple[SessionConfig, bytes]:
    if msg.get("type") != "init":
        raise ValueError("First message must be type=init")

    ref_b64 = msg.get("reference_wav_b64")
    if not isinstance(ref_b64, str) or not ref_b64:
        raise ValueError("reference_wav_b64 required")
    ref_bytes = base64.b64decode(ref_b64.encode("utf-8"))

    cfg = SessionConfig(
        kind=str(msg.get("kind", "vevotimbre")),
        sample_rate=int(msg.get("sample_rate", 24000)),
        window_samples=int(msg.get("window_samples", 24000)),
        hop_samples=int(msg.get("hop_samples", 24000)),
        fade_samples=int(msg.get("fade_samples", 480)),
        flow_matching_steps=int(msg.get("flow_matching_steps", 16)),
        diffusion_cfg=float(msg.get("diffusion_cfg", 1.0)),
        diffusion_rescale_cfg=float(msg.get("diffusion_rescale_cfg", 0.75)),
        seed=int(msg["seed"]) if msg.get("seed") is not None else None,
        ar_max_length=int(msg.get("ar_max_length", 2000)),
        ar_temperature=float(msg.get("ar_temperature", 0.8)),
        ar_top_k=int(msg.get("ar_top_k", 50)),
        ar_top_p=float(msg.get("ar_top_p", 0.9)),
        ar_repeat_penalty=float(msg.get("ar_repeat_penalty", 1.0)),
        ar_min_new_tokens=int(msg.get("ar_min_new_tokens", 50)),
        prepend_style_ref_to_input=bool(msg.get("prepend_style_ref_to_input", True)),
        normalize_align=str(msg.get("normalize_align", "end")),
    )

    if cfg.sample_rate != VevoStreamingEngine.model_sr:
        raise ValueError(
            f"Server expects sample_rate={VevoStreamingEngine.model_sr} float32 mono. "
            f"Got sample_rate={cfg.sample_rate}. Resample on client."
        )
    if cfg.window_samples <= 0 or cfg.hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be > 0")
    if cfg.hop_samples > cfg.window_samples:
        raise ValueError("hop_samples must be <= window_samples")
    if cfg.fade_samples < 0:
        raise ValueError("fade_samples must be >= 0")
    if cfg.normalize_align not in ("start", "end"):
        raise ValueError("normalize_align must be start|end")

    return cfg, ref_bytes


async def _serve_client(websocket, repo_cache_dir: str, device: Optional[str]) -> None:
    init_raw = await websocket.recv()
    if isinstance(init_raw, bytes):
        raise ValueError("Expected init JSON message, got binary")

    init_msg = json.loads(init_raw)
    cfg, ref_bytes = _parse_init(init_msg)

    converter = VevoConverter.from_pretrained(
        kind=cfg.kind,  # type: ignore[arg-type]
        device=device,
        repo_cache_dir=repo_cache_dir,
    )
    engine = VevoStreamingEngine(converter)
    engine.prepare_reference_bytes(ref_bytes)

    ring = AudioRingBuffer(cfg.window_samples)
    prev_tail: Optional[np.ndarray] = None

    await websocket.send(
        json.dumps(
            {
                "type": "ready",
                "model_sr": engine.model_sr,
                "window_samples": cfg.window_samples,
                "hop_samples": cfg.hop_samples,
                "fade_samples": cfg.fade_samples,
                "device": str(converter.device),
            }
        )
    )

    window_count = 0
    t0 = time.time()
    while True:
        msg = await websocket.recv()
        if isinstance(msg, str):
            data = json.loads(msg)
            if data.get("type") == "close":
                return
            continue

        # binary audio chunk: float32 mono with exactly hop_samples
        chunk = np.frombuffer(msg, dtype=np.float32)
        if chunk.ndim != 1 or len(chunk) != cfg.hop_samples:
            raise ValueError(f"Expected {cfg.hop_samples} float32 samples, got {chunk.shape}")

        ring.write(chunk)
        if ring.size < cfg.window_samples:
            await websocket.send(np.zeros(cfg.hop_samples, dtype=np.float32).tobytes())
            continue

        window = ring.read_last(cfg.window_samples)

        out_window = engine.convert_window(
            window,
            flow_matching_steps=cfg.flow_matching_steps,
            diffusion_cfg=cfg.diffusion_cfg,
            diffusion_rescale_cfg=cfg.diffusion_rescale_cfg,
            seed=None if cfg.seed is None else cfg.seed + window_count,
            ar_max_length=cfg.ar_max_length,
            ar_temperature=cfg.ar_temperature,
            ar_top_k=cfg.ar_top_k,
            ar_top_p=cfg.ar_top_p,
            ar_repeat_penalty=cfg.ar_repeat_penalty,
            ar_min_new_tokens=cfg.ar_min_new_tokens,
            prepend_style_ref_to_input=cfg.prepend_style_ref_to_input,
        )

        out_window = normalize_length(out_window, cfg.window_samples, align=cfg.normalize_align)  # type: ignore[arg-type]
        hop = out_window[-cfg.hop_samples :].astype(np.float32, copy=False)
        hop = crossfade_inplace(hop, prev_tail, cfg.fade_samples)
        prev_tail = hop[-cfg.fade_samples :].copy() if cfg.fade_samples > 0 else None

        await websocket.send(hop.tobytes())
        window_count += 1

        # occasional heartbeat log
        if window_count % max(1, int(5 * engine.model_sr / cfg.hop_samples)) == 0:
            dt = time.time() - t0
            rtf = (window_count * cfg.hop_samples / engine.model_sr) / max(dt, 1e-6)
            print(
                f"[live_server] windows={window_count} elapsed_s={dt:.1f} "
                f"throughput_x={rtf:.2f} kind={cfg.kind} steps={cfg.flow_matching_steps}",
                flush=True,
            )


async def _main_async(args) -> None:
    try:
        import websockets
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: websockets. Install with `pip install websockets`."
        ) from e

    async def handler(websocket):
        await _serve_client(
            websocket,
            repo_cache_dir=args.repo_cache_dir,
            device=args.device,
        )

    print(f"[live_server] Listening on ws://{args.host}:{args.port}", flush=True)
    async with websockets.serve(handler, args.host, args.port, max_size=32 * 1024 * 1024):
        await asyncio.Future()  # run forever


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Vevo live VC server (GPU host).")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    # Make torch init deterministic-ish by default for live usage.
    torch.set_grad_enabled(False)

    asyncio.run(_main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


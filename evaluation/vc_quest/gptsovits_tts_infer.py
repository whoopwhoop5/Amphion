#!/usr/bin/env python3
# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def _read_text_arg(text: str, text_file: str) -> str:
    t = str(text).strip()
    if t:
        return t
    p = str(text_file).strip()
    if not p:
        raise ValueError("Must provide --text or --text_file")
    return Path(p).read_text(encoding="utf-8").strip()


def _resolve_maybe_relative(path: str, *, base_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS TTS inference wrapper (for ASR→TTS voice changer experiments)."
    )
    parser.add_argument("--gptsovits_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="GPT_SoVITS/configs/tts_infer.yaml")
    parser.add_argument("--t2s_weights_path", type=str, default="")
    parser.add_argument("--vits_weights_path", type=str, default="")

    parser.add_argument("--ref_audio_path", type=str, required=True)
    parser.add_argument("--prompt_text", type=str, default="")
    parser.add_argument("--prompt_text_file", type=str, default="")
    parser.add_argument("--prompt_lang", type=str, default="auto")

    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--text_file", type=str, default="")
    parser.add_argument("--text_lang", type=str, default="auto")

    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument("--out_wav", type=str, required=True)
    parser.add_argument("--meta_json", type=str, default="")
    args = parser.parse_args(argv)

    repo_dir = Path(args.gptsovits_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(repo_dir)

    out_wav = Path(args.out_wav).resolve()
    meta_json = Path(args.meta_json).resolve() if str(args.meta_json).strip() else None

    ref_audio = Path(args.ref_audio_path).resolve()
    if not ref_audio.exists():
        raise FileNotFoundError(ref_audio)

    text = _read_text_arg(str(args.text), str(args.text_file))
    prompt_text = _read_text_arg(str(args.prompt_text), str(args.prompt_text_file))

    cfg_path = _resolve_maybe_relative(str(args.config_path), base_dir=repo_dir)
    t2s_weights_path = (
        _resolve_maybe_relative(str(args.t2s_weights_path), base_dir=repo_dir)
        if str(args.t2s_weights_path).strip()
        else None
    )
    vits_weights_path = (
        _resolve_maybe_relative(str(args.vits_weights_path), base_dir=repo_dir)
        if str(args.vits_weights_path).strip()
        else None
    )

    # GPT-SoVITS expects to run with cwd at repo root due to relative paths.
    os.chdir(str(repo_dir))
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "GPT_SoVITS"))

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # type: ignore[import-not-found]

    tts_cfg = TTS_Config(str(cfg_path))
    tts = TTS(tts_cfg)

    if t2s_weights_path is not None:
        tts.init_t2s_weights(str(t2s_weights_path))
    if vits_weights_path is not None:
        tts.init_vits_weights(str(vits_weights_path))

    req = {
        "text": str(text),
        "text_lang": str(args.text_lang).lower(),
        "ref_audio_path": str(ref_audio),
        "prompt_text": str(prompt_text),
        "prompt_lang": str(args.prompt_lang).lower(),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "speed_factor": float(args.speed_factor),
        "seed": int(args.seed),
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": False,
    }

    t0 = time.perf_counter()
    gen = tts.run(req)
    sr, audio = next(gen)
    elapsed = time.perf_counter() - t0

    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), wav, int(sr))

    dur_sec = float(len(wav) / float(sr)) if int(sr) > 0 else 0.0
    rtf = float(elapsed / dur_sec) if dur_sec > 0 else float("nan")

    meta = {
        "kind": "gptsovits_tts",
        "config": {
            "gptsovits_dir": str(repo_dir),
            "config_path": str(cfg_path),
            "t2s_weights_path": str(t2s_weights_path) if t2s_weights_path else "",
            "vits_weights_path": str(vits_weights_path) if vits_weights_path else "",
            "ref_audio_path": str(ref_audio),
            "prompt_lang": str(args.prompt_lang).lower(),
            "text_lang": str(args.text_lang).lower(),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "temperature": float(args.temperature),
            "speed_factor": float(args.speed_factor),
            "seed": int(args.seed),
        },
        "stats": {
            "elapsed_sec": float(elapsed),
            "out_sr": int(sr),
            "out_samples": int(len(wav)),
            "out_dur_sec": float(dur_sec),
            "rtf": float(rtf) if np.isfinite(rtf) else float("nan"),
        },
        "paths": {
            "out_wav": str(out_wav),
        },
    }

    if meta_json is not None:
        meta_json.parent.mkdir(parents=True, exist_ok=True)
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


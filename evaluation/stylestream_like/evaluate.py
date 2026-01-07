# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from evaluation.stylestream_like.metrics import (
    Emotion2VecEmbedder,
    ResemblyzerEmbedder,
    SpeechBrainEmbedder,
    WhisperTranscriber,
    word_error_rate,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Vevo vs StyleStream-like objective metrics on fixed pairs.")
    parser.add_argument("--manifest", type=str, default="runs/stylestream_like/manifest.json")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument("--out_dir", type=str, default="runs/stylestream_like/eval")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--flow_matching_steps", type=int, default=16)
    parser.add_argument("--diffusion_cfg", type=float, default=1.0)
    parser.add_argument("--diffusion_rescale_cfg", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--whisper_model", type=str, default="large-v3")
    parser.add_argument("--use_accent", action="store_true", help="Enable accent similarity (SpeechBrain model).")
    parser.add_argument("--use_emotion", action="store_true", help="Enable emotion similarity (emotion2vec).")
    parser.add_argument("--max_pairs", type=int, default=0, help="Limit number of pairs (0 = all).")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    sources = {s["id"]: s for s in manifest["sources"]}
    targets = {t["id"]: t for t in manifest["targets"]}
    pairs = list(manifest["pairs"])
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deg_dir = out_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)

    from models.vc.vevo.runner import VevoConverter

    converter = VevoConverter.from_pretrained(
        kind=args.kind,  # type: ignore[arg-type]
        device=args.device,
        repo_cache_dir=args.repo_cache_dir,
    )

    torch_device = converter.device
    whisper = WhisperTranscriber(model_size=args.whisper_model, device=str(torch_device))
    speaker = ResemblyzerEmbedder(device=str(torch_device))
    accent = (
        SpeechBrainEmbedder(
            hf_repo_id="Jzuluaga/accent-id-commonaccent_ecapa",
            device=str(torch_device),
        )
        if args.use_accent
        else None
    )
    ms_device: Optional[int] = 0 if torch_device.type == "cuda" else -1
    emotion = (
        Emotion2VecEmbedder(model_id="iic/emotion2vec_base", device=ms_device)
        if args.use_emotion
        else None
    )
    # Cache target embeddings (targets are reused across many pairs).
    target_speaker_emb: dict[str, np.ndarray] = {}
    target_accent_emb: dict[str, np.ndarray] = {}
    target_emotion_emb: dict[str, np.ndarray] = {}
    for tid, t in targets.items():
        wav_path = str(t["wav_path"])
        target_speaker_emb[tid] = speaker.embed(wav_path)
        if accent is not None:
            target_accent_emb[tid] = accent.embed(wav_path)
        if emotion is not None:
            target_emotion_emb[tid] = emotion.embed(wav_path)

    def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(a) + eps)
        nb = float(np.linalg.norm(b) + eps)
        return float(np.dot(a, b) / (na * nb))

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, p in enumerate(pairs):
        sid = p["source_id"]
        tid = p["target_id"]
        src = sources[sid]
        tgt = targets[tid]

        out_path = deg_dir / f"pair_{i:05d}.wav"
        converter.convert_file(
            src_wav_path=str(src["wav_path"]),
            reference_wav_path=str(tgt["wav_path"]),
            output_path=str(out_path),
            flow_matching_steps=args.flow_matching_steps,
            diffusion_cfg=args.diffusion_cfg,
            diffusion_rescale_cfg=args.diffusion_rescale_cfg,
            seed=args.seed + i,
        )

        pred_text, _lang = whisper.transcribe(str(out_path))
        wer = word_error_rate(str(pred_text), str(src["transcript"]))

        out_spk = speaker.embed(str(out_path))
        s_sim = cosine(target_speaker_emb[tid], out_spk)

        a_sim = float("nan")
        if accent is not None:
            out_acc = accent.embed(str(out_path))
            a_sim = cosine(target_accent_emb[tid], out_acc)

        e_sim = float("nan")
        if emotion is not None:
            out_emo = emotion.embed(str(out_path))
            e_sim = cosine(target_emotion_emb[tid], out_emo)

        row = {
            "pair_index": i,
            "source_id": sid,
            "target_id": tid,
            "source_accent": src.get("accent") or "",
            "target_accent": tgt.get("accent") or "",
            "target_emotion": tgt.get("emotion") or "",
            "wer": float(wer) if np.isfinite(wer) else float("nan"),
            "s_sim": float(s_sim) if np.isfinite(s_sim) else float("nan"),
            "a_sim": float(a_sim) if np.isfinite(a_sim) else float("nan"),
            "e_sim": float(e_sim) if np.isfinite(e_sim) else float("nan"),
        }
        rows.append(row)

        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"[eval] {i+1}/{len(pairs)} elapsed_s={dt:.1f}", flush=True)

    # Aggregate
    def mean_of(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    report = {
        "meta": {
            "kind": args.kind,
            "flow_matching_steps": args.flow_matching_steps,
            "diffusion_cfg": args.diffusion_cfg,
            "diffusion_rescale_cfg": args.diffusion_rescale_cfg,
            "seed": args.seed,
            "whisper_model": args.whisper_model,
            "use_accent": bool(args.use_accent),
            "use_emotion": bool(args.use_emotion),
            "num_pairs": len(rows),
            "elapsed_sec": float(time.time() - t0),
        },
        "means": {
            "wer": mean_of("wer"),
            "s_sim": mean_of("s_sim"),
            "a_sim": mean_of("a_sim"),
            "e_sim": mean_of("e_sim"),
        },
        "rows": rows,
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {report_path}")
    print(json.dumps(report["means"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

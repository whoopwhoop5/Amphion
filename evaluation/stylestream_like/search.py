# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class OfflineSearchConfig:
    kind: str
    flow_matching_steps: int
    diffusion_cfg: float
    diffusion_rescale_cfg: float
    seed: int

    # vevovoice-only knobs (kept here so we can tune them too).
    ar_max_length: int = 2000
    ar_temperature: float = 0.8
    ar_top_k: int = 50
    ar_top_p: float = 0.9
    ar_repeat_penalty: float = 1.0
    ar_min_new_tokens: int = 50

    def uid(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:10]


def _parse_int_grid(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Empty int grid")
    return out


def _parse_float_grid(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Empty float grid")
    return out


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a) + eps)
    nb = float(np.linalg.norm(b) + eps)
    return float(np.dot(a, b) / (na * nb))


def _score(
    means: dict[str, float],
    *,
    w_spk: float,
    w_wer: float,
    w_a_sim: float,
    w_e_sim: float,
    w_steps: float,
    steps: int,
) -> float:
    wer = float(means.get("wer", float("nan")))
    s_sim = float(means.get("s_sim", float("nan")))
    a_sim = float(means.get("a_sim", float("nan")))
    e_sim = float(means.get("e_sim", float("nan")))

    if not math.isfinite(wer):
        wer = 1.0
    if not math.isfinite(s_sim):
        s_sim = 0.0
    if not math.isfinite(a_sim):
        a_sim = 0.0
    if not math.isfinite(e_sim):
        e_sim = 0.0

    # Higher is better.
    score = 0.0
    score += w_spk * s_sim
    score += w_a_sim * a_sim
    score += w_e_sim * e_sim
    score -= w_wer * wer
    score -= w_steps * (float(steps) / 32.0)
    return float(score)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _run_eval(
    *,
    converter,
    pairs: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    out_dir: Path,
    cfg: OfflineSearchConfig,
    whisper: WhisperTranscriber,
    speaker: ResemblyzerEmbedder,
    accent: Optional[SpeechBrainEmbedder],
    emotion: Optional[Emotion2VecEmbedder],
    target_speaker_emb: dict[str, np.ndarray],
    target_accent_emb: Optional[dict[str, np.ndarray]],
    target_emotion_emb: Optional[dict[str, np.ndarray]],
    keep_first_n_audio: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    deg_dir = out_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)

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
            flow_matching_steps=int(cfg.flow_matching_steps),
            diffusion_cfg=float(cfg.diffusion_cfg),
            diffusion_rescale_cfg=float(cfg.diffusion_rescale_cfg),
            seed=int(cfg.seed) + int(i),
            ar_max_length=int(cfg.ar_max_length),
            ar_temperature=float(cfg.ar_temperature),
            ar_top_k=int(cfg.ar_top_k),
            ar_top_p=float(cfg.ar_top_p),
            ar_repeat_penalty=float(cfg.ar_repeat_penalty),
            ar_min_new_tokens=int(cfg.ar_min_new_tokens),
        )

        pred_text, _lang = whisper.transcribe(str(out_path))
        wer = word_error_rate(str(pred_text), str(src.get("transcript") or ""))

        out_spk = speaker.embed(str(out_path))
        s_sim = _cosine(target_speaker_emb[tid], out_spk)

        a_sim = float("nan")
        if accent is not None and target_accent_emb is not None:
            out_acc = accent.embed(str(out_path))
            a_sim = _cosine(target_accent_emb[tid], out_acc)

        e_sim = float("nan")
        if emotion is not None and target_emotion_emb is not None:
            out_emo = emotion.embed(str(out_path))
            e_sim = _cosine(target_emotion_emb[tid], out_emo)

        row = {
            "pair_index": int(i),
            "source_id": sid,
            "target_id": tid,
            "wer": float(wer) if np.isfinite(wer) else float("nan"),
            "s_sim": float(s_sim) if np.isfinite(s_sim) else float("nan"),
            "a_sim": float(a_sim) if np.isfinite(a_sim) else float("nan"),
            "e_sim": float(e_sim) if np.isfinite(e_sim) else float("nan"),
            "target_accent": str(tgt.get("accent") or ""),
            "target_emotion": str(tgt.get("emotion") or ""),
        }
        rows.append(row)

        # Keep disk usage bounded: only retain a small listening set.
        if keep_first_n_audio <= 0 or i >= keep_first_n_audio:
            try:
                out_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"[search] eval {i+1}/{len(pairs)} elapsed_s={dt:.1f}", flush=True)

    means = {
        "wer": _mean(rows, "wer"),
        "s_sim": _mean(rows, "s_sim"),
        "a_sim": _mean(rows, "a_sim"),
        "e_sim": _mean(rows, "e_sim"),
    }
    return {
        "meta": {
            "kind": cfg.kind,
            "flow_matching_steps": int(cfg.flow_matching_steps),
            "diffusion_cfg": float(cfg.diffusion_cfg),
            "diffusion_rescale_cfg": float(cfg.diffusion_rescale_cfg),
            "seed": int(cfg.seed),
            "ar_temperature": float(cfg.ar_temperature),
            "ar_top_p": float(cfg.ar_top_p),
            "ar_top_k": int(cfg.ar_top_k),
            "ar_repeat_penalty": float(cfg.ar_repeat_penalty),
            "num_pairs": int(len(rows)),
            "elapsed_sec": float(time.time() - t0),
            "whisper_model": getattr(whisper, "_model", None).__class__.__name__,
        },
        "means": means,
        "rows": rows,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic offline Vevo parameter search (StyleStream-like metrics).")
    parser.add_argument("--manifest", type=str, default="runs/stylestream_like/manifest.json")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument("--out_dir", type=str, default="runs/stylestream_like_search")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--max_pairs", type=int, default=50, help="Pairs used during search (0=all).")
    parser.add_argument("--final_max_pairs", type=int, default=0, help="Pairs used for final eval of best config (0=all).")
    parser.add_argument("--keep_first_n_audio", type=int, default=5, help="How many deg wavs to keep per run for listening.")

    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model for search stage.")
    parser.add_argument("--final_whisper_model", type=str, default="large-v3", help="Whisper model for final eval.")
    parser.add_argument("--use_accent", action="store_true", help="Include A-SIM in search/eval.")
    parser.add_argument("--use_emotion", action="store_true", help="Include E-SIM in search/eval (downloads ~1GB).")

    parser.add_argument("--flow_steps_grid", type=str, default="8,12,16,24,32")
    parser.add_argument("--diffusion_cfg_grid", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--diffusion_rescale_grid", type=str, default="0.0,0.5,0.75")

    # vevovoice-only grids (kept small by default).
    parser.add_argument("--ar_temperature_grid", type=str, default="0.8")
    parser.add_argument("--ar_top_p_grid", type=str, default="0.9")
    parser.add_argument("--ar_top_k_grid", type=str, default="50")
    parser.add_argument("--ar_repeat_penalty_grid", type=str, default="1.0")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--patience", type=int, default=999999, help="Early stop after N stale configs.")
    parser.add_argument("--min_improve", type=float, default=1e-4)

    parser.add_argument("--w_spk", type=float, default=1.0)
    parser.add_argument("--w_wer", type=float, default=0.5)
    parser.add_argument("--w_a_sim", type=float, default=0.0)
    parser.add_argument("--w_e_sim", type=float, default=0.0)
    parser.add_argument("--w_steps", type=float, default=0.05, help="Penalty per 32 steps (proxy for runtime).")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    sources = {s["id"]: s for s in manifest["sources"]}
    targets = {t["id"]: t for t in manifest["targets"]}
    pairs = list(manifest["pairs"])
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

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
        SpeechBrainEmbedder(hf_repo_id="Jzuluaga/accent-id-commonaccent_ecapa", device=str(torch_device))
        if args.use_accent
        else None
    )
    ms_device: Optional[int] = 0 if torch_device.type == "cuda" else -1
    emotion = Emotion2VecEmbedder(model_id="iic/emotion2vec_base", device=ms_device) if args.use_emotion else None

    # Cache target embeddings (targets are reused across many pairs/configs).
    target_speaker_emb: dict[str, np.ndarray] = {}
    target_accent_emb: Optional[dict[str, np.ndarray]] = {} if accent is not None else None
    target_emotion_emb: Optional[dict[str, np.ndarray]] = {} if emotion is not None else None
    for tid, t in targets.items():
        wav_path = str(t["wav_path"])
        target_speaker_emb[tid] = speaker.embed(wav_path)
        if accent is not None and target_accent_emb is not None:
            target_accent_emb[tid] = accent.embed(wav_path)
        if emotion is not None and target_emotion_emb is not None:
            target_emotion_emb[tid] = emotion.embed(wav_path)

    flow_steps_grid = _parse_int_grid(args.flow_steps_grid)
    cfg_grid = _parse_float_grid(args.diffusion_cfg_grid)
    rescale_grid = _parse_float_grid(args.diffusion_rescale_grid)
    ar_temperature_grid = _parse_float_grid(args.ar_temperature_grid)
    ar_top_p_grid = _parse_float_grid(args.ar_top_p_grid)
    ar_top_k_grid = _parse_int_grid(args.ar_top_k_grid)
    ar_repeat_penalty_grid = _parse_float_grid(args.ar_repeat_penalty_grid)

    results: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    best_score = -1e9
    stale = 0

    configs: list[OfflineSearchConfig] = []
    for steps in flow_steps_grid:
        for dcfg in cfg_grid:
            for rcfg in rescale_grid:
                if args.kind == "vevotimbre":
                    configs.append(
                        OfflineSearchConfig(
                            kind=args.kind,
                            flow_matching_steps=int(steps),
                            diffusion_cfg=float(dcfg),
                            diffusion_rescale_cfg=float(rcfg),
                            seed=int(args.seed),
                        )
                    )
                    continue

                for temp in ar_temperature_grid:
                    for top_p in ar_top_p_grid:
                        for top_k in ar_top_k_grid:
                            for rep in ar_repeat_penalty_grid:
                                configs.append(
                                    OfflineSearchConfig(
                                        kind=args.kind,
                                        flow_matching_steps=int(steps),
                                        diffusion_cfg=float(dcfg),
                                        diffusion_rescale_cfg=float(rcfg),
                                        seed=int(args.seed),
                                        ar_temperature=float(temp),
                                        ar_top_p=float(top_p),
                                        ar_top_k=int(top_k),
                                        ar_repeat_penalty=float(rep),
                                    )
                                )

    print(f"[search] configs={len(configs)} pairs={len(pairs)}", flush=True)

    for cfg in configs:
        uid = cfg.uid()
        cfg_dir = out_root / "configs" / uid
        if cfg_dir.exists():
            shutil.rmtree(cfg_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

        record = _run_eval(
            converter=converter,
            pairs=pairs,
            sources=sources,
            targets=targets,
            out_dir=cfg_dir,
            cfg=cfg,
            whisper=whisper,
            speaker=speaker,
            accent=accent,
            emotion=emotion,
            target_speaker_emb=target_speaker_emb,
            target_accent_emb=target_accent_emb,
            target_emotion_emb=target_emotion_emb,
            keep_first_n_audio=int(args.keep_first_n_audio),
        )

        means = record["means"]
        score = _score(
            means,
            w_spk=float(args.w_spk),
            w_wer=float(args.w_wer),
            w_a_sim=float(args.w_a_sim),
            w_e_sim=float(args.w_e_sim),
            w_steps=float(args.w_steps),
            steps=int(cfg.flow_matching_steps),
        )
        record["score"] = float(score)
        (cfg_dir / "result.json").write_text(json.dumps(record, indent=2))

        results.append({"uid": uid, "score": float(score), "config": asdict(cfg), "means": means})

        if score > best_score + float(args.min_improve):
            best_score = float(score)
            best = results[-1]
            stale = 0

            best_dir = out_root / "best_search"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "best_config.json").write_text(json.dumps(best["config"], indent=2))
            (best_dir / "best_means.json").write_text(json.dumps(best["means"], indent=2))
            (best_dir / "best_score.txt").write_text(f"{best_score:.6f}\n")
            shutil.copytree(cfg_dir / "deg", best_dir / "deg")
        else:
            stale += 1

        print(
            f"[search] uid={uid} steps={cfg.flow_matching_steps} cfg={cfg.diffusion_cfg} r={cfg.diffusion_rescale_cfg} "
            f"wer={means['wer']:.4f} s={means['s_sim']:.4f} a={means['a_sim']:.4f} e={means['e_sim']:.4f} "
            f"score={score:.4f} best={best_score:.4f}",
            flush=True,
        )

        if stale >= int(args.patience):
            print("[search] Early stop: plateau reached.", flush=True)
            break

    leaderboard = sorted(results, key=lambda r: float(r["score"]), reverse=True)
    (out_root / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    print(f"[search] Wrote: {out_root / 'leaderboard.json'}", flush=True)

    if not best:
        raise RuntimeError("No configs evaluated")

    # Final eval for best config (optionally on the full manifest).
    final_pairs = list(manifest["pairs"])
    if args.final_max_pairs:
        final_pairs = final_pairs[: args.final_max_pairs]
    best_cfg = OfflineSearchConfig(**best["config"])

    final_whisper = WhisperTranscriber(model_size=args.final_whisper_model, device=str(torch_device))
    final_dir = out_root / "best_final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "config.json").write_text(json.dumps(asdict(best_cfg), indent=2))

    final_record = _run_eval(
        converter=converter,
        pairs=final_pairs,
        sources=sources,
        targets=targets,
        out_dir=final_dir,
        cfg=best_cfg,
        whisper=final_whisper,
        speaker=speaker,
        accent=accent,
        emotion=emotion,
        target_speaker_emb=target_speaker_emb,
        target_accent_emb=target_accent_emb,
        target_emotion_emb=target_emotion_emb,
        keep_first_n_audio=int(args.keep_first_n_audio),
    )
    final_means = final_record["means"]
    final_record["score"] = _score(
        final_means,
        w_spk=float(args.w_spk),
        w_wer=float(args.w_wer),
        w_a_sim=float(args.w_a_sim),
        w_e_sim=float(args.w_e_sim),
        w_steps=float(args.w_steps),
        steps=int(best_cfg.flow_matching_steps),
    )
    (final_dir / "result.json").write_text(json.dumps(final_record, indent=2))
    (final_dir / "means.json").write_text(json.dumps(final_means, indent=2))
    print(f"[search] Best final means: {json.dumps(final_means)}", flush=True)
    print(f"[search] Wrote: {final_dir / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


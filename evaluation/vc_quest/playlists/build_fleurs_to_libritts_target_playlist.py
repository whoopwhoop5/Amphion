# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from evaluation.vc_quest.playlist import VCPlaylistManifest, VCPairItem, VCSourceItem, VCTargetItem, load_vc_playlist_manifest


def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def _copy_wav(src: str | Path, dst: str | Path) -> None:
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_p, dst_p)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic mixed playlist: FLEURS French sources + LibriTTS target reference."
    )
    parser.add_argument(
        "--fleurs_manifest",
        type=str,
        default="data/vc_quest_playlists/fleurs_fr_fr_dev_v1/manifest.json",
        help="Path to an existing FLEURS playlist manifest.json (sources come from here).",
    )
    parser.add_argument(
        "--libritts_target_manifest",
        type=str,
        required=True,
        help="Path to export_libritts_speaker_dataset manifest.json (provides target speaker + ref wav).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--num_sources",
        type=int,
        default=30,
        help="Number of FLEURS sources to include (0 means all).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory under data/vc_quest_playlists (defaults to a name derived from speaker_id).",
    )
    args = parser.parse_args(argv)

    fleurs_manifest_path = Path(args.fleurs_manifest).resolve()
    if not fleurs_manifest_path.is_file():
        raise FileNotFoundError(f"Missing fleurs_manifest: {fleurs_manifest_path}")

    target_manifest_path = Path(args.libritts_target_manifest).resolve()
    if not target_manifest_path.is_file():
        raise FileNotFoundError(f"Missing libritts_target_manifest: {target_manifest_path}")

    target_raw = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    target_meta = dict(target_raw.get("meta") or {})
    speaker_id = str(target_meta.get("speaker_id") or "").strip()
    ref_wav_path = str(target_meta.get("ref_wav_path") or "").strip()
    if not speaker_id:
        raise RuntimeError(f"Missing speaker_id in target manifest meta: {target_manifest_path}")
    if not ref_wav_path:
        raise RuntimeError(f"Missing ref_wav_path in target manifest meta: {target_manifest_path}")

    ref_wav_abs: Path
    ref_wav_p = Path(ref_wav_path)
    if ref_wav_p.is_absolute():
        candidates = [ref_wav_p]
    else:
        # Support both:
        # - paths relative to the target manifest dir (recommended)
        # - paths relative to the repo cwd (older exports sometimes included the out_dir prefix)
        candidates = [
            (target_manifest_path.parent / ref_wav_p),
            (Path.cwd() / ref_wav_p),
        ]

    for cand in candidates:
        if cand.is_file():
            ref_wav_abs = cand.resolve()
            break
    else:
        raise FileNotFoundError(f"ref_wav_path not found: tried {candidates}")

    out_dir = Path(str(args.out_dir).strip()) if str(args.out_dir).strip() else None
    if out_dir is None:
        out_dir = Path("data/vc_quest_playlists") / f"fleurs_fr_fr_to_libritts_s{speaker_id}_v1"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fleurs = load_vc_playlist_manifest(fleurs_manifest_path).resolve_paths(fleurs_manifest_path)
    sources = list(fleurs.sources)
    sources.sort(key=lambda s: _stable_hash(f"{int(args.seed)}:{s.id}"))
    if int(args.num_sources) > 0:
        sources = sources[: int(args.num_sources)]
    if not sources:
        raise RuntimeError("No sources selected from the FLEURS manifest.")

    # Copy sources (keep stable IDs/filenames).
    out_sources: list[VCSourceItem] = []
    for s in sources:
        rel = Path("wav") / "sources" / f"{s.id}.wav"
        dst = out_dir / rel
        _copy_wav(s.wav_path, dst)
        out_sources.append(VCSourceItem(**{**s.__dict__, "wav_path": str(rel)}))

    # Copy target reference wav.
    target_id = f"libritts_t{speaker_id}"
    rel_t = Path("wav") / "targets" / f"{target_id}.wav"
    _copy_wav(ref_wav_abs, out_dir / rel_t)
    out_targets = [
        VCTargetItem(
            id=target_id,
            dataset=str(target_meta.get("dataset") or target_meta.get("repo_id") or "mythicinfinity/libritts"),
            wav_path=str(rel_t),
            speaker_id=speaker_id,
            gender="",
        )
    ]

    pairs = [VCPairItem(source_id=s.id, target_id=target_id) for s in out_sources]

    manifest = VCPlaylistManifest(
        meta={
            "seed": int(args.seed),
            "source_playlist": {
                "type": "fleurs",
                "manifest": str(fleurs_manifest_path),
                "meta": dict(fleurs.meta),
                "num_sources": len(out_sources),
            },
            "target": {
                "type": "libritts",
                "manifest": str(target_manifest_path),
                "meta": target_meta,
            },
            "num_targets": 1,
            "num_pairs": len(pairs),
        },
        sources=out_sources,
        targets=out_targets,
        pairs=pairs,
    )

    out_manifest = out_dir / "manifest.json"
    out_manifest.write_text(
        json.dumps(
            {
                "meta": manifest.meta,
                "sources": [asdict(s) for s in manifest.sources],
                "targets": [asdict(t) for t in manifest.targets],
                "pairs": [asdict(p) for p in manifest.pairs],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[build_fleurs_to_libritts_target_playlist] Wrote: {out_manifest} (pairs={len(pairs)})")
    print(f"[build_fleurs_to_libritts_target_playlist] Wavs: {out_dir / 'wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

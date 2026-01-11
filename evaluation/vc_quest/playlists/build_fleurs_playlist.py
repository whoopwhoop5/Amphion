# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from huggingface_hub import HfApi, hf_hub_download

from evaluation.vc_quest.playlist import VCPlaylistManifest, VCPairItem, VCSourceItem, VCTargetItem


def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def _pick_deterministic(items: list[str], *, k: int, seed: int) -> list[str]:
    if k <= 0 or not items:
        return []
    items_sorted = sorted(items, key=lambda x: _stable_hash(f"{seed}:{x}"))
    return items_sorted[: min(k, len(items_sorted))]


def _extract_members(
    tar_gz_path: Path,
    *,
    members: set[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz_path, "r:gz") as tf:
        missing = set(members)
        for name in sorted(members):
            try:
                m = tf.getmember(name)
            except KeyError:
                continue

            missing.discard(name)
            if not m.isfile():
                continue
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe tar member path: {m.name}")

            out_path = out_dir / p
            out_path.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                raise RuntimeError(f"Failed to extract tar member: {m.name}")
            out_path.write_bytes(f.read())

        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} members in tar ({tar_gz_path.name}), e.g.: {sorted(list(missing))[:5]}"
            )


def _parse_fleurs_tsv(tsv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with tsv_path.open("r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        for parts in r:
            if not parts:
                continue
            # Format (no header): speaker_id, wav_name, raw_text, normalized_text, phones, num_samples, gender
            if len(parts) < 7:
                raise ValueError(f"Unexpected fleurs TSV columns={len(parts)} in line: {parts[:3]}")
            rows.append(
                {
                    "speaker_id": str(parts[0]),
                    "wav_name": str(parts[1]),
                    "raw_text": str(parts[2]),
                    "text": str(parts[3]),
                    "phones": str(parts[4]),
                    "num_samples": str(parts[5]),
                    "gender": str(parts[6]),
                }
            )
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic French-focused VC playlist from FLEURS.")
    parser.add_argument("--lang", type=str, default="fr_fr", help="FLEURS language ID (e.g., fr_fr).")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument(
        "--revision",
        type=str,
        default="d7c758a6dceecd54a98cac43404d3d576e721f07",
        help="Pinned HF dataset revision (commit).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_targets", type=int, default=10)
    parser.add_argument("--num_sources", type=int, default=30)
    parser.add_argument("--ref_min_sec", type=float, default=3.0)
    parser.add_argument("--ref_max_sec", type=float, default=12.0)
    parser.add_argument("--src_min_sec", type=float, default=2.0)
    parser.add_argument("--src_max_sec", type=float, default=12.0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/vc_quest_playlists/fleurs_fr_fr_dev_v1",
        help="Output directory (ignored by git).",
    )
    args = parser.parse_args(argv)

    lang = str(args.lang)
    split = str(args.split)
    revision = str(args.revision).strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    if revision:
        fleurs_rev = revision
    else:
        fleurs_rev = api.dataset_info("google/fleurs").sha

    tsv_filename = f"data/{lang}/{split}.tsv"
    tar_filename = f"data/{lang}/audio/{split}.tar.gz"

    tsv_path = Path(
        hf_hub_download("google/fleurs", repo_type="dataset", filename=tsv_filename, revision=fleurs_rev)
    )
    tar_path = Path(
        hf_hub_download("google/fleurs", repo_type="dataset", filename=tar_filename, revision=fleurs_rev)
    )

    rows = _parse_fleurs_tsv(tsv_path)
    # Sample counts correspond to 16kHz wavs in FLEURS.
    sr = 16000

    # Speaker -> utterances
    by_spk: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_spk.setdefault(r["speaker_id"], []).append(r)

    # Target speaker selection: must have at least one candidate reference.
    candidate_targets: list[str] = []
    for spk, utts in by_spk.items():
        ok = False
        for u in utts:
            dur = float(u["num_samples"]) / float(sr)
            if float(args.ref_min_sec) <= dur <= float(args.ref_max_sec):
                ok = True
                break
        if ok:
            candidate_targets.append(spk)

    target_speakers = _pick_deterministic(candidate_targets, k=int(args.num_targets), seed=int(args.seed))
    if not target_speakers:
        raise RuntimeError("No target speakers selected; check ref_min_sec/ref_max_sec or dataset split.")

    targets: list[VCTargetItem] = []
    target_wav_members: set[str] = set()
    used_target_speakers: set[str] = set()

    for spk in target_speakers:
        utts = list(by_spk.get(spk, []))
        # Pick the longest utterance in range (deterministic).
        cand: list[tuple[float, str, dict[str, str]]] = []
        for u in utts:
            dur = float(u["num_samples"]) / float(sr)
            if float(args.ref_min_sec) <= dur <= float(args.ref_max_sec):
                cand.append((dur, u["wav_name"], u))
        if not cand:
            continue
        cand.sort(key=lambda x: (-x[0], _stable_hash(x[1])))
        _, wav_name, u = cand[0]

        tid = f"fleurs_{lang}_{split}_t{spk}"
        rel_wav = Path("wav") / "targets" / f"{tid}.wav"
        targets.append(
            VCTargetItem(
                id=tid,
                dataset=f"google/fleurs:{lang}/{split}",
                wav_path=str(rel_wav),
                speaker_id=str(spk),
                gender=str(u.get("gender") or ""),
            )
        )
        target_wav_members.add(f"{split}/{wav_name}")
        used_target_speakers.add(spk)

    if not targets:
        raise RuntimeError("No targets found after filtering; check selection ranges.")

    # Source utterance selection (exclude target speakers to avoid trivial same-speaker content).
    source_candidates: list[dict[str, str]] = []
    for r in rows:
        spk = r["speaker_id"]
        if spk in used_target_speakers:
            continue
        dur = float(r["num_samples"]) / float(sr)
        if float(args.src_min_sec) <= dur <= float(args.src_max_sec):
            text = str(r.get("text") or "").strip()
            if len(text.split()) >= 3:
                source_candidates.append(r)

    if not source_candidates:
        raise RuntimeError("No source candidates found; check src_min_sec/src_max_sec.")

    # Deterministic pick of source utterances.
    source_candidates.sort(key=lambda r: _stable_hash(f"{args.seed}:{r['speaker_id']}:{r['wav_name']}"))
    chosen_sources = source_candidates[: min(int(args.num_sources), len(source_candidates))]

    sources: list[VCSourceItem] = []
    source_wav_members: set[str] = set()
    for r in chosen_sources:
        spk = str(r["speaker_id"])
        wav_name = str(r["wav_name"])
        sid = f"fleurs_{lang}_{split}_s{spk}_{Path(wav_name).stem}"
        rel_wav = Path("wav") / "sources" / f"{sid}.wav"
        sources.append(
            VCSourceItem(
                id=sid,
                dataset=f"google/fleurs:{lang}/{split}",
                wav_path=str(rel_wav),
                transcript=str(r.get("text") or "").strip(),
                speaker_id=spk,
                gender=str(r.get("gender") or ""),
            )
        )
        source_wav_members.add(f"{split}/{wav_name}")

    # Pairing: all sources x all targets (keeps evaluation stable).
    pairs: list[VCPairItem] = []
    for s in sources:
        for t in targets:
            pairs.append(VCPairItem(source_id=s.id, target_id=t.id))

    # Extract only the needed audio files.
    extract_dir = out_dir / "wav"
    _extract_members(tar_path, members=target_wav_members | source_wav_members, out_dir=extract_dir)

    # Move extracted wavs into our stable names under wav/{targets,sources}/
    # (keep a small extracted cache, but normalize filenames for stable manifests).
    (extract_dir / "targets").mkdir(parents=True, exist_ok=True)
    (extract_dir / "sources").mkdir(parents=True, exist_ok=True)

    # We need the wav_name mapping for renames. Recompute from the TSV parse.
    wav_name_by_target_id: dict[str, str] = {}
    for spk in used_target_speakers:
        # Pick again (same logic as above), but keep the wav_name.
        utts = list(by_spk.get(spk, []))
        cand: list[tuple[float, str]] = []
        for u in utts:
            dur = float(u["num_samples"]) / float(sr)
            if float(args.ref_min_sec) <= dur <= float(args.ref_max_sec):
                cand.append((dur, u["wav_name"]))
        if not cand:
            continue
        cand.sort(key=lambda x: (-x[0], _stable_hash(x[1])))
        wav_name_by_target_id[f"fleurs_{lang}_{split}_t{spk}"] = cand[0][1]

    wav_name_by_source_id: dict[str, str] = {}
    for r in chosen_sources:
        spk = str(r["speaker_id"])
        wav_name = str(r["wav_name"])
        sid = f"fleurs_{lang}_{split}_s{spk}_{Path(wav_name).stem}"
        wav_name_by_source_id[sid] = wav_name

    for t in targets:
        src = extract_dir / split / wav_name_by_target_id[t.id]
        dst = out_dir / t.wav_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    for s in sources:
        src = extract_dir / split / wav_name_by_source_id[s.id]
        dst = out_dir / s.wav_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    manifest = VCPlaylistManifest(
        meta={
            "seed": int(args.seed),
            "dataset": "google/fleurs",
            "lang": lang,
            "split": split,
            "fleurs_revision": fleurs_rev,
            "sample_rate_hz": sr,
            "num_targets": len(targets),
            "num_sources": len(sources),
            "num_pairs": len(pairs),
            "ref_min_sec": float(args.ref_min_sec),
            "ref_max_sec": float(args.ref_max_sec),
            "src_min_sec": float(args.src_min_sec),
            "src_max_sec": float(args.src_max_sec),
        },
        sources=sources,
        targets=targets,
        pairs=pairs,
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "meta": manifest.meta,
                "sources": [asdict(s) for s in manifest.sources],
                "targets": [asdict(t) for t in manifest.targets],
                "pairs": [asdict(p) for p in manifest.pairs],
            },
            indent=2,
        )
    )
    print(f"Wrote: {manifest_path}")
    print(f"Wavs: {out_dir/'wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

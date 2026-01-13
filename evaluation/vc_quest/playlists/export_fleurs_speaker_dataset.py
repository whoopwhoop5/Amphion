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
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download


def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


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


def _extract_members_flat(
    tar_gz_path: Path,
    *,
    members: list[str],
    out_wavs_dir: Path,
    prefix: str,
) -> list[dict[str, str]]:
    out_wavs_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[dict[str, str]] = []
    with tarfile.open(tar_gz_path, "r:gz") as tf:
        for name in members:
            try:
                m = tf.getmember(name)
            except KeyError as e:
                raise FileNotFoundError(f"Missing tar member: {name}") from e
            if not m.isfile():
                continue
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe tar member path: {m.name}")

            f = tf.extractfile(m)
            if f is None:
                raise RuntimeError(f"Failed to extract tar member: {m.name}")
            dst = out_wavs_dir / f"{prefix}_{p.name}"
            dst.write_bytes(f.read())
            extracted.append({"member": name, "wav_path": str(dst)})

    return extracted


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a per-speaker training dataset from FLEURS by extracting wavs from the HF tarball."
    )
    parser.add_argument("--lang", type=str, default="fr_fr", help="FLEURS language ID (e.g., fr_fr).")
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "test"])
    parser.add_argument(
        "--revision",
        type=str,
        default="d7c758a6dceecd54a98cac43404d3d576e721f07",
        help="Pinned HF dataset revision (commit).",
    )
    parser.add_argument("--speaker_id", type=str, required=True, help="FLEURS speaker_id to export (e.g., 1523).")
    parser.add_argument("--min_sec", type=float, default=2.0)
    parser.add_argument("--max_sec", type=float, default=12.0)
    parser.add_argument(
        "--max_files",
        type=int,
        default=500,
        help="Maximum number of utterances to export (0 means all).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic selection seed.")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory; writes wavs/ + manifest.json.",
    )
    args = parser.parse_args(argv)

    lang = str(args.lang)
    split = str(args.split)
    speaker_id = str(args.speaker_id)
    revision = str(args.revision).strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    fleurs_rev = revision or api.dataset_info("google/fleurs").sha

    tsv_filename = f"data/{lang}/{split}.tsv"
    tar_filename = f"data/{lang}/audio/{split}.tar.gz"

    tsv_path = Path(
        hf_hub_download("google/fleurs", repo_type="dataset", filename=tsv_filename, revision=fleurs_rev)
    )
    tar_path = Path(
        hf_hub_download("google/fleurs", repo_type="dataset", filename=tar_filename, revision=fleurs_rev)
    )

    rows = _parse_fleurs_tsv(tsv_path)
    sr = 16000  # FLEURS wavs are 16kHz.

    cand: list[dict[str, str]] = []
    for r in rows:
        if str(r.get("speaker_id")) != speaker_id:
            continue
        dur = float(r["num_samples"]) / float(sr)
        if float(args.min_sec) <= dur <= float(args.max_sec):
            cand.append(r)

    if not cand:
        raise RuntimeError(f"No candidates found for speaker_id={speaker_id} in {lang}/{split}.")

    # Deterministic subset (do not overfit ordering).
    cand.sort(key=lambda r: _stable_hash(f"{int(args.seed)}:{r['wav_name']}"))
    if int(args.max_files) > 0:
        cand = cand[: int(args.max_files)]

    members = [f"{split}/{r['wav_name']}" for r in cand]
    out_wavs = out_dir / "wavs"

    extracted = _extract_members_flat(tar_path, members=members, out_wavs_dir=out_wavs, prefix=f"{lang}_{split}_{speaker_id}")

    # Join metadata back onto extracted list.
    by_member = {f"{split}/{r['wav_name']}": r for r in cand}
    items: list[dict[str, str]] = []
    total_sec = 0.0
    for ex in extracted:
        r = by_member.get(ex["member"])
        if not r:
            continue
        dur = float(r["num_samples"]) / float(sr)
        total_sec += dur
        items.append(
            {
                "wav_path": ex["wav_path"],
                "speaker_id": str(r.get("speaker_id") or ""),
                "gender": str(r.get("gender") or ""),
                "num_samples": str(r.get("num_samples") or ""),
                "duration_sec": f"{dur:.3f}",
                "text": str(r.get("text") or ""),
            }
        )

    out_manifest = out_dir / "manifest.json"
    out_manifest.write_text(
        json.dumps(
            {
                "meta": {
                    "dataset": "google/fleurs",
                    "lang": lang,
                    "split": split,
                    "speaker_id": speaker_id,
                    "fleurs_revision": fleurs_rev,
                    "sample_rate_hz": sr,
                    "min_sec": float(args.min_sec),
                    "max_sec": float(args.max_sec),
                    "seed": int(args.seed),
                    "max_files": int(args.max_files),
                    "num_exported": len(items),
                    "total_duration_sec": float(total_sec),
                    "tar_path": str(tar_path),
                    "tsv_path": str(tsv_path),
                },
                "items": items,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[export_fleurs_speaker_dataset] Wrote {out_manifest} ({len(items)} files, {total_sec/60.0:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


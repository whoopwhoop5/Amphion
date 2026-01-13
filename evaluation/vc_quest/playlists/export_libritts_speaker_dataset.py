# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from huggingface_hub import HfApi, hf_hub_download


def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def _write_audio_as_wav(out_path: Path, *, audio: Any, target_sr: int) -> tuple[float, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav: np.ndarray
    sr: int

    if isinstance(audio, (bytes, bytearray, memoryview)):
        wav, sr = sf.read(io.BytesIO(bytes(audio)), dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        sr = int(sr)
    elif isinstance(audio, dict):
        if audio.get("bytes"):
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav[:, 0]
            sr = int(sr)
        elif audio.get("array") is not None:
            wav = np.asarray(audio["array"], dtype=np.float32).reshape(-1)
            sr = int(audio.get("sampling_rate") or audio.get("sr") or target_sr)
        else:
            raise ValueError(f"Unsupported audio dict keys: {list(audio.keys())}")
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    if int(sr) != int(target_sr):
        import librosa

        wav = librosa.resample(wav.astype(np.float32, copy=False), orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)

    sf.write(out_path, wav.astype(np.float32, copy=False), int(sr))
    return float(len(wav) / max(int(sr), 1)), int(sr)


@dataclass(frozen=True)
class _RowRef:
    score: int
    parquet_path: str
    local_path: str
    row_idx: int
    uid: str


def _list_parquets(repo_id: str, *, split: str, revision: str) -> list[str]:
    api = HfApi()
    prefix = f"data/{split}/"
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    parquets = sorted([f for f in files if f.startswith(prefix) and f.endswith(".parquet")])
    if not parquets:
        raise FileNotFoundError(f"No parquet shards found for {repo_id}:{split} under {prefix}")
    return parquets


def _pick_auto_speaker_id(
    parquet_locals: list[str],
    *,
    seed: int,
    min_utterances: int,
) -> str:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    counts: dict[str, int] = {}
    for local in parquet_locals:
        table = pq.read_table(local, columns=["speaker_id"])
        vc = pc.value_counts(table["speaker_id"])
        for v, c in zip(vc.field("values"), vc.field("counts")):
            spk = str(v.as_py() or "")
            if not spk:
                continue
            counts[spk] = counts.get(spk, 0) + int(c.as_py() or 0)

    if not counts:
        raise RuntimeError("Failed to scan any speaker_id values from parquet shards.")

    cand = [spk for spk, n in counts.items() if int(n) >= int(min_utterances)]
    if not cand:
        # Fall back to the top speakers by utterance count.
        top = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:50]
        cand = [spk for spk, _ in top]

    cand.sort(key=lambda spk: _stable_hash(f"{int(seed)}:{spk}"))
    chosen = cand[0]
    return chosen


def _scan_speaker_rows(
    parquet_paths: list[str],
    parquet_locals: list[str],
    *,
    speaker_id: str,
    seed: int,
) -> list[_RowRef]:
    import pyarrow.parquet as pq

    rows: list[_RowRef] = []
    for parquet_path, local in zip(parquet_paths, parquet_locals):
        table = pq.read_table(local, columns=["speaker_id", "id"])
        spk_col = table["speaker_id"]
        id_col = table["id"]
        for i in range(table.num_rows):
            spk = str(spk_col[i].as_py() or "")
            if spk != speaker_id:
                continue
            uid = str(id_col[i].as_py() or "")
            if not uid:
                continue
            score = _stable_hash(f"{int(seed)}|{uid}")
            rows.append(_RowRef(score=score, parquet_path=parquet_path, local_path=local, row_idx=int(i), uid=uid))

    rows.sort(key=lambda r: (int(r.score), r.uid))
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a per-speaker wav dataset from LibriTTS (HF parquet) for training-based VC (e.g., RVC)."
    )
    parser.add_argument("--repo_id", type=str, default="mythicinfinity/libritts")
    parser.add_argument("--split", type=str, default="train.clean.100", help="LibriTTS split folder (e.g., train.clean.100).")
    parser.add_argument(
        "--revision",
        type=str,
        default="",
        help="Optional pinned dataset revision (commit SHA). If empty, resolves latest.",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="",
        help="LibriTTS speaker_id to export. If empty, pick deterministically from the split.",
    )
    parser.add_argument("--auto_min_utterances", type=int, default=80, help="Min utterances for auto speaker selection.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min_sec", type=float, default=2.0)
    parser.add_argument("--max_sec", type=float, default=12.0)
    parser.add_argument("--max_files", type=int, default=500, help="Maximum number of utterances to export (0 means unlimited).")
    parser.add_argument(
        "--max_minutes",
        type=float,
        default=0.0,
        help="Optional cap on total exported duration in minutes (0 means unlimited).",
    )
    parser.add_argument("--target_sr", type=int, default=24000, help="Output wav sample rate (default: 24kHz).")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory; writes wavs/ + manifest.json.",
    )
    args = parser.parse_args(argv)

    repo_id = str(args.repo_id).strip()
    split = str(args.split).strip()
    seed = int(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    revision = str(args.revision).strip() or api.dataset_info(repo_id).sha

    parquet_paths = _list_parquets(repo_id, split=split, revision=revision)
    parquet_locals = [
        hf_hub_download(repo_id, filename=p, repo_type="dataset", revision=revision) for p in parquet_paths
    ]

    speaker_id = str(args.speaker_id).strip()
    if not speaker_id:
        speaker_id = _pick_auto_speaker_id(
            parquet_locals,
            seed=seed,
            min_utterances=int(args.auto_min_utterances),
        )
        print(f"[export_libritts_speaker_dataset] Auto-picked speaker_id={speaker_id}", flush=True)

    rows = _scan_speaker_rows(parquet_paths, parquet_locals, speaker_id=speaker_id, seed=seed)
    if not rows:
        raise RuntimeError(f"No utterances found for speaker_id={speaker_id} in split={split}.")

    import bisect
    import pyarrow.parquet as pq

    # Cache ParquetFile + row-group boundaries per shard for efficient row access.
    parquet_files: dict[str, pq.ParquetFile] = {}
    row_group_starts: dict[str, list[int]] = {}
    for local in parquet_locals:
        pf = pq.ParquetFile(local)
        starts: list[int] = []
        acc = 0
        for rg in range(pf.num_row_groups):
            starts.append(acc)
            acc += int(pf.metadata.row_group(rg).num_rows)
        parquet_files[local] = pf
        row_group_starts[local] = starts

    last_key: tuple[str, int] | None = None
    last_table = None

    def _read_row(local_path: str, *, row_idx: int) -> dict[str, Any]:
        nonlocal last_key, last_table

        starts = row_group_starts[local_path]
        rg = int(bisect.bisect_right(starts, int(row_idx)) - 1)
        if rg < 0:
            rg = 0
        offset = int(row_idx) - int(starts[rg])

        key = (local_path, rg)
        if last_key != key:
            last_key = key
            last_table = parquet_files[local_path].read_row_group(
                rg,
                columns=["audio", "text_normalized", "speaker_id", "id"],
            )

        assert last_table is not None
        audio = last_table["audio"][offset].as_py()
        text = str(last_table["text_normalized"][offset].as_py() or "")
        spk = str(last_table["speaker_id"][offset].as_py() or "")
        uid = str(last_table["id"][offset].as_py() or "")
        return {"audio": audio, "text": text, "speaker_id": spk, "uid": uid}

    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    total_sec = 0.0
    max_minutes = float(args.max_minutes)
    max_sec_total = max_minutes * 60.0 if max_minutes > 0 else 0.0

    for idx, r in enumerate(rows):
        if int(args.max_files) > 0 and len(exported) >= int(args.max_files):
            break
        if max_sec_total > 0 and float(total_sec) >= max_sec_total:
            break

        row = _read_row(r.local_path, row_idx=int(r.row_idx))
        if str(row.get("speaker_id") or "") != speaker_id:
            continue

        audio = row.get("audio")
        uid = str(row.get("uid") or r.uid)
        text = str(row.get("text") or "")

        rel_wav = Path("wavs") / f"{speaker_id}_{idx:05d}.wav"
        wav_path = out_dir / rel_wav
        dur_sec, sr = _write_audio_as_wav(wav_path, audio=audio, target_sr=int(args.target_sr))
        if not (float(args.min_sec) <= float(dur_sec) <= float(args.max_sec)):
            try:
                wav_path.unlink()
            except FileNotFoundError:
                pass
            continue

        exported.append(
            {
                "wav_path": str(rel_wav),
                "speaker_id": speaker_id,
                "uid": uid,
                "text": text,
                "duration_sec": float(dur_sec),
                "sample_rate_hz": int(sr),
                "parquet_path": r.parquet_path,
                "row_idx": int(r.row_idx),
            }
        )
        total_sec += float(dur_sec)

    if not exported:
        raise RuntimeError(
            f"No utterances exported for speaker_id={speaker_id} after duration filters ({args.min_sec}..{args.max_sec}s)."
        )

    # Pick a deterministic reference clip (longest in-range; tie-break by stable hash).
    ref = max(
        exported,
        key=lambda it: (
            float(it.get("duration_sec") or 0.0),
            -_stable_hash(str(it.get("uid") or "")),
        ),
    )

    out_manifest = out_dir / "manifest.json"
    out_manifest.write_text(
        json.dumps(
            {
                "meta": {
                    "dataset": repo_id,
                    "split": split,
                    "libritts_revision": revision,
                    "speaker_id": speaker_id,
                    "seed": seed,
                    "min_sec": float(args.min_sec),
                    "max_sec": float(args.max_sec),
                    "max_files": int(args.max_files),
                    "max_minutes": float(args.max_minutes),
                    "target_sr": int(args.target_sr),
                    "num_exported": len(exported),
                    "total_duration_sec": float(total_sec),
                    "ref_uid": str(ref.get("uid") or ""),
                    "ref_wav_path": str(ref.get("wav_path") or ""),
                },
                "items": exported,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[export_libritts_speaker_dataset] Wrote {out_manifest} ({len(exported)} files, {total_sec/60.0:.1f} min, speaker={speaker_id})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import hashlib
import json
import io
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download


RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


@dataclass(frozen=True)
class SourceItem:
    id: str
    dataset: str
    transcript: str
    wav_path: str
    accent: Optional[str] = None


@dataclass(frozen=True)
class TargetItem:
    id: str
    dataset: str
    wav_path: str
    accent: Optional[str] = None
    emotion: Optional[str] = None


@dataclass(frozen=True)
class PairItem:
    source_id: str
    target_id: str


def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def _write_audio_bytes_as_wav(
    out_path: Path,
    *,
    audio_bytes: bytes,
    target_sr: int = 24000,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Most HF Audio bytes are already wav containers, but normalize sample rate and channels
    # for Vevo compatibility.
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if int(sr) != target_sr:
        import librosa

        wav = librosa.resample(wav.astype(np.float32, copy=False), orig_sr=int(sr), target_sr=target_sr)
        sr = target_sr
    sf.write(out_path, wav.astype(np.float32, copy=False), int(sr))


def _download_globe_sources(
    *,
    out_dir: Path,
    num_sources: int,
    seed: int,
    revision: Optional[str] = None,
) -> list[SourceItem]:
    repo_id = "MushanW/GLOBE"
    parquet_path = "data/test-00000-of-00001.parquet"
    local = hf_hub_download(repo_id, parquet_path, repo_type="dataset", revision=revision)

    table = pq.read_table(local)
    # Columns: audio (struct bytes,path), transcript, accent, speaker_id, ...
    n = table.num_rows

    ids: list[str] = []
    scores: list[int] = []
    for i in range(n):
        spk = str(table["speaker_id"][i].as_py() or "")
        transcript = str(table["transcript"][i].as_py() or "")
        accent = str(table["accent"][i].as_py() or "")
        sid = f"globe_source:{spk}:{i}"
        ids.append(sid)
        scores.append(_stable_hash(f"{sid}|{accent}|{transcript}"))

    scores_np = np.asarray(scores, dtype=np.uint64)
    # deterministic shuffle by hash, then take first K
    order = np.argsort(scores_np, kind="stable")
    pick = order[:num_sources]

    sources: list[SourceItem] = []
    for idx in pick.tolist():
        row = {name: table[name][idx].as_py() for name in table.column_names}
        transcript = str(row.get("transcript") or "")
        if not transcript.strip():
            continue
        audio = row["audio"]
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            continue

        wav_path = out_dir / "sources" / f"globe_{idx:05d}.wav"
        _write_audio_bytes_as_wav(wav_path, audio_bytes=audio_bytes)

        sources.append(
            SourceItem(
                id=ids[idx],
                dataset=repo_id,
                transcript=transcript,
                wav_path=os.fspath(wav_path),
                accent=str(row.get("accent") or ""),
            )
        )

    # If we filtered empty transcript rows, top up deterministically.
    if len(sources) < num_sources:
        extra_needed = num_sources - len(sources)
        remaining = [i for i in order.tolist() if i not in set(pick.tolist())]
        for idx in remaining:
            if extra_needed <= 0:
                break
            row = {name: table[name][idx].as_py() for name in table.column_names}
            transcript = str(row.get("transcript") or "")
            audio = row["audio"]
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            if not transcript.strip() or not audio_bytes:
                continue
            wav_path = out_dir / "sources" / f"globe_{idx:05d}.wav"
            _write_audio_bytes_as_wav(wav_path, audio_bytes=audio_bytes)
            sources.append(
                SourceItem(
                    id=ids[idx],
                    dataset=repo_id,
                    transcript=transcript,
                    wav_path=os.fspath(wav_path),
                    accent=str(row.get("accent") or ""),
                )
            )
            extra_needed -= 1

    return sources[:num_sources]


def _download_globe_accent_targets(
    *,
    out_dir: Path,
    target_accents: list[str],
    revision: Optional[str] = None,
) -> list[TargetItem]:
    repo_id = "MushanW/GLOBE"
    parquet_path = "data/test-00000-of-00001.parquet"
    local = hf_hub_download(repo_id, parquet_path, repo_type="dataset", revision=revision)
    table = pq.read_table(local)

    def _slug(s: str) -> str:
        s = s.strip().lower()
        out = []
        for ch in s:
            if ch.isalnum():
                out.append(ch)
            elif ch in (" ", "-", "_"):
                out.append("_")
            else:
                out.append("_")
        slug = "".join(out)
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_") or "accent"

    # Find first example for each accent (deterministic by row order).
    targets: list[TargetItem] = []
    found = set()
    for idx in range(table.num_rows):
        accent = str(table["accent"][idx].as_py() or "")
        if accent not in target_accents or accent in found:
            continue
        audio = table["audio"][idx].as_py()
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not audio_bytes:
            continue

        wav_path = out_dir / "targets" / f"globe_accent_{_slug(accent)}.wav"
        _write_audio_bytes_as_wav(wav_path, audio_bytes=audio_bytes)
        tid = f"globe_target:{accent}:{idx}"
        targets.append(TargetItem(id=tid, dataset=repo_id, wav_path=os.fspath(wav_path), accent=accent))
        found.add(accent)
        if len(found) == len(target_accents):
            break

    missing = [a for a in target_accents if a not in found]
    if missing:
        raise RuntimeError(f"Missing requested GLOBE accents: {missing}")

    return targets


def _download_ravdess_emotion_targets(
    *,
    out_dir: Path,
    emotions: list[str],
    revision: Optional[str] = None,
) -> list[TargetItem]:
    repo_id = "birgermoell/ravdess"

    # Prefer Actor_01, statement 01, intensity 01, repetition 01.
    targets: list[TargetItem] = []
    for emo in emotions:
        # map emotion name -> code
        emo_code = None
        for k, v in RAVDESS_EMOTION_MAP.items():
            if v == emo:
                emo_code = k
                break
        if emo_code is None:
            raise ValueError(f"Unknown emotion: {emo}")

        # Filename pattern: modality-channel-emotion-intensity-statement-repetition-actor.wav
        # We choose speech modality=03, vocal channel=01.
        fname = f"Actor_01/03-01-{emo_code}-01-01-01-01.wav"
        local = hf_hub_download(repo_id, fname, repo_type='dataset', revision=revision)
        wav_path = out_dir / "targets" / f"ravdess_emotion_{emo}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(Path(local).read_bytes())
        targets.append(TargetItem(id=f"ravdess:{emo}", dataset=repo_id, wav_path=os.fspath(wav_path), emotion=emo))

    return targets


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic StyleStream-like eval manifest (small).")
    parser.add_argument("--out_dir", type=str, default="runs/stylestream_like")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_sources", type=int, default=100)

    parser.add_argument(
        "--globe_target_accents",
        type=str,
        default="England English|United States English|Australian English|Canadian English|Irish English",
        help="Target accents to pick from GLOBE-test (use '|' as separator).",
    )
    parser.add_argument(
        "--ravdess_target_emotions",
        type=str,
        default="happy,angry,sad,fearful,calm",
        help="Comma-separated emotions to pick from RAVDESS.",
    )
    parser.add_argument("--pairs_per_source", type=int, default=10)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    globe_rev = api.dataset_info("MushanW/GLOBE").sha
    ravdess_rev = api.dataset_info("birgermoell/ravdess").sha

    sources = _download_globe_sources(
        out_dir=out_dir,
        num_sources=args.num_sources,
        seed=args.seed,
        revision=globe_rev,
    )
    globe_targets = _download_globe_accent_targets(
        out_dir=out_dir,
        target_accents=[a.strip() for a in args.globe_target_accents.split("|") if a.strip()],
        revision=globe_rev,
    )
    ravdess_targets = _download_ravdess_emotion_targets(
        out_dir=out_dir,
        emotions=[e.strip() for e in args.ravdess_target_emotions.split(",") if e.strip()],
        revision=ravdess_rev,
    )

    targets = globe_targets + ravdess_targets
    targets_by_id = {t.id: t for t in targets}

    rng = np.random.default_rng(args.seed)
    pairs: list[PairItem] = []
    for s in sources:
        # Pick a fixed number of targets per source (deterministic with seed).
        pick = rng.choice(list(targets_by_id.keys()), size=min(args.pairs_per_source, len(targets)), replace=False)
        for tid in pick.tolist():
            pairs.append(PairItem(source_id=s.id, target_id=tid))

    manifest = {
        "meta": {
            "seed": int(args.seed),
            "globe_revision": globe_rev,
            "ravdess_revision": ravdess_rev,
            "num_sources": len(sources),
            "num_targets": len(targets),
            "num_pairs": len(pairs),
        },
        "sources": [asdict(s) for s in sources],
        "targets": [asdict(t) for t in targets],
        "pairs": [asdict(p) for p in pairs],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

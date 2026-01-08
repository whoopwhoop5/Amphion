# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'pyarrow' required for reading HF dataset parquet shards. "
        "Install it with `pip install -U pyarrow`."
    ) from e


@dataclass(frozen=True)
class PlaylistItem:
    id: str
    dataset: str
    transcript: str
    wav_path: str
    speaker_id: str = ""
    accent: str = ""


def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def _write_audio_as_wav(
    out_path: Path,
    *,
    audio: Any,
    target_sr: int = 24000,
    max_sec: float = 0.0,
    min_sec: float = 0.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav: np.ndarray
    sr: int
    if isinstance(audio, (bytes, bytearray, memoryview)):
        wav, sr = sf.read(io.BytesIO(bytes(audio)), dtype="float32")
        sr = int(sr)
    elif isinstance(audio, dict):
        if audio.get("bytes"):
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            sr = int(sr)
        elif audio.get("array") is not None:
            wav = np.asarray(audio["array"], dtype=np.float32).reshape(-1)
            sr = int(audio.get("sampling_rate") or audio.get("sr") or target_sr)
        else:
            raise ValueError(f"Unsupported audio dict keys: {list(audio.keys())}")
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    if wav.ndim > 1:
        wav = wav[:, 0]

    if int(sr) != int(target_sr):
        import librosa

        wav = librosa.resample(wav.astype(np.float32, copy=False), orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)

    if max_sec and max_sec > 0:
        max_samples = int(round(float(max_sec) * sr))
        if len(wav) > max_samples:
            wav = wav[:max_samples]

    dur_sec = float(len(wav)) / float(sr) if sr > 0 else 0.0
    if min_sec and min_sec > 0 and dur_sec < float(min_sec):
        return

    sf.write(out_path, wav.astype(np.float32, copy=False), int(sr))


def _download_globe_sources(
    *,
    out_dir: Path,
    num_sources: int,
    seed: int,
    revision: str,
    target_sr: int,
    max_sec: float,
    min_sec: float,
) -> list[PlaylistItem]:
    repo_id = "MushanW/GLOBE"
    parquet_path = "data/test-00000-of-00001.parquet"
    local = hf_hub_download(repo_id, parquet_path, repo_type="dataset", revision=revision)
    table = pq.read_table(local, columns=["audio", "transcript", "accent", "speaker_id"])

    candidates: list[tuple[int, str, str, str, Any]] = []
    for i in range(table.num_rows):
        transcript = str(table["transcript"][i].as_py() or "")
        if not transcript.strip():
            continue
        audio = table["audio"][i].as_py()
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not audio_bytes:
            continue
        accent = str(table["accent"][i].as_py() or "")
        speaker_id = str(table["speaker_id"][i].as_py() or "")
        sid = f"globe:{speaker_id}:{i}"
        candidates.append((i, sid, transcript, accent, audio))

    if not candidates:
        raise RuntimeError("No usable rows found in GLOBE parquet.")

    scores = [_stable_hash(f"{seed}|{sid}|{accent}|{transcript}") for _, sid, transcript, accent, _ in candidates]
    order = np.argsort(np.asarray(scores, dtype=np.uint64), kind="stable").tolist()

    items: list[PlaylistItem] = []
    playlist_dir = out_dir / "playlist"
    playlist_dir.mkdir(parents=True, exist_ok=True)
    for pick_idx in order:
        _, sid, transcript, accent, audio = candidates[pick_idx]
        wav_path = playlist_dir / f"globe_{len(items):05d}.wav"
        _write_audio_as_wav(
            wav_path,
            audio=audio,
            target_sr=target_sr,
            max_sec=max_sec,
            min_sec=min_sec,
        )
        if not wav_path.exists():
            continue
        items.append(
            PlaylistItem(
                id=sid,
                dataset=repo_id,
                transcript=transcript,
                wav_path=os.fspath(wav_path),
                speaker_id=sid.split(":", 2)[1] if ":" in sid else "",
                accent=accent,
            )
        )
        if len(items) >= num_sources:
            break

    return items


def _download_libritts_test_clean_sources(
    *,
    out_dir: Path,
    num_sources: int,
    seed: int,
    revision: str,
    target_sr: int,
    max_sec: float,
    min_sec: float,
) -> list[PlaylistItem]:
    repo_id = "mythicinfinity/libritts"
    parquet_paths = [
        "data/test.clean/test.clean-00000-of-00004.parquet",
        "data/test.clean/test.clean-00001-of-00004.parquet",
        "data/test.clean/test.clean-00002-of-00004.parquet",
        "data/test.clean/test.clean-00003-of-00004.parquet",
    ]

    rows: list[tuple[str, str, str, Any]] = []
    for shard, parquet_path in enumerate(parquet_paths):
        local = hf_hub_download(repo_id, parquet_path, repo_type="dataset", revision=revision)
        table = pq.read_table(local, columns=["audio", "text_normalized", "speaker_id", "id"])
        for i in range(table.num_rows):
            transcript = str(table["text_normalized"][i].as_py() or "")
            speaker_id = str(table["speaker_id"][i].as_py() or "")
            uid = str(table["id"][i].as_py() or "")
            audio = table["audio"][i].as_py()
            sid = f"libritts:{speaker_id}:{uid}:{shard}:{i}"
            rows.append((sid, transcript, speaker_id, audio))

    scores = [_stable_hash(f"{seed}|{sid}|{transcript}") for sid, transcript, _, _ in rows]
    order = np.argsort(np.asarray(scores, dtype=np.uint64), kind="stable").tolist()

    items: list[PlaylistItem] = []
    playlist_dir = out_dir / "playlist"
    playlist_dir.mkdir(parents=True, exist_ok=True)
    for idx in order:
        sid, transcript, speaker_id, audio = rows[idx]
        if not transcript.strip():
            continue
        if not isinstance(audio, dict) or not audio.get("bytes"):
            continue
        wav_path = playlist_dir / f"libritts_{len(items):05d}.wav"
        _write_audio_as_wav(
            wav_path,
            audio=audio,
            target_sr=target_sr,
            max_sec=max_sec,
            min_sec=min_sec,
        )
        if not wav_path.exists():
            continue
        items.append(
            PlaylistItem(
                id=sid,
                dataset=repo_id,
                transcript=transcript,
                wav_path=os.fspath(wav_path),
                speaker_id=speaker_id,
                accent="",
            )
        )
        if len(items) >= num_sources:
            break

    return items


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic 24kHz wav playlist for Vevo live evaluation.")
    parser.add_argument("--out_dir", type=str, default="runs/vevo_live/eval_playlist")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_sources", type=int, default=30)
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument("--max_sec", type=float, default=8.0, help="Trim each clip to at most this many seconds (0 to disable).")
    parser.add_argument(
        "--min_sec",
        type=float,
        default=6.0,
        help="Require each clip to be at least this long after resampling/trimming (0 to disable).",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="globe",
        choices=["globe", "libritts_test_clean"],
        help="Source dataset to sample from.",
    )
    parser.add_argument("--globe_revision", type=str, default="", help="Optional dataset revision for MushanW/GLOBE")
    parser.add_argument("--libritts_revision", type=str, default="", help="Optional dataset revision for mythicinfinity/libritts")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    playlist_dir = out_dir / "playlist"
    if playlist_dir.exists():
        for p in playlist_dir.glob("*.wav"):
            p.unlink()
    playlist_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    globe_rev = str(args.globe_revision).strip() or api.dataset_info("MushanW/GLOBE").sha
    libritts_rev = str(args.libritts_revision).strip() or api.dataset_info("mythicinfinity/libritts").sha

    items: list[PlaylistItem]
    if args.dataset == "globe":
        items = _download_globe_sources(
            out_dir=out_dir,
            num_sources=int(args.num_sources),
            seed=int(args.seed),
            revision=globe_rev,
            target_sr=int(args.target_sr),
            max_sec=float(args.max_sec),
            min_sec=float(args.min_sec),
        )
        used = {"dataset": "globe", "repo_id": "MushanW/GLOBE", "revision": globe_rev}
    else:
        items = _download_libritts_test_clean_sources(
            out_dir=out_dir,
            num_sources=int(args.num_sources),
            seed=int(args.seed),
            revision=libritts_rev,
            target_sr=int(args.target_sr),
            max_sec=float(args.max_sec),
            min_sec=float(args.min_sec),
        )
        used = {"dataset": "libritts_test_clean", "repo_id": "mythicinfinity/libritts", "revision": libritts_rev}

    manifest = {
        "meta": {
            "seed": int(args.seed),
            "num_sources": int(args.num_sources),
            "target_sr": int(args.target_sr),
            "max_sec": float(args.max_sec),
            "min_sec": float(args.min_sec),
            **used,
        },
        "items": [asdict(i) for i in items],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[build_playlist] wrote {len(items)} wavs -> {out_dir / 'playlist'}", flush=True)
    print(f"[build_playlist] manifest: {out_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

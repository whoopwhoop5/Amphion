# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VCSourceItem:
    id: str
    dataset: str
    wav_path: str
    transcript: str
    speaker_id: str = ""
    gender: str = ""


@dataclass(frozen=True)
class VCTargetItem:
    id: str
    dataset: str
    wav_path: str
    speaker_id: str = ""
    gender: str = ""


@dataclass(frozen=True)
class VCPairItem:
    source_id: str
    target_id: str


@dataclass(frozen=True)
class VCPlaylistManifest:
    meta: dict[str, Any]
    sources: list[VCSourceItem]
    targets: list[VCTargetItem]
    pairs: list[VCPairItem]

    def resolve_paths(self, manifest_path: str | Path) -> "VCPlaylistManifest":
        base = Path(manifest_path).resolve().parent

        def _abs(p: str) -> str:
            pp = Path(p)
            return str(pp if pp.is_absolute() else (base / pp))

        return VCPlaylistManifest(
            meta=dict(self.meta),
            sources=[
                VCSourceItem(**{**s.__dict__, "wav_path": _abs(s.wav_path)}) for s in self.sources
            ],
            targets=[
                VCTargetItem(**{**t.__dict__, "wav_path": _abs(t.wav_path)}) for t in self.targets
            ],
            pairs=list(self.pairs),
        )

    def sources_by_id(self) -> dict[str, VCSourceItem]:
        return {s.id: s for s in self.sources}

    def targets_by_id(self) -> dict[str, VCTargetItem]:
        return {t.id: t for t in self.targets}


def load_vc_playlist_manifest(path: str | Path) -> VCPlaylistManifest:
    p = Path(path)
    raw = json.loads(p.read_text())
    sources = [VCSourceItem(**s) for s in raw["sources"]]
    targets = [VCTargetItem(**t) for t in raw["targets"]]
    pairs = [VCPairItem(**x) for x in raw["pairs"]]
    return VCPlaylistManifest(meta=dict(raw.get("meta") or {}), sources=sources, targets=targets, pairs=pairs)


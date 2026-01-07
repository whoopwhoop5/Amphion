## Continuity Ledger
- Goal (incl. success criteria): Build a Vevo-powered offline + live buffered voice conversion pipeline (≈1s latency) with a deterministic evaluation/auto-tuning harness and regression suite.
- Constraints/Assumptions: Use Amphion’s official Vevo; enforce stable timing in a streaming wrapper; run heavy inference/eval on Vast RTX 4090 (CUDA 12.9); use git for sync; avoid committing secrets.
- Key decisions: UNCONFIRMED
- Done: Initialized Beads tracking (`bd init`).
- Now: Survey existing Vevo inference entrypoints and model requirements; define wrapper API for offline + streaming.
- Next: Implement offline CLI; implement live buffered mic->convert->playback; add deterministic eval harness + parameter search; add regression suite; wire GPU-host run scripts.
- Open questions (UNCONFIRMED if needed): Target sample rate & hop sizes for Vevo; best duration-normalization strategy for minimal artifacts in streaming; preferred audio I/O backend on macOS (sounddevice vs. alternatives).
- Working set (files/ids/commands): `CONTINUITY.md`, `.beads/*`, `bd list`, `bd ready`

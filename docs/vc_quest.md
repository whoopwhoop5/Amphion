# VC Quest: Low-Latency Timbre VC (vs Vevo)

Goal: Find a timbre-focused VC pipeline that can run with **smaller streaming context** (target: **<= 600ms chunks**) while matching or exceeding our current **Vevo `vevotimbre`** quality.

Key constraints:
- Real-time is the priority (live calls).
- Deterministic + repeatable evaluation (use existing Amphion eval metrics where possible).
- Prefer **reference-clip** conditioning (no target-speaker training) when feasible; if training is required, document data requirements explicitly.

## Baseline (Vevo)
- Model: Amphion Vevo `vevotimbre`
- Current best live-like config (RTX 4090): `window_ms=2000`, `hop_ms=500`, `flow_matching_steps=6`, `fade_ms=10`
- Known limitation: needs ~2s context to stay intelligible/clean; startup delay ~2s+.
- Latest live fix: wrapper-level VAD + limiter; removed per-window RMS normalization (reduces noise/cutouts).

## Candidates (in progress)
### 1) OpenVoice (tone color conversion)
- Bead: `Amphion-ehh.1`
- Status: in_progress
- Hypothesis: zero-shot tone-color conversion could allow <=600ms chunking with acceptable quality.
- Next: fork/clone, implement offline + streaming simulation wrapper, run same metrics as Vevo.

### 2) FreeVC
- Bead: `Amphion-ehh.2`
- Status: open
- Hypothesis: strong zero-shot VC baseline; may stream with overlap-add.

## What we record for each candidate
- **Streaming config:** sample rate, chunk/window, hop, crossfade/OLA, VAD settings, any lookahead.
- **Speed:** mean/p95 chunk processing time, estimated RTF on RTX 4090.
- **Objective quality:** Whisper WER (aligned), speaker similarity (WavLM), artifact metrics (silence leakage, dropout, clipping).
- **Artifacts for listening:** offline + live-sim wavs for the same source/reference pair.


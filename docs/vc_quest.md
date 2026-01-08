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
- Status: evaluated (likely reject)
- Hypothesis: zero-shot tone-color conversion could allow <=600ms chunking with acceptable quality.
- Implementation: `evaluation/vc_quest/openvoice_convert.py` + `scripts/vc_quest/openvoice_*`
- Artifacts: `runs/vc_quest/openvoice/user_pair/*`
- Setup notes:
  - OpenVoice V2 converter checkpoints from MyShell S3.
  - Installed without upstream deps (to avoid `faster-whisper` / PyAV build), then added minimal text deps for import chain.
  - Use `OpenVoiceBaseClass` directly (avoids optional `wavmark` watermark dependency).
- Results (RTX 4090, window=600ms, hop=600ms, fade=10ms, tau=0.3):
  - Speed: ~30ms per 600ms window (mean), so comfortably real-time.
  - Quality (Whisper `base` WER on our French samples is very high; speaker sim is high):
    - `v5_to_fr_offline`: S-SIM≈0.969, WER≈0.835
    - `fr_to_v5_offline`: S-SIM≈0.934, WER≈0.693
    - `v5_to_fr_stream`:  S-SIM≈0.962, WER≈0.927
    - `fr_to_v5_stream`:  S-SIM≈0.949, WER≈0.705
- Conclusion: extremely fast and high speaker similarity, but weak content preservation (WER), consistent with OpenVoice being designed for tone-color conversion rather than robust any-to-any VC. Keep as a reference, but move on to FreeVC.

### 2) FreeVC
- Bead: `Amphion-ehh.2`
- Status: open
- Hypothesis: strong zero-shot VC baseline; may stream with overlap-add.

## What we record for each candidate
- **Streaming config:** sample rate, chunk/window, hop, crossfade/OLA, VAD settings, any lookahead.
- **Speed:** mean/p95 chunk processing time, estimated RTF on RTX 4090.
- **Objective quality:** Whisper WER (aligned), speaker similarity (WavLM), artifact metrics (silence leakage, dropout, clipping).
- **Artifacts for listening:** offline + live-sim wavs for the same source/reference pair.

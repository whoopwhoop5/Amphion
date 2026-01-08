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
- Status: in_progress
- Hypothesis: strong zero-shot VC baseline; may stream with overlap-add.
- Implementation: `evaluation/vc_quest/freevc_convert.py` + `scripts/vc_quest/freevc_*`
- Setup notes:
  - Uses FreeVC repo under `~/deps/FreeVC` on GPU host (cloned by setup script).
  - Downloads pretrained checkpoints from HF Space `OlaWod/FreeVC`.
  - Uses HF `microsoft/wavlm-large` for content (note: authors mention HF ckpt differs slightly vs their training ckpt).
  - Output sample-rate is inferred from `upsample_rates` product (FreeVC-24 outputs 24kHz).
- Results (RTX 4090, variant=freevc-24, window=600ms, hop=600ms, fade=10ms):
  - Speed: ~77ms per 600ms window (mean), so real-time budget is comfortable.
  - Quality (Whisper `base` WER on our French sample pair is still high; stream can be worse than offline):
    - `v5_to_fr_offline`: S-SIM≈0.952, WER≈0.659
    - `fr_to_v5_offline`: S-SIM≈0.880, WER≈0.847
    - `v5_to_fr_stream`:  S-SIM≈0.938, WER≈0.945  (significant degradation)
    - `fr_to_v5_stream`:  S-SIM≈0.887, WER≈0.648
  - Artifacts: noticeable silence leakage/noise on some runs (e.g., silent_out_db_p95 ≈ -30dB on v5_to_fr).
- Grid search v1 (RMS VAD, emit_align=end):
  - Run: `runs/vc_quest/freevc/user_pair_search/*`
  - Selector: `evaluation/vc_quest/select_best.py` (quality-tier: min S-SIM>=0.85, silent_out_p95<=-25dB, dropout<=0.01, realtime)
  - Best (quality-tier): `window_ms=600, hop_ms=300` (RTF_p95≈0.234)
    - `v5_to_fr_stream_w600_h300`:  S-SIM≈0.929, WER≈0.872, silent_out_p95≈-35.0dB, drop≈0.0002
    - `fr_to_v5_stream_w600_h300`:  S-SIM≈0.868, WER≈0.677, silent_out_p95≈-36.6dB, drop≈0.0036
  - Observation: overlap helps stability/noise, but streaming still degrades content vs offline (esp. `v5_to_fr`).

- Grid search v2 (WebRTC VAD + hangover, crossfade-prefix, emit_align=center):
  - Code: `evaluation/vc_quest/freevc_convert.py` + `evaluation/vc_quest/streaming_utils.py`
  - Run: `runs/vc_quest/freevc/user_pair_search_webrtc_center/*`
  - Default search script: `scripts/vc_quest/freevc_search_user_pair_gpu.sh`
  - Top candidates (quality-tier, RTX 4090):
    - **Best by mean WER:** `window_ms=800, hop_ms=200` (mean_WER≈0.580, min_S-SIM≈0.858, leak_p95≈-30.2dB, drop≈0.004, RTF_p95≈0.359)
    - **Best by speaker/noise:** `window_ms=800, hop_ms=400` (mean_WER≈0.594, min_S-SIM≈0.875, leak_p95≈-32.2dB, drop≈0.003, RTF_p95≈0.174)
  - Observation: WebRTC VAD + crossfade reduces silence noise substantially and improves `fr_to_v5` content vs v1; some configs still produce loud silence leakage (e.g. `w600_h400`), so selection must gate on artifacts.

- Next:
  - Listen to the v2 top candidates (`w800_h200` vs `w800_h400`) and decide whether FreeVC meets our real-time quality bar.
  - If not, move to the next VC candidate (likely target-trained real-time VC) while keeping Vevo as the high-quality baseline.

## What we record for each candidate
- **Streaming config:** sample rate, chunk/window, hop, crossfade/OLA, VAD settings, any lookahead.
- **Speed:** mean/p95 chunk processing time, estimated RTF on RTX 4090.
- **Objective quality:** Whisper WER (aligned), speaker similarity (WavLM), artifact metrics (silence leakage, dropout, clipping).
- **Artifacts for listening:** offline + live-sim wavs for the same source/reference pair.

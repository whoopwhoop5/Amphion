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
- User-pair scoring (Whisper `base` WER + WavLM speaker similarity; scored on macOS; outputs in `runs/vevo_live/user_pair/*`):
  - Offline:
    - `v5_to_fr_offline`: S-SIM≈0.888, WER≈0.762
    - `fr_to_v5_offline`: S-SIM≈0.963, WER≈0.432
  - Online (these are the files you listened to and reported “noise/cutouts”):
    - `v5_to_fr_online`: S-SIM≈0.914, WER≈0.866, dropout≈0.036
    - `fr_to_v5_online`: S-SIM≈0.984, WER≈0.636, dropout≈0.123, silence_leak_p95≈-21.8dB
- Takeaway: Vevo offline is strong, but our current Vevo streaming artifacts (dropouts / silence leakage) show up clearly in objective metrics.

## Next candidates (actionable backlog)
Actionable now (public repo + downloadable weights/checkpoints):
- **TinyVC** (`uthree/tinyvc`) — real-time SOLA streaming; pretrained `encoder.pt`/`decoder.pt` on HF. (Bead: `Amphion-ehh.12`, in progress)
- **FragmentVC** (`yistLin/FragmentVC`) — pretrained available; streaming behavior unclear (may need wrapper-level chunking).
- **HiFi-VC** (`tinkoff-ai/hifi_vc`, archived) — checkpoint exists; likely offline-only; can still simulate streaming via wrapper.

Likely out-of-scope for “no target training” (but can be revisited if we accept per-target models):
- **LLVC** (`KoeAI/LLVC`) — any-to-one / per-target-speaker model.

Unclear fit / needs investigation:
- **SPARC** (`Berkeley-Speech-Group/Speech-Articulatory-Coding`) — may not provide an end-to-end VC pipeline.

Not actionable yet (paper/demo only or no public checkpoints):
- CONAN, RT-VC, ALO-VC, StreamVC (no checkpoint), DiffVC+ (no checkpoint), PFlow-VC (demo), ReFlow-VC (demo).

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
- Status: evaluated (promising)
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

### 3) MeanVC (streaming zero-shot VC)
- Bead: `Amphion-ehh.5`
- Status: evaluated (likely reject)
- Hypothesis: purpose-built streaming zero-shot VC can hit <=200ms chunks with stable timing and strong timbre transfer.
- Implementation: `evaluation/vc_quest/meanvc_convert.py` + `scripts/vc_quest/meanvc_*`
- Setup notes:
  - Uses MeanVC repo under `~/deps/MeanVC` on GPU host (cloned by setup script).
  - Downloads inference checkpoints via HF model `ASLP-lab/MeanVC` (`download_ckpt.py`).
  - Downloads speaker verification checkpoint (`wavlm_large_finetune.pth`) via Google Drive using `gdown` (per MeanVC README).
  - Important: `vocos.pt` TorchScript `.decode()` is unstable on CUDA in our environment (fails after repeated calls with `UNSUPPORTED DTYPE: complex`), so our wrapper runs vocoder decode on CPU (fast, ~10ms per 200ms chunk).
- Artifacts: `runs/vc_quest/meanvc/user_pair/*`
- Results (RTX 4090, steps=2, window=200ms, hop=200ms, fade=10ms, WebRTC VAD):
  - Speed: `p95_window_sec≈0.050s` => `RTF_p95≈0.25` (real-time is easy).
  - Quality (on our user pair) is **not competitive**:
    - Offline:
      - `v5_to_fr_offline`:  S-SIM≈0.718, WER≈0.935
      - `fr_to_v5_offline`:  S-SIM≈0.909, WER≈1.000
    - Stream:
      - `v5_to_fr_stream_w200_h200`: S-SIM≈0.744, WER≈0.942, dropout≈0.084
      - `fr_to_v5_stream_w200_h200`: S-SIM≈0.886, WER≈1.011, dropout≈0.043
  - Conclusion: extremely low-latency but weak content preservation (high WER) and noticeable dropouts; likely reject.

### 4) Seed-VC (zero-shot VC)
- Bead: `Amphion-ehh.6`
- Status: evaluated (likely reject for streaming)
- Hypothesis: Seed-VC may provide strong any-to-any VC quality while supporting small streaming chunks.
- Implementation: `evaluation/vc_quest/seedvc_convert.py` + `scripts/vc_quest/seedvc_*`
- Setup notes:
  - Uses Seed-VC repo under `~/deps/seed-vc` on GPU host (cloned by setup script).
  - Uses the **realtime** V1 checkpoint/config by default (`DiT_uvit_tat_xlsr_ema.pth` + `config_dit_mel_seed_uvit_xlsr_tiny.yml`).
  - Seed-VC’s `modules/` is a namespace package (no `__init__.py`) and collides with Amphion’s `modules/`; our wrapper removes Amphion repo-root from `sys.path` during Seed-VC imports.
  - Requires `descript-audio-codec` (`dac.*`) even for the xlsr-tiny preset.
- Artifacts: `runs/vc_quest/seedvc/user_pair/*`
- Results (RTX 4090, sr=22.05kHz, steps=10, cfg=0.7, max_prompt=3s):
  - Offline:
    - `v5_to_fr_offline`:  S-SIM≈0.946, WER≈0.610, leak_p95≈-43.8dB, drop≈0.000
    - `fr_to_v5_offline`:  S-SIM≈0.860, WER≈0.318, leak_p95≈-52.7dB, drop≈0.000
  - Streaming (SOLA-style wrapper, VAD=rms @ -55dB, extra_ce=2.5s, extra=0.5s, right=0.02s, crossfade=40ms):
    - `w300/h300`: p95_window≈0.136s => RTF_p95≈0.45
      - `v5_to_fr_stream_w300_h300`: S-SIM≈0.926, WER≈0.470, leak_p95≈-18.6dB, drop≈0.048
      - `fr_to_v5_stream_w300_h300`: S-SIM≈0.953, WER≈0.307, leak_p95≈-9.21dB, drop≈0.056
    - `w600/h600`: p95_window≈0.139s => RTF_p95≈0.23
      - `v5_to_fr_stream_w600_h600`: S-SIM≈0.928, WER≈0.671, leak_p95≈-23.9dB, drop≈0.072
      - `fr_to_v5_stream_w600_h600`: S-SIM≈0.948, WER≈0.636, leak_p95≈-12.5dB, drop≈0.098
  - Conclusion: offline quality is decent, but streaming has **very loud silence leakage** and **high voiced dropouts** vs our artifact gates; likely reject for real-time timbre VC unless we can drastically improve silence handling/alignment.

### 5) kNN-VC (nearest-neighbor VC)
- Bead: `Amphion-ehh.7`
- Status: evaluated (likely reject)
- Hypothesis: kNN-VC can be streamed with small windows (<=600ms); main costs are WavLM feature extraction + kNN search + HiFiGAN.
- Implementation: `evaluation/vc_quest/knnvc_convert.py` + `scripts/vc_quest/knnvc_*`
- Notes:
  - Requires ~16kHz I/O; output is 16kHz.
  - Quality usually improves with more reference speech; we start with our 10s reference clips.
  - Wrapper default disables per-window loudness normalization (`tgt_loudness_db=none`) to avoid gain pumping; we rely on our artifact gates + peak limiter.
- Artifacts: `runs/vc_quest/knnvc/user_pair/*`
- Results (RTX 4090, window=600ms, hop=300ms, fade=10ms, VAD=WebRTC):
  - Speed: mean per window ≈ 35ms => RTF ≈ 0.06 (very fast).
  - Offline:
    - `v5_to_fr_offline`: S-SIM≈0.929, WER≈0.901, leak_p95≈-47.9dB, drop≈0.0005
    - `fr_to_v5_offline`: S-SIM≈0.977, WER≈0.909, leak_p95≈-32.6dB, drop≈0.0000
  - Stream:
    - `v5_to_fr_stream`: S-SIM≈0.944, WER≈0.872, leak_p95≈-46.4dB, drop≈0.0246
    - `fr_to_v5_stream`: S-SIM≈0.977, WER≈0.870, leak_p95≈-39.0dB, drop≈0.0008
  - Conclusion: extremely fast and high speaker similarity, but weak content preservation (high WER) and occasional streaming dropouts; likely reject for our use case.

### 6) SpeechT5 (seq2seq voice conversion)
- Bead: `Amphion-ehh.8`
- Status: evaluated (reject)
- Hypothesis: strong content preservation (ASR-friendly) with decent timbre transfer via speaker embeddings; may be slower and may have length drift chunk-to-chunk because it is not designed for streaming.
- Implementation: `evaluation/vc_quest/speecht5_convert.py` + `scripts/vc_quest/speecht5_*`
- Notes:
  - Uses `microsoft/speecht5_vc` + `microsoft/speecht5_hifigan`.
  - Speaker conditioning uses WavLM x-vector (`microsoft/wavlm-base-plus-sv`) and L2-normalization (no extra `speechbrain` dependency).
  - Output is 16kHz; wrapper enforces fixed window/hop timing via length normalization + crossfade.
- Artifacts: `runs/vc_quest/speecht5/user_pair/*`
- Results (RTX 4090, window=600ms, hop=300ms, fade=10ms, VAD=WebRTC):
  - Speed: `p95_window_sec≈1.55s` => `RTF_p95≈2.6` (not real-time).
  - Offline (already bad):
    - `v5_to_fr_offline`: S-SIM≈0.771, WER≈0.986, leak_p95≈-13.6dB, drop≈0.053
    - `fr_to_v5_offline`: S-SIM≈0.687, WER≈2.57, leak_p95≈-16.1dB, drop≈0.081
  - Stream (also bad):
    - `v5_to_fr_stream`: S-SIM≈0.703, WER≈0.973, leak_p95≈-17.7dB, drop≈0.064
    - `fr_to_v5_stream`: S-SIM≈0.505, WER≈1.00, leak_p95≈-20.9dB, drop≈0.045
  - Conclusion: fails real-time budget and artifact gates (very loud silence leakage + dropouts) and does not preserve content; reject.

### 7) EZ-VC (F5-TTS based, zero-shot any-to-any VC)
- Bead: `Amphion-ehh.9`
- Status: evaluated (reject)
- Hypothesis: unit-based non-autoregressive generation may preserve content better than “tone color” systems while still supporting <=600ms streaming via wrapper-level timing normalization.
- Implementation: `evaluation/vc_quest/ezvc_convert.py` + `scripts/vc_quest/ezvc_*`
- Setup notes:
  - Uses upstream repo `EZ-VC/EZ-VC` as a package (`f5_tts`) + an espnet fork for XEUS unit extraction.
  - **Model weights are gated on HF** (`SPRINGLab/EZ-VC`), so the GPU host must be authenticated once via `huggingface-cli login`.
  - BigVGAN vocoder import collides with Amphion's `utils/` package; wrapper forces BigVGAN directory to the front of `sys.path` so `from utils import ...` resolves correctly.
  - Vocos fallback is not compatible with EZ-VC's mel config (expects 100 mel channels vs model's 80), so we use BigVGAN (`SPRINGLab/bigvgan_16khz`).
- Results (RTX 4090, `nfe_step=12, cfg=2.0, sway=-1.0`, stream `window=600ms, hop=300ms, fade=10ms`, VAD=WebRTC):
  - Speed: `mean_window_sec≈0.43s` (=> not realtime at `hop=300ms`; would fit at `hop=600ms`).
  - Offline:
    - `v5_to_fr_offline`: S-SIM≈0.958, WER≈0.955, leak_p95≈-37.9dB, drop≈0.0566
    - `fr_to_v5_offline`: S-SIM≈0.900, WER≈0.623, leak_p95≈-8.1dB, drop≈0.0541
  - Stream:
    - `v5_to_fr_stream`:  S-SIM≈0.963, WER≈0.856, leak_p95≈-28.2dB, drop≈0.0220
    - `fr_to_v5_stream`:  S-SIM≈0.912, WER≈0.826, leak_p95≈-6.8dB, drop≈0.0109
- Artifacts: `runs/vc_quest/ezvc/user_pair/*`
- Conclusion: high speaker similarity, but content preservation is very weak (high WER) and silence leakage can be extremely loud (p95 near -6dB); reject for real-time voice calls.

### 8) YingMusic-SVC (zero-shot singing VC)
- Bead: `Amphion-ehh.10`
- Status: evaluated (promising, but borderline realtime)
- Hypothesis: may transfer timbre well, but as a singing-focused model it may struggle on speech; we still evaluate as a candidate.
- Implementation: `evaluation/vc_quest/yingmusic_svc_convert.py` + `scripts/vc_quest/yingmusic_svc_*`
- Setup notes:
  - Checkpoint: `GiantAILab/YingMusic-SVC` (`YingMusic-SVC-full.pt`).
  - Uses Whisper semantic encoder + RMVPE F0 + CAMPPlus speaker embedding + BigVGAN vocoder (44.1kHz output).
  - Name collision: YingMusic-SVC uses a top-level `modules/` package; wrapper temporarily removes Amphion root from `sys.path` during imports.

- Baseline results (RTX 4090, window=600ms, hop=300ms, steps=20, cfg_rate=0.7, fp16):
  - Speed: mean window ≈ 0.48s => RTF ≈ 1.6 (not realtime at hop=300ms).
  - Offline:
    - `v5_to_fr_offline`: S-SIM≈0.952, WER≈0.616
    - `fr_to_v5_offline`: S-SIM≈0.745, WER≈0.364
  - Stream:
    - `v5_to_fr_stream`:  S-SIM≈0.944, WER≈0.640
    - `fr_to_v5_stream`:  S-SIM≈0.881, WER≈0.386
  - Artifacts: `runs/vc_quest/yingmusic_svc/user_pair/*`

- Tuned for realtime (RTX 4090, window=600ms, hop=300ms, steps=10, cfg_rate=0.7, fp16):
  - Speed: mean window ≈ 0.29s => RTF ≈ 0.97 (barely realtime).
  - Stream:
    - `v5_to_fr_stream`:  S-SIM≈0.943, WER≈0.494
    - `fr_to_v5_stream`:  S-SIM≈0.872, WER≈0.443
  - Note: `fr_to_v5_offline` shows non-trivial clipping (`clip_frac≈0.0018`), so listening review is required.
  - Artifacts: `runs/vc_quest/yingmusic_svc/user_pair_w600_h300_s10/*`

- Conclusion: with reduced diffusion steps, YingMusic-SVC can (barely) meet 300ms-hop realtime on RTX 4090 and has competitive objective metrics on our user pair; keep as a candidate pending listening + more robustness testing (it is still a singing-focused model).

### 9) SaMoye-SVC (zero-shot singing VC)
- Bead: `Amphion-ehh.11`
- Status: evaluated (reject)
- Hypothesis: similar to YingMusic-SVC; evaluate speech viability + streaming stability.
- Implementation: `evaluation/vc_quest/samoye_svc_convert.py` + `scripts/vc_quest/samoye_svc_*`
- Setup notes:
  - Checkpoints: `karl-wang/SaMoyeSVC` (`sovits_spk_1700h_0020.pt`, Whisper `large-v2.pt`, HuBERT-soft, speaker encoder).
  - Model is VITS/SVC-style (Whisper PPG + HuBERT units + F0 + speaker embedding).
  - Wrapper uses `pyworld` F0 (avoids missing `crepe` weights in upstream repo).

- Results (RTX 4090, sr=32kHz, window=600ms, hop=300ms, crossfade=10ms, fp16, VAD=rms):
  - Speed: mean window ≈ 0.073s => RTF ≈ 0.24 (very fast).
  - Offline:
    - `v5_to_fr_offline`: S-SIM≈0.836, WER≈0.799
    - `fr_to_v5_offline`: S-SIM≈0.933, WER≈0.671
  - Stream:
    - `v5_to_fr_stream`:  S-SIM≈0.834, WER≈0.927
    - `fr_to_v5_stream`:  S-SIM≈0.923, WER≈1.071
  - Artifacts: `runs/vc_quest/samoye_svc/user_pair/*`

- Conclusion: extremely fast with good speaker similarity on `fr_to_v5`, but content preservation is weak (high WER both offline and streaming) on our user pair; reject for real-time timbre VC.

### 10) FACodec (NaturalSpeech3, zero-shot VC via factorized codec)
- Bead: `Amphion-6rv`
- Status: evaluated (speed is excellent; quality mixed)
- Hypothesis: codec-based decomposition should support short windows (<=600ms) with stable timing and very low RTF, potentially suitable for live calls.
- Implementation: `evaluation/vc_quest/facodec_convert.py` + `scripts/vc_quest/facodec_run_user_pair_gpu.sh`
- Setup notes:
  - Uses HF checkpoints from `amphion/naturalspeech3_facodec` (`ns3_facodec_encoder_v2.bin`, `ns3_facodec_decoder_v2.bin`).
  - Output is 16kHz; wrapper enforces fixed window/hop timing via length normalization + crossfade-prefix + peak limiter.
- Results (RTX 4090, `use_residual=false`, window=600ms, hop=300ms, fade=10ms, VAD=WebRTC):
  - Speed: `mean_window_sec≈0.052s` => `RTF_mean≈0.175` (very fast; comfortably real-time).
  - Offline:
    - `v5_to_fr_offline`: S-SIM≈0.932, WER≈0.913, leak_p95≈-47.2dB, drop≈0.0000
    - `fr_to_v5_offline`: S-SIM≈0.956, WER≈0.420, leak_p95≈-44.8dB, drop≈0.0000
  - Stream:
    - `v5_to_fr_stream`:  S-SIM≈0.917, WER≈0.771, leak_p95≈-43.8dB, drop≈0.0000
    - `fr_to_v5_stream`:  S-SIM≈0.959, WER≈0.457, leak_p95≈-45.9dB, drop≈0.0000
- Artifacts: `runs/vc_quest/facodec/user_pair/*`
- Conclusion: extremely fast + stable, but content preservation is inconsistent on our French user pair (one direction has very high WER). Worth a quick follow-up sweep (e.g., `use_residual=true`) to see if intelligibility improves, but not yet a clear winner over FreeVC.

### 11) ChatterboxVC (ResembleAI/chatterbox)
- Bead: `Amphion-bck`
- Status: evaluated (candidate, but not yet real-time at our preferred hop size)
- Hypothesis: zero-shot VC backend with strong speaker similarity; may support buffered streaming via our wrapper.
- Implementation: `evaluation/vc_quest/chatterbox_convert.py` + `scripts/vc_quest/chatterbox_{setup,run}_user_pair_gpu.sh`
- Setup notes:
  - Uses conda env `chatterbox` + HF cache under `/root/.hf_home` (avoid Vast `/workspace` disk).
  - Imports `chatterbox.vc` without executing upstream `chatterbox/__init__.py` to avoid installing full TTS deps (but VC still requires `diffusers`, `transformers`, `conformer`, `omegaconf`).
- Results (RTX 4090, watermark enabled, WebRTC VAD, emit_align=center, streaming `w800/h400`):
  - `cfm_timesteps=10` (best quality so far, but *slightly* slower than real-time):
    - Speed: mean_window≈0.433s, p95_window≈0.437s ⇒ **RTF_p95≈1.09** @ hop=400ms
    - Stream:
      - `v5_to_fr_stream`: S-SIM≈0.958, WER≈0.681, leak_p95≈-37.2dB, drop≈0.0007
      - `fr_to_v5_stream`: S-SIM≈0.963, WER≈0.736, leak_p95≈-42.7dB, drop≈0.0000
  - `cfm_timesteps=8` (best real-time tradeoff so far):
    - Speed: **RTF_p95≈0.91** @ hop=400ms
    - Stream:
      - `v5_to_fr_stream`: S-SIM≈0.958, WER≈0.675, leak_p95≈-36.3dB, drop≈0.0012
      - `fr_to_v5_stream`: S-SIM≈0.960, WER≈0.714, leak_p95≈-43.2dB, drop≈0.0000
  - Offline:
    - `v5_to_fr_offline`: S-SIM≈0.911, WER≈0.604
    - `fr_to_v5_offline`: S-SIM≈0.858, WER≈0.482
  - Artifacts:
    - `runs/vc_quest/chatterbox/user_pair/*` (timesteps=10, simple run script)
    - `runs/vc_quest/chatterbox/user_pair_search_s8/*` (timesteps=8, realtime; recommended for comparison)
- Conclusion: speaker similarity + stability are excellent, but content preservation (WER) is still worse than FreeVC on this pair. Chatterbox can be made real-time at hop=400ms by reducing `cfm_timesteps` (best so far: `cfm_timesteps=8`), but it remains behind FreeVC in intelligibility under our Whisper-WER metric.

### 12) TinyVC (uthree/tinyvc)
- Bead: `Amphion-ehh.12`
- Status: evaluated (reject)
- Hypothesis: TinyVC’s SOLA streaming wrapper can provide **<=600ms effective context** with high stability (small IO blocks) and competitive timbre transfer, potentially even on CPU.
- Implementation: `evaluation/vc_quest/tinyvc_convert.py` + `scripts/vc_quest/tinyvc_*`
- Setup notes:
  - Pretrained weights are public on HF (`uthree/tinyvc`: `models/encoder.pt`, `models/decoder.pt`).
  - TinyVC operates at a fixed **24kHz** sample-rate.
  - Streaming uses TinyVC’s built-in SOLA (`module.infer.StreamInfer`) with a small IO block size (default `block_size=1920` samples ≈ 80ms).
- Results (macOS M3 Max, MPS, block_size=1920 ≈ 80ms, VAD=rms @ -55dB, peak_limit=0.99):
  - Speed: `p95_window_sec≈0.0225s` => `RTF_p95≈0.28` (real-time is easy).
  - Offline:
    - `v5_to_fr_offline`: S-SIM≈0.656, WER≈0.945
    - `fr_to_v5_offline`: S-SIM≈0.692, WER≈1.000
  - Stream:
    - `v5_to_fr_stream`: S-SIM≈0.610, WER≈0.862, leak_p95≈-40.8dB, drop≈0.058
    - `fr_to_v5_stream`: S-SIM≈0.591, WER≈1.068, leak_p95≈-24.2dB, drop≈0.059
- Conclusion: extremely fast, but **content preservation and speaker similarity are not competitive** on our user pair in both offline and stream modes. Likely reject for our live-call VC use case (unless we accept per-speaker fine-tuning/index building beyond a short reference clip).

### 13) FragmentVC (yistLin/FragmentVC)
- Bead: `Amphion-ehh.13`
- Status: in_progress
- Hypothesis: classic any-to-any VC baseline; may perform well offline, but streaming likely degrades because it is not designed to be causal.
- Implementation: `evaluation/vc_quest/fragmentvc_convert.py` + `scripts/vc_quest/fragmentvc_{setup,run}_user_pair_gpu.sh`
- Setup notes:
  - Uses TorchScript weights from FragmentVC GitHub Release `v1.0` (`fragmentvc.pt`, `vocoder.pt`).
  - Uses HF `facebook/wav2vec2-base` for content features (avoids building legacy fairseq).
- Next: run RTX 4090 offline + streaming sim and record WER/S-SIM/artifacts + realtime factor.

### 14) HiFi-VC (tinkoff-ai/hifi_vc)
- Status: planned
- Hypothesis: high-quality any-to-any VC offline; streaming may be slow/unstable due to ASR/F0 dependencies and lack of native chunking.
- Next: integrate notebook inference into a CLI wrapper and evaluate offline + streaming sim.

- Next:
  - Have user listen to FreeVC v2 artifacts (`runs/vc_quest/freevc/user_pair_search_webrtc_center/*`) and Seed-VC (`runs/vc_quest/seedvc/user_pair/*`) to sanity-check objective metrics vs perception.
  - If FreeVC v2 is acceptable: implement a minimal real-time FreeVC runner (mic->buffer->GPU inference->playback) using the selected window/hop.

## What we record for each candidate
- **Streaming config:** sample rate, chunk/window, hop, crossfade/OLA, VAD settings, any lookahead.
- **Speed:** mean/p95 chunk processing time, estimated RTF on RTX 4090.
- **Objective quality:** Whisper WER (aligned), speaker similarity (WavLM), artifact metrics (silence leakage, dropout, clipping).
- **Artifacts for listening:** offline + live-sim wavs for the same source/reference pair.

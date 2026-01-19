# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from evaluation.vc_quest.streaming_utils import (
    apply_peak_limiter,
    build_rms_mask,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
    rms_db,
)
from models.vc.live_io import (
    AudioRingBuffer,
    OutputRingBuffer,
    normalize_len_end,
    parse_device_arg,
    print_device_help,
    resample_audio,
)


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    import sys

    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _set_determinism(seed: int) -> None:
    import torch

    seed = int(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _trim_ref_librosa(wav_16k: np.ndarray, *, top_db: float) -> np.ndarray:
    wav_16k = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
    if len(wav_16k) == 0:
        return wav_16k
    if float(top_db) <= 0:
        return wav_16k
    import librosa

    trimmed, _ = librosa.effects.trim(wav_16k, top_db=float(top_db))
    return np.asarray(trimmed, dtype=np.float32).reshape(-1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FreeVC live VC (single-process, local inference).")
    parser.add_argument("--freevc_dir", type=str, default="~/deps/FreeVC", help="Path to FreeVC repo checkout.")
    parser.add_argument(
        "--variant",
        type=str,
        default="freevc",
        choices=["freevc", "freevc-s", "freevc-24"],
        help="Which FreeVC checkpoint/config to use.",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, mps, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")
    parser.add_argument("--wavlm_model", type=str, default="microsoft/wavlm-large")

    parser.add_argument("--ref", type=str, default=None, help="Reference wav (required unless --passthrough).")
    parser.add_argument("--ref_max_sec", type=float, default=10.0, help="Trim reference audio duration for speed.")
    parser.add_argument("--ref_trim_db", type=float, default=20.0, help="librosa.effects.trim top_db for reference.")

    parser.add_argument("--io_sample_rate", type=int, default=48000, help="Audio device sample rate.")
    parser.add_argument("--window_ms", type=int, default=800)
    parser.add_argument("--hop_ms", type=int, default=400)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--emit_align", type=str, default="end", choices=["start", "center", "end"])

    parser.add_argument("--vad_mode", type=str, default="rms", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)

    parser.add_argument("--gain_mode", type=str, default="off", choices=["off", "match_src_rms"])
    parser.add_argument("--gain_target_delta_db", type=float, default=10.0)
    parser.add_argument("--gain_max_boost_db", type=float, default=18.0)
    parser.add_argument("--gain_smoothing", type=float, default=0.0)

    parser.add_argument("--mask_mode", type=str, default="off", choices=["off", "rms"])
    parser.add_argument("--mask_db", type=float, default=-50.0)
    parser.add_argument("--mask_frame_ms", type=float, default=10.0)
    parser.add_argument("--mask_smooth_ms", type=float, default=10.0)

    parser.add_argument("--peak_limit", type=float, default=0.99)
    parser.add_argument("--input_device", type=str, default=None)
    parser.add_argument("--output_device", type=str, default=None)
    parser.add_argument("--block_ms", type=int, default=20)
    parser.add_argument("--list_devices", action="store_true")
    parser.add_argument("--passthrough", action="store_true", help="Debug: bypass FreeVC and play back mic audio.")
    args = parser.parse_args(argv)

    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: sounddevice. Install with `pip install sounddevice`.") from e

    if args.list_devices:
        print_device_help(sd)
        return 0

    io_sr = int(args.io_sample_rate)
    block_samples_io = int(round(float(args.block_ms) / 1000.0 * float(io_sr)))
    if block_samples_io <= 0:
        block_samples_io = 256

    input_device = parse_device_arg(args.input_device)
    output_device = parse_device_arg(args.output_device)

    in_sr = 16000

    window_in = int(round(float(args.window_ms) / 1000.0 * float(in_sr)))
    hop_in = int(round(float(args.hop_ms) / 1000.0 * float(in_sr)))
    window_in = max(1, window_in)
    hop_in = max(1, hop_in)
    if hop_in > window_in:
        raise ValueError("hop_ms must be <= window_ms")

    hop_seconds = hop_in / float(in_sr)
    hop_samples_io = int(round(hop_seconds * float(io_sr)))
    hop_samples_io = max(1, hop_samples_io)

    if args.passthrough:
        out_sr = int(in_sr)
        window_out = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
        hop_out = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
        fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))
        window_out = max(1, window_out)
        hop_out = max(1, hop_out)
        out_model = None
        g_tgt = None
        mel_tgt = None
        net_g = None
        hps = None
        cmodel = None
    else:
        if not args.ref:
            raise ValueError("--ref is required unless --passthrough")

        import logging
        import sys

        # Prevent FreeVC's `utils.py` from setting global DEBUG logging (which makes numba extremely noisy).
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
        logging.getLogger("numba").setLevel(logging.WARNING)

        import torch

        device = (
            torch.device(args.device)
            if args.device
            else torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        )

        freevc_dir = os.path.abspath(str(Path(args.freevc_dir).expanduser()))
        if not os.path.isdir(freevc_dir):
            raise FileNotFoundError(f"freevc_dir not found: {freevc_dir}")

        _add_sys_path_first(freevc_dir)

        import utils as freevc_utils  # type: ignore[import-not-found]
        from mel_processing import mel_spectrogram_torch  # type: ignore[import-not-found]
        from models import SynthesizerTrn  # type: ignore[import-not-found]
        from speaker_encoder.voice_encoder import SpeakerEncoder  # type: ignore[import-not-found]
        from transformers import WavLMModel  # type: ignore[import-not-found]

        variant = str(args.variant)
        ckpt_name = {
            "freevc": "freevc.pth",
            "freevc-s": "freevc-s.pth",
            "freevc-24": "freevc-24.pth",
        }[variant]
        cfg_name = {
            "freevc": "freevc.json",
            "freevc-s": "freevc-s.json",
            "freevc-24": "freevc-24.json",
        }[variant]

        ckpt_path = Path(freevc_dir) / "checkpoints" / ckpt_name
        cfg_path = Path(freevc_dir) / "configs" / cfg_name
        spk_ckpt = Path(freevc_dir) / "speaker_encoder" / "ckpt" / "pretrained_bak_5805000.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing FreeVC checkpoint: {ckpt_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing FreeVC config: {cfg_path}")
        if not spk_ckpt.exists():
            raise FileNotFoundError(f"Missing speaker encoder ckpt: {spk_ckpt}")

        print(f"[freevc_live_local] Loading {variant} from {cfg_path} + {ckpt_path}", flush=True)
        hps = freevc_utils.get_hparams_from_file(str(cfg_path))

        net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        ).to(device)
        net_g.eval()
        freevc_utils.load_checkpoint(str(ckpt_path), net_g, None)

        smodel = SpeakerEncoder(str(spk_ckpt))
        cmodel = WavLMModel.from_pretrained(str(args.wavlm_model)).to(device)
        cmodel.eval()

        # Infer output sample rate (FreeVC uses WavLM at 16kHz).
        wavlm_hop = 320  # WavLM conv feature extractor hop @16kHz (20ms).
        upsample_rates = list(getattr(hps.model, "upsample_rates", []))
        if not upsample_rates:
            raise ValueError("Missing hps.model.upsample_rates; cannot infer output sample rate.")
        prod_upsample = int(np.prod(np.asarray(upsample_rates, dtype=np.int64)))
        out_sr = int(round(float(in_sr) / float(wavlm_hop) * float(prod_upsample)))
        if out_sr <= 0:
            raise ValueError(f"Invalid inferred output sample rate: {out_sr}")

        window_out = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
        hop_out = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
        fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))
        window_out = max(1, window_out)
        hop_out = max(1, hop_out)

        ref_wav, ref_sr = _load_mono(str(args.ref))
        if float(args.ref_max_sec) > 0:
            ref_wav = ref_wav[: int(round(float(args.ref_max_sec) * float(ref_sr)))]
        ref_16k = resample_audio(ref_wav, ref_sr, in_sr)
        ref_16k = _trim_ref_librosa(ref_16k, top_db=float(args.ref_trim_db))

        # Cache speaker conditioning.
        if bool(getattr(hps.model, "use_spk", False)):
            g_np = smodel.embed_utterance(ref_16k)
            g_tgt = torch.from_numpy(np.asarray(g_np, dtype=np.float32)).unsqueeze(0).to(device)
            mel_tgt = None
        else:
            ref_t = torch.from_numpy(ref_16k).unsqueeze(0).to(device)
            mel_tgt = mel_spectrogram_torch(
                ref_t,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            g_tgt = None

        out_model = {
            "device": device,
            "mel_spectrogram_torch": mel_spectrogram_torch,
        }

    window_out = int(window_out)  # type: ignore[has-type]
    hop_out = int(hop_out)  # type: ignore[has-type]
    fade_out = int(fade_out)  # type: ignore[has-type]
    out_sr = int(out_sr)  # type: ignore[has-type]

    if args.emit_align == "start":
        emit_start_out = 0
        emit_start_in = 0
    elif args.emit_align == "center":
        emit_start_out = max(0, (window_out - hop_out) // 2)
        emit_start_in = max(0, (window_in - hop_in) // 2)
    elif args.emit_align == "end":
        emit_start_out = max(0, window_out - hop_out)
        emit_start_in = max(0, window_in - hop_in)
    else:
        raise ValueError(f"Unknown emit_align: {args.emit_align}")

    input_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
    out_buf = OutputRingBuffer(capacity=int(io_sr * 10))
    stop = threading.Event()

    def in_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            pass
        x = indata[:, 0].copy().astype(np.float32, copy=False)
        try:
            input_q.put_nowait(x)
        except queue.Full:
            try:
                _ = input_q.get_nowait()
            except queue.Empty:
                pass
            input_q.put_nowait(x)

    def out_callback(outdata, frames, time_info, status):  # noqa: ANN001
        if status:
            pass
        y = out_buf.read(frames)
        outdata[:] = y.reshape(-1, 1)

    def worker() -> None:
        torch = None
        if not args.passthrough:
            import torch as _torch

            torch = _torch

        ring = AudioRingBuffer(window_in)
        prev_tail: Optional[np.ndarray] = None
        buf_io = np.zeros(0, dtype=np.float32)

        timings: list[float] = []
        gain_db_state = 0.0
        hop_ms_eff = float(args.hop_ms)
        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(hop_ms_eff, 1e-6)))
        hangover_left = 0
        window_count = 0

        while not stop.is_set():
            try:
                x_block = input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buf_io = np.concatenate([buf_io, x_block])
            while len(buf_io) >= hop_samples_io and not stop.is_set():
                hop_io = buf_io[:hop_samples_io]
                buf_io = buf_io[hop_samples_io:]

                if args.passthrough:
                    out_io = apply_peak_limiter(hop_io, peak_limit=float(args.peak_limit))
                    out_buf.write(out_io.astype(np.float32, copy=False))
                    window_count += 1
                    continue

                hop_16k = resample_audio(hop_io, io_sr, in_sr)
                hop_16k = normalize_len_end(hop_16k, hop_in)
                ring.write(hop_16k)

                if ring.size < window_in:
                    if fade_out > 0:
                        prev_tail = np.zeros(fade_out, dtype=np.float32)
                    out_buf.write(np.zeros(hop_samples_io, dtype=np.float32))
                    continue

                window = ring.read_last(window_in)
                vad_segment = window[emit_start_in : emit_start_in + hop_in]

                silent_rms = bool(
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        vad_segment,
                        sample_rate=in_sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )

                vad_mode = str(args.vad_mode)
                if vad_mode == "off":
                    voiced = True
                elif vad_mode == "rms":
                    voiced = not silent_rms
                elif vad_mode == "webrtc":
                    webrtc_voiced = is_voiced_webrtcvad(
                        vad_segment,
                        sample_rate=in_sr,
                        frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                        aggressiveness=int(args.vad_webrtc_aggressiveness),
                        min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                    )
                    voiced = bool(webrtc_voiced) and (not silent_rms)
                else:
                    raise ValueError(f"Unknown vad_mode: {vad_mode}")

                if not voiced and hangover_left > 0 and (not silent_rms):
                    voiced = True
                    hangover_left -= 1
                elif voiced:
                    hangover_left = hangover_hops

                if not voiced:
                    out_hop = np.zeros(hop_out, dtype=np.float32)
                else:
                    t0 = time.time()
                    device = out_model["device"]
                    src_t = torch.from_numpy(window).unsqueeze(0).to(device)  # type: ignore[union-attr]
                    c = cmodel(src_t).last_hidden_state.transpose(1, 2).to(device)  # type: ignore[union-attr]
                    _set_determinism(int(args.seed) + int(window_count))
                    if g_tgt is not None:
                        audio = net_g.infer(c, g=g_tgt)  # type: ignore[union-attr]
                    else:
                        audio = net_g.infer(c, mel=mel_tgt)  # type: ignore[union-attr,arg-type]
                    audio_np = audio[0][0].detach().cpu().float().numpy()
                    timings.append(time.time() - t0)
                    if len(timings) > 200:
                        timings = timings[-200:]

                    out_window = np.asarray(audio_np, dtype=np.float32).reshape(-1)
                    out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))
                    out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

                if voiced and str(args.gain_mode) == "match_src_rms":
                    alpha = float(np.clip(float(args.gain_smoothing), 0.0, 0.999))
                    src_db = rms_db(vad_segment, eps=1e-9)
                    out_db = rms_db(out_hop, eps=1e-9)
                    desired_boost_db = (src_db - out_db) - float(args.gain_target_delta_db)
                    desired_boost_db = float(np.clip(desired_boost_db, 0.0, float(args.gain_max_boost_db)))
                    gain_db_state = alpha * gain_db_state + (1.0 - alpha) * desired_boost_db
                    gain = float(10.0 ** (gain_db_state / 20.0))
                    out_hop = (out_hop * gain).astype(np.float32, copy=False)
                elif not voiced:
                    gain_db_state *= float(np.clip(float(args.gain_smoothing), 0.0, 0.999))

                if str(args.mask_mode) == "rms":
                    mask = build_rms_mask(
                        vad_segment,
                        in_sample_rate=in_sr,
                        out_sample_rate=out_sr,
                        out_len=hop_out,
                        frame_ms=float(args.mask_frame_ms),
                        threshold_db=float(args.mask_db),
                        smooth_ms=float(args.mask_smooth_ms),
                    )
                    out_hop = (out_hop * mask).astype(np.float32, copy=False)

                out_hop = crossfade_prefix_inplace(out_hop, prev_tail, fade_out)
                out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
                if fade_out > 0:
                    prev_tail = out_hop[-fade_out:].astype(np.float32, copy=True)

                out_io = resample_audio(out_hop, out_sr, io_sr)
                out_io = normalize_len_end(out_io, hop_samples_io)
                out_io = apply_peak_limiter(out_io, peak_limit=float(args.peak_limit))
                out_buf.write(out_io)
                window_count += 1

                if window_count % max(1, int(2.5 / max(hop_seconds, 1e-6))) == 0 and timings:
                    mean_sec = float(np.mean(np.asarray(timings, dtype=np.float64)))
                    rtf = mean_sec / max(hop_seconds, 1e-9)
                    print(
                        f"[freevc_live_local] windows={window_count} mean_win_s={mean_sec:.3f} rtf={rtf:.2f} "
                        f"out_buf_s={out_buf.size()/io_sr:.2f} underflows={out_buf.underflows} "
                        f"overflows={out_buf.overflows} in_q={input_q.qsize()}",
                        flush=True,
                    )
                    timings = timings[-50:]

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    stream_in = sd.InputStream(
        samplerate=io_sr,
        channels=1,
        dtype="float32",
        blocksize=block_samples_io,
        device=input_device,
        callback=in_callback,
    )
    stream_out = sd.OutputStream(
        samplerate=io_sr,
        channels=1,
        dtype="float32",
        blocksize=block_samples_io,
        device=output_device,
        callback=out_callback,
    )

    print(
        f"[freevc_live_local] io_sr={io_sr} in_sr={in_sr} out_sr={out_sr} window={args.window_ms}ms hop={args.hop_ms}ms "
        f"emit_align={args.emit_align} passthrough={bool(args.passthrough)}",
        flush=True,
    )

    with stream_in, stream_out:
        try:
            while thread.is_alive():
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            thread.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

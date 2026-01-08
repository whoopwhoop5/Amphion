# Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement

[![arXiv](https://img.shields.io/badge/OpenReview-Paper-COLOR.svg)](https://openreview.net/pdf?id=anQDiQZhDP)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/Vevo)
[![WebPage](https://img.shields.io/badge/WebPage-Demo-red)](https://versavoice.github.io/)

We present our reproduction of [Vevo](https://openreview.net/pdf?id=anQDiQZhDP), a versatile zero-shot voice imitation framework with controllable timbre and style. We invite you to explore the [audio samples](https://versavoice.github.io/) to experience Vevo's capabilities firsthand.

<br>
<div align="center">
<img src="../../../imgs/vc/vevo.png" width="100%">
</div>
<br>

We have included the following pre-trained Vevo models at Amphion:

- **Vevo-Timbre**: It can conduct *style-preserved* voice conversion.
- **Vevo-Style**: It can conduct style conversion, such as *accent conversion* and *emotion conversion*.
- **Vevo-Voice**: It can conduct *style-converted* voice conversion.
- **Vevo-TTS**: It can conduct *style and timbre controllable* TTS.

Besides, we also release the **content tokenizer** and **content-style tokenizer** proposed by Vevo. Notably, all these pre-trained models are trained on [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset), containing 101k hours of speech data among six languages (English, Chinese, German, French, Japanese, and Korean).

## Quickstart (Inference Only)

To run this model, you need to follow the steps below:

1. Clone the repository and install the environment.
2. Run the inference script.

### Clone and Environment Setup

#### 1. Clone the repository

```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion
```

#### 2. Install the environment

Before start installing, making sure you are under the `Amphion` directory. If not, use `cd` to enter.

Since we use `phonemizer` to convert text to phoneme, you need to install `espeak-ng` first. More details can be found [here](https://bootphon.github.io/phonemizer/install.html). Choose the correct installation command according to your operating system:

```bash
# For Debian-like distribution (e.g. Ubuntu, Mint, etc.)
sudo apt-get install espeak-ng
# For RedHat-like distribution (e.g. CentOS, Fedora, etc.) 
sudo yum install espeak-ng

# For Windows
# Please visit https://github.com/espeak-ng/espeak-ng/releases to download .msi installer
```

Now, we are going to install the environment. It is recommended to use conda to configure:

```bash
conda create -n vevo python=3.10
conda activate vevo

pip install -r models/vc/vevo/requirements.txt
```

### Inference Script

```bash
# Vevo-Timbre
python -m models.vc.vevo.infer_vevotimbre

# Vevo-Style
python -m models.vc.vevo.infer_vevostyle

# Vevo-Voice
python -m models.vc.vevo.infer_vevovoice

# Vevo-TTS
python -m models.vc.vevo.infer_vevotts
```

Running this will automatically download the pretrained model from HuggingFace and start the inference process. The result audio is by default saved in `models/vc/vevo/wav/output*.wav`, you can change this in the scripts  `models/vc/vevo/infer_vevo*.py`

## Offline CLI (WAV -> WAV)

This repo also includes a small, reusable CLI wrapper around the official Vevo inference:

```bash
# Vevo-Timbre (style-preserved voice conversion)
python -m models.vc.vevo.convert \
  --kind vevotimbre \
  --src assets/vevo_live/playlist/source_clip_00.wav \
  --ref assets/vevo_live/target_ref.wav \
  --out runs/vevo_live/offline_vevotimbre.wav \
  --flow_matching_steps 16

# Vevo-Voice (style-converted voice conversion)
python -m models.vc.vevo.convert \
  --kind vevovoice \
  --src assets/vevo_live/playlist/source_clip_00.wav \
  --ref assets/vevo_live/target_ref.wav \
  --out runs/vevo_live/offline_vevovoice.wav \
  --flow_matching_steps 16
```

## Live Buffered Conversion (GPU server + local client)

This is a buffered (~1s) streaming wrapper around Vevo that enforces stable chunk timing at the wrapper level.

Install optional live dependencies:

```bash
pip install -r models/vc/vevo/requirements_live.txt
```

On the GPU host (runs the model):

```bash
python -m models.vc.vevo.live_server --host 0.0.0.0 --port 8080
```

If the server runs on a remote GPU machine (e.g. Vast), forward the port to your local machine:

```bash
# Example (adjust host/port to your setup)
ssh -L 8080:localhost:8080 vastai-gpu-1
```

On your local machine (mic -> server -> playback):

```bash
python -m models.vc.vevo.live_client \
  --server ws://localhost:8080 \
  --ref assets/vevo_live/target_ref.wav \
  --kind vevotimbre \
  --window_ms 2000 --hop_ms 1000 --fade_ms 10 \
  --flow_matching_steps 8
```

Or load a saved autotune config:

```bash
python -m models.vc.vevo.live_client \
  --server ws://localhost:8080 \
  --ref assets/vevo_live/target_ref.wav \
  --config_json evaluation/vevo_live/best_configs/vevotimbre.json
```

## Live Buffered Conversion (Single-Process / Local Inference)

If you want to avoid network/server RTT (run fully on the local machine), use the single-process runner:

```bash
python -m models.vc.vevo.live_local \
  --ref assets/vevo_live/target_ref.wav \
  --kind vevotimbre \
  --window_ms 2000 --hop_ms 1000 --fade_ms 10 \
  --flow_matching_steps 8
```

Or load a saved config (recommended for macOS MPS):

```bash
python -m models.vc.vevo.live_local \
  --ref assets/vevo_live/target_ref.wav \
  --config_json evaluation/vevo_live/best_configs/vevotimbre.macos_steps6_2000w_1000h.json
```

Notes:
- For live usage, a short (≈5–10s) clean reference clip is recommended; `live_local` trims the reference by default (`--ref_max_sec 10`).
- If you hear noise, first verify your audio I/O pipeline with passthrough mode. If passthrough sounds fine but Vevo sounds like noise, increase the streaming context (Vevo often needs ≥1.5–2.0s windows):

```bash
python -m models.vc.vevo.live_local \
  --passthrough \
  --window_ms 2000 --hop_ms 1000 --fade_ms 10
```

To test deterministically without a microphone, simulate “mic input” from a wav file:

```bash
python -m models.vc.vevo.live_local \
  --ref assets/vevo_live/target_ref.wav \
  --config_json evaluation/vevo_live/best_configs/vevotimbre.macos_steps6_2000w_1000h.json \
  --src_wav assets/vevo_live/playlist/source_clip_00.wav \
  --out_wav runs/vevo_live/live_local_sim.wav \
  --sim_realtime
```

To select audio devices:

```bash
python -m models.vc.vevo.live_client --list_devices
python -m models.vc.vevo.live_local --list_devices
python -m models.vc.vevo.live_local ... --input_device "Your Mic Name" --output_device "Your Output Name"
```

## Deterministic Autotune + Regression

```bash
# Parameter search (writes `runs/vevo_live/search/best/*`)
python -m evaluation.vevo_live.search \
  --kind vevotimbre \
  --reference_wav assets/vevo_live/target_ref.wav \
  --playlist_dir assets/vevo_live/playlist \
  --eval_seconds 4

# Regression (uses a committed baseline config; adjust thresholds after establishing your own baseline)
python -m evaluation.vevo_live.regress \
  --config_json evaluation/vevo_live/best_configs/vevotimbre.json \
  --reference_wav assets/vevo_live/target_ref.wav \
  --playlist_dir assets/vevo_live/playlist
```
## Training Recipe

For advanced users, we provide the following training recipe:

### Emilia data preparation

1. Please download the dataset following the official instructions provided by [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset).

2. Due to Emilia's substantial storage requirements, data loading logic may vary slightly depending on storage configuration. We provide a reference implementation for local disk loading [in this file](../../base/emilia_dataset.py). After downloading the Emilia dataset, please adapt the data loading logic accordingly. In most cases, only modifying the paths specified in [Lines 36-37](../../base/emilia_dataset.py#L36) should be sufficient: 

   ```python
   MNT_PATH = "[Please fill out your emilia data root path]"
   CACHE_PATH = "[Please fill out your emilia cache path]"
   ```

### Launch Training

Train the Vevo tokenizers, the auto-regressive model, and the flow-matching model, respectively:

> **Note**: You need to run the following commands under the `Amphion` root path:
> ```
> git clone https://github.com/open-mmlab/Amphion.git
> cd Amphion
> ```

#### Tokenizers

Run the following script:

```bash
# Content Tokenizer (Vocab = 32)
sh egs/codec/vevo/fvq32.sh

# Content-Style Tokenizer (Vocab = 8192)
sh egs/codec/vevo/fvq8192.sh
```

If you want to try different vocabulary sizes, just specify it in the `egs/codec/vevo/fvq*.json`:

```json
{
    ...
     "model": {
        "repcodec": {
            "codebook_size": 8192, // Specify the vocabulary size here.
            ...
        },
        ...
    },
    ...
}
```

#### Auto-regressive Transformer

Specify the content tokenizer and content-style tokenizer paths in the `egs/vc/AutoregressiveTransformer/ar_conversion.json`:

```json
{
    ...
    "model": {
        "input_repcodec": {
            "codebook_size": 32,
            "hidden_size": 1024, // Representations Dim
            "codebook_dim": 8,
            "vocos_dim": 384,
            "vocos_intermediate_dim": 2048,
            "vocos_num_layers": 12,
            "pretrained_path": "[Please fill out your pretrained model path]/model.safetensors" // The pre-trained content tokenizer
        },
        "output_repcodec": {
            "codebook_size": 8192, // VQ Codebook Size
            "hidden_size": 1024, // Representations Dim
            "codebook_dim": 8,
            "vocos_dim": 384,
            "vocos_intermediate_dim": 2048,
            "vocos_num_layers": 12,
            "pretrained_path": "[Please fill out your pretrained model path]/model.safetensors" // The pre-trained content-style tokenizer
        }
    },
    ...
}
```

Run the following script:

```bash
sh egs/vc/AutoregressiveTransformer/ar_conversion.sh
```

Similarly, you can run the following script for Vevo-TTS training:

```bash
sh egs/vc/AutoregressiveTransformer/ar_synthesis.sh
```

#### Flow-matching Transformer

Specify the pre-trained content-style tokenizer path in the `egs/vc/FlowMatchingTransformer/fm_contentstyle.json`:

```json
{
    ...
    "model": {
        "repcodec": {
            "codebook_size": 8192, // VQ Codebook Size
            "hidden_size": 1024, // Representations Dim
            "codebook_dim": 8,
            "vocos_dim": 384,
            "vocos_intermediate_dim": 2048,
            "vocos_num_layers": 12,
            "pretrained_path": "[Please fill out your pretrained model path]/model.safetensors" // The pre-trained content-style tokenizer
        }
    },
    ...
}
```

Run the following script:

```bash
sh egs/vc/FlowMatchingTransformer/fm_contentstyle.sh
```

#### Vocoder
We provide a unified vocos-based vocoder training recipe for both speech and singing voice. See our [Vevo1.5](../../svc/vevosing/README.md#vocoder) framework for the details.

## Citations

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{vevo,
  author       = {Xueyao Zhang and Xiaohui Zhang and Kainan Peng and Zhenyu Tang and Vimal Manohar and Yingru Liu and Jeff Hwang and Dangna Li and Yuhao Wang and Julian Chan and Yuan Huang and Zhizheng Wu and Mingbo Ma},
  title        = {Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement},
  booktitle    = {{ICLR}},
  publisher    = {OpenReview.net},
  year         = {2025}
}
```

If you use the Vevo pre-trained models or training recipe of Amphion, please also cite:

```bibtex
@article{amphion2,
  title        = {Overview of the Amphion Toolkit (v0.2)},
  author       = {Jiaqi Li and Xueyao Zhang and Yuancheng Wang and Haorui He and Chaoren Wang and Li Wang and Huan Liao and Junyi Ao and Zeyu Xie and Yiqiao Huang and Junan Zhang and Zhizheng Wu},
  year         = {2025},
  journal      = {arXiv preprint arXiv:2501.15442},
}

@inproceedings{amphion,
    author={Xueyao Zhang and Liumeng Xue and Yicheng Gu and Yuancheng Wang and Jiaqi Li and Haorui He and Chaoren Wang and Ting Song and Xi Chen and Zihao Fang and Haopeng Chen and Junan Zhang and Tze Ying Tang and Lexiao Zou and Mingxuan Wang and Jun Han and Kai Chen and Haizhou Li and Zhizheng Wu},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```

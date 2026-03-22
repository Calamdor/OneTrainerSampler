# OneTrainer Sampler

A standalone image sampler GUI for [OneTrainer](https://github.com/Nerogar/OneTrainer) models, designed to run inference without launching the full training application.

## Features

- **Chroma support** — full Chroma (FLUX-based) inference with NF4, BF16, FP16, W8A8, and GGUF weight dtypes
- **LoRA support** — load, weight, enable/disable, and hot-swap multiple LoRAs; accepts OT internal, ai-toolkit, kohya, and musubi-tuner key formats
- **Job queue** — queue multiple jobs with different prompts/seeds/settings; each job captures its settings at enqueue time
- **Two-column layout** — controls on the left, generated image preview and run details on the right
- **Schedulers** — Euler and Heun with configurable sigma shift
- **Attention backends** — Auto, Flash Attention, SageAttention, PyTorch SDPA
- **SVDQuant** — optional low-rank correction for improved quantized quality
- **Layer offload** — stream transformer blocks CPU↔GPU for low-VRAM inference
- **torch.compile** — optional block-level compilation for BF16/FP16 speed
- **Text encoder cache** — disk-cache T5 embeddings to skip re-encoding identical prompts
- **Quantization cache** — persist computed quantization to disk for fast subsequent loads
- **PNG metadata** — generation parameters embedded in every output image

## Requirements

- [OneTrainer](https://github.com/Nerogar/OneTrainer) installed in an adjacent directory (`../OneTrainer/`) with its virtual environment set up
- Python packages from the OneTrainer venv (PyTorch, diffusers, transformers, Pillow, etc.)

## Setup

1. Clone this repository adjacent to your OneTrainer installation:
   ```
   OneTrainer/
   OneTrainerSampler/   ← this repo
   ```

2. Launch via the batch file (sets up MSVC for torch.compile and pins cache directories):
   ```
   start_onetrainer_sampler.bat
   ```

   Or directly:
   ```
   ..\OneTrainer\venv\Scripts\python.exe onetrainer_sampler_gui.py
   ```

## Usage

1. Set the **Base model** path (HuggingFace diffusers format, local or Hub ID)
2. Choose **Transformer dtype** (NF4 recommended for most GPUs)
3. Click **Load Model**
4. Enter a prompt and adjust generation parameters
5. Click **+ Add to Queue** — the job runs automatically once the model is loaded

Generated images are saved to the configured output directory with the filename format `YYYY-MM-DD_HH-MM-SS-sample-{seed}.png`.

## Architecture

```
OneTrainerSampler/
├── onetrainer_sampler_gui.py      # Entry point
├── start_onetrainer_sampler.bat   # Windows launcher (sets MSVC + cache dirs)
├── chroma/
│   ├── backend.py                 # Chroma model loading and inference
│   ├── gui.py                     # Chroma-specific UI panels
│   └── lora_keys.py               # LoRA key translation for Chroma
└── sampler_core/
    ├── gui/
    │   ├── app_base.py            # Abstract base GUI (queue, LoRA panel, preview)
    │   ├── launcher.py            # Model selector launcher
    │   ├── theme.py               # Dark theme
    │   └── tooltip.py             # Tooltip widget
    ├── backend/
    │   └── base.py                # Abstract sampler backend
    ├── lora/
    │   ├── hooks.py               # LoRA forward hook injection
    │   └── key_util.py            # Key format utilities
    └── util/
        ├── config.py              # JSON config persistence
        ├── dtype_maps.py          # Weight dtype mappings
        ├── gguf_util.py           # GGUF file utilities
        ├── ot_bridge.py           # OneTrainer workspace integration
        ├── png_meta.py            # PNG metadata embedding
        └── resolution.py         # Resolution / aspect ratio calculation
```

## License

GNU Affero General Public License v3 — see [LICENSE](LICENSE) for details.

This project depends on [OneTrainer](https://github.com/Nerogar/OneTrainer), also licensed under AGPLv3.

# OneTrainer Sampler

A standalone image/video sampler GUI for [OneTrainer](https://github.com/Nerogar/OneTrainer) models, designed to run inference without launching the full training application.

## Features

### Models
- **Chroma** — full Chroma (FLUX-based) inference
- **Wan 2.2 T2V-A14B** — text-to-video inference with dual-expert transformer

### Weight dtypes
- NF4, INT8, FP8, W8A8F, W8A8I, BF16, FP16, FP32
- **GGUF** — load `.gguf` quantized transformer files
- **GGUF_A8I / GGUF_A8F** — GGUF with int8/fp8 activation requantization; `torch.compile`-friendly (requires RTX 3000+ / RTX 4000+)

### Workflow
- **Job queue** — queue multiple jobs; each captures its settings at enqueue time
- **Prompt Library** — persistent history of all generated prompt pairs, auto-saved after every run; grouped by date, searchable, with Use/Delete buttons
- **Seed controls** — fixed seed, Rnd (random every run), Gen (one-shot random), or reset to 42
- **PNG/MP4 metadata** — generation parameters embedded in every output file

### File import
- **Load File / Drag-and-drop** — drop a PNG, MP4, or JSON onto the preview panel to extract and reuse prompt metadata
- Supports our own sampler outputs, ComfyUI PNG workflows, ComfyUI JSON exports, and ComfyUI MP4 outputs
- **Use Positive / Use Negative / Use Both** buttons copy extracted prompts directly into the prompt fields

### Performance
- **torch.compile** — optional block-level compilation for BF16/FP16 and all GGUF variants
- **Layer offload** — stream transformer blocks CPU↔GPU for low-VRAM inference
- **Text encoder cache** — disk-cache T5 embeddings to skip re-encoding identical prompts
- **Quantization cache** — persist computed quantization to disk for fast subsequent loads
- **SVDQuant** — optional low-rank correction for improved quantized quality (Chroma)

### Other
- **LoRA support** — load, weight, enable/disable, and hot-swap multiple LoRAs; accepts OT internal, ai-toolkit, kohya, and musubi-tuner key formats
- **Schedulers** — Euler and Heun with configurable sigma shift
- **Attention backends** — Auto, Flash Attention, SageAttention, PyTorch SDPA

## Requirements

- [OneTrainer](https://github.com/Nerogar/OneTrainer) installed in an adjacent directory (`../OneTrainer/`) with its virtual environment set up
- Python packages from the OneTrainer venv (PyTorch, diffusers, transformers, Pillow, etc.)

### Optional
- `tkinterdnd2` — enables drag-and-drop files onto the preview panel. The sampler will offer to install it automatically on first launch, or install manually:
  ```
  pip install tkinterdnd2
  ```

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

1. Select the **model type** from the dropdown at the top
2. Set the **Base model** path (HuggingFace diffusers format, local or Hub ID)
3. Choose **Transformer dtype** (NF4 recommended for most GPUs)
4. Click **Load Model**
5. Enter a prompt and adjust generation parameters
6. Click **+ Add to Queue** — the job runs automatically once the model is loaded

Generated images are saved to the configured output directory with the filename `YYYY-MM-DD_HH-MM-SS-sample-{seed}.png` (or `.mp4` for video).

## Architecture

```
OneTrainerSampler/
├── onetrainer_sampler_gui.py      # Entry point + optional dependency installer
├── requirements_sampler.txt       # Optional extra packages (tkinterdnd2)
├── start_onetrainer_sampler.bat   # Windows launcher (sets MSVC + cache dirs)
├── chroma/
│   ├── backend.py                 # Chroma model loading and inference
│   ├── gui.py                     # Chroma-specific UI panels
│   └── lora_keys.py               # LoRA key translation for Chroma
├── wan/
│   ├── backend.py                 # Wan 2.2 model loading and inference
│   ├── gui.py                     # Wan-specific UI panels
│   └── lora_keys.py               # LoRA key translation for Wan
└── sampler_core/
    ├── gui/
    │   ├── app_base.py            # Abstract base GUI (queue, LoRA panel, preview, file import)
    │   ├── launcher.py            # Model selector launcher
    │   ├── prompt_library.py      # Prompt Library window and auto-save logic
    │   ├── theme.py               # Dark theme
    │   └── tooltip.py             # Tooltip widget
    ├── backend/
    │   └── base.py                # Abstract sampler backend
    ├── lora/
    │   ├── hooks.py               # LoRA forward hook injection (merge + GGUF compile-friendly)
    │   └── key_util.py            # Key format utilities
    └── util/
        ├── config.py              # JSON config persistence
        ├── dtype_maps.py          # Weight dtype mappings
        ├── file_import.py         # File metadata extraction (our format + ComfyUI)
        ├── gguf_util.py           # GGUF file utilities
        ├── ot_bridge.py           # OneTrainer workspace integration
        ├── png_meta.py            # PNG/MP4 metadata embedding and reading
        ├── resolution.py          # Resolution / aspect ratio calculation
        └── tokenizer_patch.py     # Tokenizer truncation patch
```

## License

GNU Affero General Public License v3 — see [LICENSE](LICENSE) for details.

This project depends on [OneTrainer](https://github.com/Nerogar/OneTrainer), also licensed under AGPLv3.

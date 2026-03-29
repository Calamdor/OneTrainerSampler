"""ChromaSamplerApp — Chroma-specific sampler GUI.

Implements the abstract interface of BaseSamplerApp.  All shared logic
(queue, lora panel, output panel, generate/abort, blink, token counter)
lives in the base class.
"""
import os
import random

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import sampler_core  # noqa: F401 — inject OT path

from sampler_core.gui.app_base import BaseSamplerApp
from sampler_core.gui.tooltip import Tooltip
from sampler_core.gui.theme import BG, BLUE
from sampler_core.util.dtype_maps import (
    TRANSFORMER_DTYPE_OPTIONS, TEXT_ENC_DTYPE_OPTIONS,
    COMPUTE_DTYPE_OPTIONS, SVD_DTYPE_OPTIONS,
)
from sampler_core.util.resolution import (
    ATTN_BACKEND_OPTIONS, ASPECT_RATIO_LABELS, PIXEL_TARGET_OPTIONS,
    compute_dims,
)
from chroma.backend import ChromaBackend

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_NEGATIVE = (
    "This low-quality, greyscale, unfinished sketch is inaccurate and flawed. "
    "The image is very blurred and lacks detail, with excessive chromatic "
    "aberrations and artifacts. The image is overly saturated with excessive bloom."
)

DEFAULTS = {
    "base_model":       "lodestones/Chroma1-HD",
    "transformer_gguf": "",
    "weight_dtype":     "BF16",
    "text_enc_dtype":   "BF16",
    "svd_enabled":      False,
    "svd_rank":         16,
    "svd_dtype":        "BF16",
    "quant_layer_filter":        "",
    "quant_layer_filter_preset": "full",
    "quant_cache_dir":  "",
    "text_cache_enabled": False,
    "use_compile":      False,
    "compute_dtype":    "Auto",
    "fast_fp16_accum":  False,
    "offload_enabled":  False,
    "offload_fraction": 50,
    "attn_backend":     "Auto",
    "loras":            [],
    "prompt":           "",
    "negative_prompt":  _DEFAULT_NEGATIVE,
    "pixel_target":     1024,
    "aspect_ratio":     "1:1",
    "scheduler":        "Euler",
    "sigma_shift":      3.0,
    "steps":            30,
    "cfg_scale":        3.5,
    "seed":             42,
    "random_seed":      False,
    "output_dir":       os.path.join(os.path.dirname(_SCRIPT_DIR), "output"),
}

CONFIG_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), "config", "chroma_sampler_config.json")


class ChromaSamplerApp(BaseSamplerApp):
    def __init__(self, root: tk.Tk, container: tk.Frame | None = None):
        super().__init__(root, ChromaBackend(), DEFAULTS, CONFIG_PATH, "Chroma Sampler",
                         container=container)

    # ==================================================================
    # Abstract interface — model frame
    # ==================================================================

    def _build_model_frame(self, pad: dict) -> None:
        model_frame = ttk.LabelFrame(self.frame, text="Model Settings")
        model_frame.pack(fill="x", **pad)

        r = 0
        ttk.Label(model_frame, text="Base model:").grid(row=r, column=0, sticky="w", **pad)
        self._base_model_var = tk.StringVar(value=self.cfg.get("base_model", ""))
        _base_entry = ttk.Entry(model_frame, textvariable=self._base_model_var, width=55)
        _base_entry.grid(row=r, column=1, sticky="ew", **pad)
        Tooltip(_base_entry,
                "Path to the Chroma base model directory (HuggingFace diffusers format).\n\n"
                "Expected contents: transformer/, text_encoder/, tokenizer/,\n"
                "scheduler/scheduler_config.json, model_index.json.\n\n"
                "Example: lodestones/Chroma1-HD  or  D:/models/Chroma1-HD")
        ttk.Button(model_frame, text="Browse…", command=self._browse_model).grid(
            row=r, column=2, **pad)

        r += 1
        dtype_row = ttk.Frame(model_frame)
        dtype_row.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(dtype_row, text="Transformer dtype:").pack(side="left")
        self._dtype_var = tk.StringVar(value=self.cfg.get("weight_dtype", "NF4"))
        _dtype_cb = ttk.Combobox(
            dtype_row, textvariable=self._dtype_var,
            values=TRANSFORMER_DTYPE_OPTIONS, state="readonly", width=10,
        )
        _dtype_cb.pack(side="left", padx=(4, 16))
        Tooltip(_dtype_cb,
                "Weight dtype for the transformer (DiT).\n\n"
                "BF16 / FP16  — full precision; best quality, highest VRAM.\n"
                "NF4          — 4-bit NormalFloat; good quality, ~4× less VRAM.\n"
                "W8A8         — 8-bit weight + 8-bit activation (SVDQuant).\n"
                "GGUF         — loads a .gguf file; dequantizes to BF16 at runtime.\n"
                "GGUF_A8I     — GGUF + int8 activation requant; torch.compile-friendly. Requires RTX 3000+ (sm80+).\n"
                "GGUF_A8F     — GGUF + fp8 activation requant; torch.compile-friendly. Requires RTX 4000+ (sm89+).\n\n"
                "torch.compile benefits BF16/FP16 and all GGUF variants.\n"
                "NF4/INT8/FP8/W8A8 use custom kernels that compile cannot optimize.")
        ttk.Label(dtype_row, text="Text enc dtype:").pack(side="left")
        self._te_dtype_var = tk.StringVar(value=self.cfg.get("text_enc_dtype", "BF16"))
        _te_dtype_cb = ttk.Combobox(
            dtype_row, textvariable=self._te_dtype_var,
            values=TEXT_ENC_DTYPE_OPTIONS, state="readonly", width=7,
        )
        _te_dtype_cb.pack(side="left", padx=(4, 16))
        Tooltip(_te_dtype_cb,
                "Weight dtype for the T5-XXL text encoder.\n\n"
                "BF16 — recommended; good quality, moderate VRAM.\n"
                "NF4  — 4-bit quantization; less VRAM at slight quality cost.\n"
                "FP32 — highest precision; rarely needed.\n\n"
                "The text encoder is always offloaded to CPU after encoding\n"
                "to free VRAM for the transformer.")
        self._offload_enabled_var = tk.BooleanVar(value=self.cfg.get("offload_enabled", False))
        _offload_chk = ttk.Checkbutton(dtype_row, text="Layer offload",
                                       variable=self._offload_enabled_var)
        _offload_chk.pack(side="left")
        Tooltip(_offload_chk,
                "Layer offload: streams transformer blocks CPU↔GPU during inference.\n\n"
                "Allows running models that don't fit in VRAM at the cost of speed.\n"
                "The fraction below controls what percentage of blocks stay on GPU;\n"
                "lower = less VRAM used, more CPU↔GPU transfers, slower inference.\n\n"
                "Not compatible with torch.compile (compile is disabled automatically).")
        self._offload_fraction_var = tk.IntVar(value=self.cfg.get("offload_fraction", 50))
        _offload_spin = ttk.Spinbox(
            dtype_row, textvariable=self._offload_fraction_var,
            from_=1, to=99, increment=5, width=4,
        )
        _offload_spin.pack(side="left", padx=(2, 1))
        Tooltip(_offload_spin,
                "Percentage of transformer blocks to keep on GPU during inference.\n\n"
                "100% = all blocks on GPU (no offloading, maximum speed).\n"
                "50%  = half on GPU (moderate VRAM saving).\n"
                "1%   = nearly all on CPU (minimum VRAM, very slow).")
        ttk.Label(dtype_row, text="%", foreground="gray").pack(side="left", padx=(0, 12))
        ttk.Label(dtype_row, text="Attn:").pack(side="left", padx=(4, 2))
        self._attn_var = tk.StringVar(value=self.cfg.get("attn_backend", "Auto"))
        _attn_cb = ttk.Combobox(
            dtype_row, textvariable=self._attn_var,
            values=ATTN_BACKEND_OPTIONS, state="readonly", width=9,
        )
        _attn_cb.pack(side="left", padx=(0, 4))
        Tooltip(_attn_cb,
                "Attention kernel backend.\n\n"
                "Auto      — uses Flash Attention if available, else PyTorch SDPA.\n"
                "Flash     — xformers / flash_attn; fast, low memory (requires install).\n"
                "SageAttn  — SageAttention; fastest on supported GPUs (requires install).\n"
                "PyTorch   — built-in SDPA; always available, slightly slower.\n\n"
                "Availability is shown in the ✓/✗ indicators to the right.")
        self._attn_avail_var = tk.StringVar()
        ttk.Label(dtype_row, textvariable=self._attn_avail_var,
                  font=("TkDefaultFont", 8), foreground="gray").pack(side="left")

        r += 1
        compute_row = ttk.Frame(model_frame)
        compute_row.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(compute_row, text="Compute dtype:").pack(side="left")
        self._compute_dtype_var = tk.StringVar(value=self.cfg.get("compute_dtype", "Auto"))
        _compute_cb = ttk.Combobox(
            compute_row, textvariable=self._compute_dtype_var,
            values=COMPUTE_DTYPE_OPTIONS, state="readonly", width=7,
        )
        _compute_cb.pack(side="left", padx=(4, 16))
        Tooltip(_compute_cb,
                "Autocast dtype for compute (matmuls, attention).\n\n"
                "Auto  — matches the transformer weight dtype (recommended).\n"
                "BF16  — brain-float16; wider dynamic range, no INF/NaN overflow.\n"
                "FP16  — float16; slightly faster on some hardware.\n\n"
                "WARNING: FP16 compute with BF16 weights can produce NaN outputs\n"
                "because BF16 exponent range exceeds FP16 max. Use BF16 or Auto.")
        self._fast_fp16_var = tk.BooleanVar(value=self.cfg.get("fast_fp16_accum", False))
        _fast_fp16_chk = ttk.Checkbutton(compute_row, text="Fast FP16 accum",
                                         variable=self._fast_fp16_var)
        _fast_fp16_chk.pack(side="left")
        Tooltip(_fast_fp16_chk,
                "Enable reduced-precision FP16 accumulation in CUDA matmuls.\n\n"
                "Sets torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction.\n"
                "Faster matrix multiplications on Ampere+ GPUs (RTX 3000+).\n\n"
                "May reduce numerical precision in edge cases. Generally safe for\n"
                "inference. No effect if compute dtype is BF16 or FP32.")
        ttk.Label(compute_row,
                  text="(Auto = tied to weight dtype;  FP16 accum = faster matmuls, Ampere+)",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="left", padx=8)

        r += 1
        # GGUF row — shown only when dtype == "GGUF"
        self._gguf_frame = ttk.Frame(model_frame)
        self._gguf_frame.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(self._gguf_frame, text="Transformer GGUF:").pack(side="left")
        self._gguf_var = tk.StringVar(value=self.cfg.get("transformer_gguf", ""))
        _gguf_entry = ttk.Entry(self._gguf_frame, textvariable=self._gguf_var, width=52)
        _gguf_entry.pack(side="left", padx=4)
        Tooltip(_gguf_entry,
                "Path to a GGUF-quantized transformer file.\n\n"
                "Required for GGUF, GGUF_A8I, and GGUF_A8F dtypes.\n"
                "The base model path is still required for the text encoder,\n"
                "tokenizer, and scheduler config.")
        ttk.Button(self._gguf_frame, text="Browse…",
                   command=lambda: self._browse_gguf(self._gguf_var)).pack(side="left", padx=2)
        self._gguf_frame.grid_remove()

        r += 1
        cache_row = ttk.Frame(model_frame)
        cache_row.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(cache_row, text="Quant cache:").pack(side="left")
        self._cache_var = tk.StringVar(value=self.cfg.get("quant_cache_dir", ""))
        _cache_entry = ttk.Entry(cache_row, textvariable=self._cache_var, width=48)
        _cache_entry.pack(side="left", padx=4)
        Tooltip(_cache_entry,
                "Directory for quantization cache files (NF4, SVDQuant, etc.).\n\n"
                "First load computes quantization and saves to this directory.\n"
                "Subsequent loads read from cache — dramatically faster startup.\n\n"
                "Leave empty to auto-detect from OneTrainer's active config.\n"
                "Text encoder embeddings (if cached) are stored in a te_cache/\n"
                "subdirectory inside this directory.")
        ttk.Button(cache_row, text="Browse…", command=self._browse_cache_dir).pack(side="left", padx=2)
        _from_ot_btn = ttk.Button(cache_row, text="From OT",  command=self._use_ot_cache_dir)
        _from_ot_btn.pack(side="left", padx=2)
        Tooltip(_from_ot_btn,
                "Read cache_dir from OneTrainer's active training config\n"
                "(OneTrainer/config.json) and set it as the quant cache directory.\n\n"
                "Open a training config in OT first, then click this to sync.")
        ttk.Label(cache_row, text="(empty = auto from OT config)",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="left", padx=4)
        self._text_cache_var = tk.BooleanVar(value=self.cfg.get("text_cache_enabled", False))
        _tc_chk = ttk.Checkbutton(cache_row, text="Text embed cache",
                                  variable=self._text_cache_var)
        _tc_chk.pack(side="left", padx=(16, 2))
        Tooltip(_tc_chk,
            "Cache T5 text encoder embeddings to disk.\n\n"
            "On first use: runs T5 and saves the result under Cache dir/te_cache/.\n"
            "On subsequent uses with the same prompt + precision: skips T5 entirely.\n\n"
            "Cache key includes: positive prompt, negative prompt, text enc dtype.\n"
            "Useful when iterating on steps/cfg/seed while keeping the same prompt."
        )

        r += 1
        svd_row = ttk.Frame(model_frame)
        svd_row.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        self._svd_var = tk.BooleanVar(value=self.cfg.get("svd_enabled", False))
        _svd_chk = ttk.Checkbutton(svd_row, text="SVDQuant", variable=self._svd_var)
        _svd_chk.pack(side="left")
        Tooltip(_svd_chk,
            "SVDQuant: improves quality at a given bit-width by extracting a low-rank\n"
            "correction from the weight matrix before quantization.\n\n"
            "Uses: transformer weights (set by Weight dtype above).\n"
            "rank — number of singular vectors kept in full precision.\n"
            "residual dtype — precision of the SVD correction factors (U·S and Vᵀ);\n"
            "  the remaining residual is stored in the Weight dtype (NF4 / INT8 etc.).\n\n"
            "First load computes full SVD (slow); set Quant cache dir to save results\n"
            "so subsequent loads are instant."
        )
        _lbl_rank = ttk.Label(svd_row, text="rank:")
        _lbl_rank.pack(side="left", padx=(8, 2))
        self._svd_rank_var = tk.IntVar(value=self.cfg.get("svd_rank", 16))
        _svd_spin = ttk.Spinbox(svd_row, textvariable=self._svd_rank_var,
                                from_=1, to=256, increment=1, width=5)
        _svd_spin.pack(side="left", padx=(0, 8))
        _lbl_res = ttk.Label(svd_row, text="residual dtype:")
        _lbl_res.pack(side="left")
        self._svd_dtype_var = tk.StringVar(value=self.cfg.get("svd_dtype", "BF16"))
        _svd_combo = ttk.Combobox(svd_row, textvariable=self._svd_dtype_var,
                                  values=SVD_DTYPE_OPTIONS, state="readonly", width=6)
        _svd_combo.pack(side="left", padx=4)
        self._compile_var = tk.BooleanVar(value=self.cfg.get("use_compile", False))
        _compile_chk = ttk.Checkbutton(svd_row, text="torch.compile",
                                       variable=self._compile_var)
        _compile_chk.pack(side="left", padx=(24, 0))
        Tooltip(_compile_chk,
                "torch.compile — fuses operations for faster inference.\n\n"
                "Only beneficial for float weight dtypes (BF16, FP16).\n"
                "Has NO effect on quantized weights (NF4, W8A8, GGUF, etc.) —\n"
                "those use custom CUDA kernels that compile cannot optimize.\n\n"
                "First sample after loading will be slow (compile warmup).\n"
                "Subsequent samples reuse the cached compiled graph.")
        self._svd_row_widgets = [_svd_chk, _lbl_rank, _svd_spin, _lbl_res, _svd_combo]

        r += 1
        _LAYER_PRESETS = ["full", "blocks", "attn-mlp", "attn-only"]
        _LAYER_PRESET_FILTERS = {
            "full": "", "blocks": "transformer_blocks",
            "attn-mlp": "attn,ff.net", "attn-only": "attn",
        }
        qf_row = ttk.Frame(model_frame)
        qf_row.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(qf_row, text="Quant layer filter:").pack(side="left")
        self._qf_preset_var = tk.StringVar(
            value=self.cfg.get("quant_layer_filter_preset", "full"))
        _qf_combo = ttk.Combobox(qf_row, textvariable=self._qf_preset_var,
                                 values=_LAYER_PRESETS, state="readonly", width=10)
        _qf_combo.pack(side="left", padx=4)
        Tooltip(_qf_combo,
            "Preset defining which layers to quantize.\n\n"
            "full — quantize all linear layers (default)\n"
            "blocks — transformer blocks only (skip embedders/proj_out)\n"
            "attn-mlp — attention + MLP layers only\n"
            "attn-only — attention layers only\n\n"
            "Selecting a preset fills the custom filter box.")
        self._qf_entry_var = tk.StringVar(
            value=self.cfg.get("quant_layer_filter", ""))
        _qf_entry = ttk.Entry(qf_row, textvariable=self._qf_entry_var, width=30)
        _qf_entry.pack(side="left", padx=4)
        Tooltip(_qf_entry,
            "Custom comma-separated layer filter for quantization.\n"
            "Any linear layer whose name contains one of these substrings\n"
            "will be quantized; others stay in the base weight dtype.\n\n"
            "Leave empty to quantize all layers (same as 'full' preset).")
        def _on_preset_change(*_):
            preset = self._qf_preset_var.get()
            self._qf_entry_var.set(_LAYER_PRESET_FILTERS.get(preset, ""))
        _qf_combo.bind("<<ComboboxSelected>>", _on_preset_change)

        r += 1
        self._model_status_var = tk.StringVar(value="Not loaded — will load on first job")
        self._model_status_label = tk.Label(
            model_frame, textvariable=self._model_status_var, foreground="gray",
            background=BG,
        )
        self._model_status_label.grid(row=r, column=0, columnspan=3, sticky="w", **pad)

        model_frame.columnconfigure(1, weight=1)

        self._dtype_var.trace_add("write", self._on_dtype_changed)
        self._on_dtype_changed()

    # ==================================================================
    # Abstract interface — generation params
    # ==================================================================

    def _build_gen_params(self, gen_frame: ttk.LabelFrame, r: int) -> int:
        pad = {"padx": 6, "pady": 3}

        res_row = ttk.Frame(gen_frame)
        res_row.grid(row=r, column=0, columnspan=7, sticky="w", **pad)
        ttk.Label(res_row, text="Pixels:").pack(side="left")
        self._pixel_target_var = tk.StringVar(value=str(self.cfg.get("pixel_target", 1024)))
        _pixel_cb = ttk.Combobox(
            res_row, textvariable=self._pixel_target_var,
            values=PIXEL_TARGET_OPTIONS, width=6,
        )
        _pixel_cb.pack(side="left", padx=(3, 12))
        Tooltip(_pixel_cb,
                "Target pixel count (width × height) for the generated image.\n\n"
                "The actual resolution is computed by scaling the chosen aspect ratio\n"
                "to match this pixel budget, then rounding to the nearest 16px multiple.\n\n"
                "Higher pixel counts need more VRAM and more steps for good detail.\n"
                "Chroma was trained at 1 MP (1024² equivalent) — this is the sweet spot.")
        ttk.Label(res_row, text="Aspect:").pack(side="left")
        self._aspect_var = tk.StringVar(value=self.cfg.get("aspect_ratio", "1:1"))
        _aspect_cb = ttk.Combobox(
            res_row, textvariable=self._aspect_var,
            values=ASPECT_RATIO_LABELS, state="readonly", width=7,
        )
        _aspect_cb.pack(side="left", padx=(3, 10))
        Tooltip(_aspect_cb,
                "Aspect ratio of the generated image.\n\n"
                "Combined with the Pixels target to determine the exact resolution\n"
                "(shown in blue to the right after selection).\n\n"
                "1:1 = square.  16:9 = widescreen.  9:16 = portrait.")
        self._dims_label_var = tk.StringVar(value="")
        ttk.Label(res_row, textvariable=self._dims_label_var,
                  foreground=BLUE, font=("TkDefaultFont", 8)).pack(side="left")

        for _v in (self._pixel_target_var, self._aspect_var):
            _v.trace_add("write", lambda *_: self._update_dims_label())
        self._update_dims_label()

        r += 1
        # Sampling params row — scheduler / shift / steps / cfg / seed / rnd all inline
        self._steps_var       = tk.IntVar(value=self.cfg.get("steps", 30))
        self._cfg_var         = tk.DoubleVar(value=self.cfg.get("cfg_scale", 3.5))
        self._seed_var        = tk.IntVar(value=self.cfg.get("seed", 42))
        self._rnd_var         = tk.BooleanVar(value=self.cfg.get("random_seed", False))
        self._scheduler_var   = tk.StringVar(value=self.cfg.get("scheduler", "Euler"))
        self._sigma_shift_var = tk.DoubleVar(value=self.cfg.get("sigma_shift", 3.0))

        params_row = ttk.Frame(gen_frame)
        params_row.grid(row=r, column=0, columnspan=7, sticky="w", **pad)

        ttk.Label(params_row, text="Scheduler:").pack(side="left")
        _sched_cb = ttk.Combobox(
            params_row, textvariable=self._scheduler_var,
            values=["Euler", "Heun"], state="readonly", width=6,
        )
        _sched_cb.pack(side="left", padx=(3, 0))
        Tooltip(_sched_cb,
                "Euler — 1st order, 1 model call per step (default).\n"
                "Heun  — 2nd order predictor-corrector, 2 model calls per step.\n\n"
                "Heun produces better quality per sigma interval at 2× the cost.\n"
                "Typical use: Heun at half the step count of Euler.")

        ttk.Label(params_row, text="Shift:").pack(side="left", padx=(10, 0))
        _shift_sb = ttk.Spinbox(
            params_row, textvariable=self._sigma_shift_var,
            from_=0.5, to=10.0, increment=0.5, width=4, format="%.1f",
        )
        _shift_sb.pack(side="left", padx=(3, 0))
        Tooltip(_shift_sb,
                "Sigma shift — controls how timesteps are distributed.\n\n"
                "Higher values concentrate steps toward high-noise timesteps\n"
                "(generally better at high resolution).\n"
                "Default: 3.0 (FLUX/Chroma standard). Range: 0.5 – 10.0.")

        ttk.Separator(params_row, orient="vertical").pack(
            side="left", fill="y", padx=10, pady=2)

        ttk.Label(params_row, text="Steps:").pack(side="left", padx=(0, 2))
        _steps_entry = ttk.Entry(params_row, textvariable=self._steps_var, width=5)
        _steps_entry.pack(side="left", padx=(0, 8))
        Tooltip(_steps_entry,
                "Number of diffusion steps.\n\n"
                "More steps = better quality, slower generation.\n"
                "Euler: 20–30 is a good range. Heun: 10–15 (2× NFE per step).\n"
                "Diminishing returns beyond ~40 steps.")
        ttk.Label(params_row, text="CFG:").pack(side="left", padx=(0, 2))
        _cfg_entry = ttk.Entry(params_row, textvariable=self._cfg_var, width=5)
        _cfg_entry.pack(side="left", padx=(0, 8))
        Tooltip(_cfg_entry,
                "Classifier-Free Guidance scale.\n\n"
                "Controls how strongly the image follows the prompt.\n"
                "1.0 = unconditioned (negative prompt has no effect).\n"
                "3.0–4.5 = typical range for Chroma.\n"
                "Higher values sharpen prompt adherence but can over-saturate.")
        ttk.Label(params_row, text="Seed:").pack(side="left", padx=(0, 2))
        _seed_entry = ttk.Entry(params_row, textvariable=self._seed_var, width=8)
        _seed_entry.pack(side="left", padx=(0, 4))
        Tooltip(_seed_entry,
                "Random seed for the initial noise tensor.\n\n"
                "Same seed + same settings = same image (reproducible).\n"
                "Ignored when Rnd is checked.")
        _rnd_chk = ttk.Checkbutton(params_row, text="Rnd", variable=self._rnd_var)
        _rnd_chk.pack(side="left", padx=(0, 2))
        Tooltip(_rnd_chk,
                "Random seed — generates a new random seed for each image.\n\n"
                "When checked, the Seed field is ignored.\n"
                "The seed used is shown in the output filename.")
        _gen_btn = ttk.Button(params_row, text="Gen", width=4,
                              command=lambda: (self._seed_var.set(random.randint(0, 2**31 - 1)),
                                              self._rnd_var.set(False)))
        _gen_btn.pack(side="left", padx=(0, 2))
        Tooltip(_gen_btn,
                "Generate a one-time random seed.\n\n"
                "Rolls a random number into the Seed field and clears Rnd.\n"
                "The seed stays fixed until you change it or click Gen again.")
        _reset_btn = ttk.Button(params_row, text="42", width=3,
                                command=lambda: (self._seed_var.set(42),
                                                self._rnd_var.set(False)))
        _reset_btn.pack(side="left", padx=(0, 8))
        Tooltip(_reset_btn, "Reset seed to 42 and clear Rnd.")

        return r + 1

    def _update_dims_label(self):
        try:
            target = int(self._pixel_target_var.get())
        except (ValueError, tk.TclError):
            self._dims_label_var.set("")
            return
        w, h = compute_dims(target, self._aspect_var.get())
        self._dims_label_var.set(f"→  {w} × {h}")

    # ==================================================================
    # Abstract interface — collect / LoRA
    # ==================================================================

    def _collect_cfg(self) -> dict:
        try:
            pixel = int(self._pixel_target_var.get())
        except (ValueError, AttributeError):
            pixel = 1024
        try:
            aspect = self._aspect_var.get()
        except AttributeError:
            aspect = "1:1"
        w, h = compute_dims(pixel, aspect)
        return {
            "base_model":       self._base_model_var.get(),
            "transformer_gguf": self._gguf_var.get(),
            "weight_dtype":     self._dtype_var.get(),
            "text_enc_dtype":   self._te_dtype_var.get(),
            "svd_enabled":      self._svd_var.get(),
            "svd_rank":         self._svd_rank_var.get(),
            "svd_dtype":        self._svd_dtype_var.get(),
            "quant_layer_filter":        self._qf_entry_var.get(),
            "quant_layer_filter_preset": self._qf_preset_var.get(),
            "quant_cache_dir":     self._cache_var.get(),
            "text_cache_enabled":  self._text_cache_var.get(),
            "use_compile":         self._compile_var.get(),
            "compute_dtype":    self._compute_dtype_var.get(),
            "fast_fp16_accum":  self._fast_fp16_var.get(),
            "offload_enabled":  self._offload_enabled_var.get(),
            "offload_fraction": self._offload_fraction_var.get(),
            "attn_backend":     self._attn_var.get(),
            "loras":            self._get_lora_list(),
            "prompt":           self._prompt_text.get("1.0", "end-1c"),
            "negative_prompt":  self._neg_text.get("1.0", "end-1c"),
            "pixel_target":     pixel,
            "aspect_ratio":     aspect,
            "width":            w,
            "height":           h,
            "scheduler":        self._scheduler_var.get(),
            "sigma_shift":      self._sigma_shift_var.get(),
            "steps":            self._steps_var.get(),
            "cfg_scale":        self._cfg_var.get(),
            "seed":             self._seed_var.get(),
            "random_seed":      self._rnd_var.get(),
            "output_dir":       self._outdir_var.get(),
        }

    def _add_lora(self, path: str = "", weight: float = 1.0, **kwargs) -> None:
        enabled = kwargs.get("enabled", True)
        if path == "":
            path = filedialog.askopenfilename(
                title="Select LoRA",
                filetypes=[("Safetensors", "*.safetensors"), ("All", "*.*")],
            )
            if not path:
                return

        row_frame   = ttk.Frame(self._lora_inner)
        weight_var  = tk.DoubleVar(value=weight)
        enabled_var = tk.BooleanVar(value=enabled)

        row = {"path": path, "weight_var": weight_var,
               "enabled_var": enabled_var, "frame": row_frame}

        short = os.path.basename(path)
        lbl = ttk.Label(row_frame, text=short, width=46, anchor="w")
        lbl.pack(side="left", padx=2)
        lbl.bind("<Double-Button-1>", lambda e, p=path: messagebox.showinfo("Full path", p))

        ttk.Entry(row_frame, textvariable=weight_var, width=7).pack(side="left", padx=2)
        ttk.Checkbutton(row_frame, variable=enabled_var).pack(side="left", padx=2)
        tk.Button(
            row_frame, text="✕", fg="white", bg="#cc3333",
            activebackground="#ff4444", activeforeground="white",
            relief="flat", width=2, padx=1, pady=0,
            command=lambda r=row: self._remove_lora_row(r),
        ).pack(side="left", padx=(4, 2))

        row_frame.pack(fill="x", pady=1)
        self._bind_mousewheel_recursive(row_frame)
        self._lora_rows.append(row)
        self._on_lora_frame_configure()
        self._save_cfg()

    def _get_lora_list(self) -> list[dict]:
        return [
            {
                "path":    row["path"],
                "weight":  row["weight_var"].get(),
                "enabled": row["enabled_var"].get(),
            }
            for row in self._lora_rows
        ]

    def _load_loras_from_config(self) -> None:
        for entry in self.cfg.get("loras", []):
            self._add_lora(
                path=entry.get("path", ""),
                weight=entry.get("weight", 1.0),
                enabled=entry.get("enabled", True),
            )

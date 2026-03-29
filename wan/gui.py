"""WanSamplerApp — Wan2.2 T2V-A14B sampler GUI.

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
from sampler_core.gui.theme import BG, BLUE, BG_INPUT, FG
from sampler_core.util.dtype_maps import (
    TRANSFORMER_DTYPE_OPTIONS, TEXT_ENC_DTYPE_OPTIONS,
    COMPUTE_DTYPE_OPTIONS, SVD_DTYPE_OPTIONS,
)
from sampler_core.util.resolution import (
    ATTN_BACKEND_OPTIONS, ASPECT_RATIO_LABELS,
    compute_dims, WAN_PIXEL_TARGET_OPTIONS,
)
from wan.backend import WanBackend

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "base_model_path":    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "transformer_1_gguf": "",
    "transformer_2_gguf": "",
    "weight_dtype":       "BF16",
    "text_enc_dtype":     "BF16",
    "svd_enabled":        False,
    "svd_rank":           16,
    "svd_dtype":          "BF16",
    "quant_layer_filter":        "",
    "quant_layer_filter_preset": "full",
    "quant_cache_dir":    "",
    "use_compile":        False,
    "compute_dtype":      "Auto",
    "fast_fp16_accum":    False,
    "offload_enabled":    False,
    "offload_fraction":   50,
    "attn_backend":       "Auto",
    "scheduler":          "Euler",
    "text_cache_enabled": False,
    "loras":              [],
    "prompt":             "",
    "negative_prompt":    "",
    "pixel_target":       640,
    "aspect_ratio":       "16:9",
    "frames":             81,
    "cfg_scale":          5.0,
    "cfg_scale_2":        5.0,
    "steps_high":         20,
    "steps_low":          0,
    "seed":               42,
    "random_seed":        False,
    "output_dir":         os.path.join(os.path.dirname(_SCRIPT_DIR), "output"),
}

CONFIG_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), "config", "wan_sampler_config.json")


class WanSamplerApp(BaseSamplerApp):
    def __init__(self, root: tk.Tk, container: tk.Frame | None = None):
        super().__init__(root, WanBackend(), DEFAULTS, CONFIG_PATH, "Wan2.2 Sampler",
                         container=container)

    # ==================================================================
    # Abstract interface — model frame
    # ==================================================================

    def _build_model_frame(self, pad: dict) -> None:
        model_frame = ttk.LabelFrame(self.frame, text="Model Settings")
        model_frame.pack(fill="x", **pad)

        r = 0
        ttk.Label(model_frame, text="Base model:").grid(row=r, column=0, sticky="w", **pad)
        self._base_model_var = tk.StringVar(value=self.cfg.get("base_model_path", ""))
        _base_entry = ttk.Entry(model_frame, textvariable=self._base_model_var, width=55)
        _base_entry.grid(row=r, column=1, sticky="ew", **pad)
        Tooltip(_base_entry,
                "Path to the Wan2.2 base model directory (HuggingFace diffusers format).\n\n"
                "Expected contents: transformer/, transformer_2/, text_encoder/,\n"
                "tokenizer/, scheduler/, model_index.json.\n\n"
                "Example: Wan-AI/Wan2.2-T2V-A14B-Diffusers  or  D:/models/Wan2.2-A14B")
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
                "Weight dtype for both transformer experts (HIGH and LOW).\n\n"
                "BF16 / FP16  — full precision; best quality, highest VRAM.\n"
                "NF4          — 4-bit NormalFloat; good quality, ~4× less VRAM.\n"
                "GGUF         — loads .gguf files for T1 and T2; dequantizes to BF16 at runtime.\n"
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
                "NF4  — 4-bit quantization; less VRAM at slight quality cost.\n\n"
                "The text encoder is always offloaded to CPU after encoding.")
        self._offload_enabled_var = tk.BooleanVar(value=self.cfg.get("offload_enabled", False))
        _offload_chk = ttk.Checkbutton(dtype_row, text="Layer offload",
                                       variable=self._offload_enabled_var)
        _offload_chk.pack(side="left")
        Tooltip(_offload_chk,
                "Layer offload: streams transformer blocks CPU↔GPU during inference.\n\n"
                "Allows running the 14B dual-expert model with limited VRAM.\n"
                "The fraction below controls what percentage of blocks stay on GPU;\n"
                "lower = less VRAM used, more CPU↔GPU transfers, slower inference.")
        self._offload_fraction_var = tk.IntVar(value=self.cfg.get("offload_fraction", 50))
        _offload_spin = ttk.Spinbox(
            dtype_row, textvariable=self._offload_fraction_var,
            from_=1, to=99, increment=5, width=4,
        )
        _offload_spin.pack(side="left", padx=(2, 1))
        Tooltip(_offload_spin,
                "Percentage of transformer blocks to keep on GPU per expert.\n\n"
                "99% = almost all blocks on GPU (minimal offloading).\n"
                "50% = half on GPU (moderate VRAM saving).\n"
                "1%  = nearly all on CPU (minimum VRAM, very slow).")
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
                "Auto     — uses Flash Attention if available, else PyTorch SDPA.\n"
                "Flash    — flash_attn; fast, low memory (requires install).\n"
                "SageAttn — SageAttention; fastest on supported GPUs (requires install).\n"
                "           Note: may produce Q/K int8 overflow artifacts on Wan2.2.")
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
                "BF16  — brain-float16; wider dynamic range.\n"
                "FP16  — float16; slightly faster on some hardware.\n\n"
                "WARNING: FP16 compute with BF16 weights can produce NaN outputs.")
        self._fast_fp16_var = tk.BooleanVar(value=self.cfg.get("fast_fp16_accum", False))
        _fast_fp16_chk = ttk.Checkbutton(compute_row, text="Fast FP16 accum",
                                         variable=self._fast_fp16_var)
        _fast_fp16_chk.pack(side="left")
        Tooltip(_fast_fp16_chk,
                "Enable reduced-precision FP16 accumulation in CUDA matmuls.\n\n"
                "Faster on Ampere+ GPUs (RTX 3000+). Generally safe for inference.")
        ttk.Label(compute_row,
                  text="(Auto = tied to weight dtype;  FP16 accum = faster matmuls, Ampere+)",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="left", padx=8)

        r += 1
        # GGUF row — shown only when dtype == "GGUF"; contains two file paths (T1 and T2)
        self._gguf_frame = ttk.Frame(model_frame)
        self._gguf_frame.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(self._gguf_frame, text="T1 GGUF:").pack(side="left")
        self._t1_gguf_var = tk.StringVar(value=self.cfg.get("transformer_1_gguf", ""))
        _t1_entry = ttk.Entry(self._gguf_frame, textvariable=self._t1_gguf_var, width=38)
        _t1_entry.pack(side="left", padx=(3, 2))
        Tooltip(_t1_entry, "Path to GGUF file for the HIGH-noise expert (transformer 1).")
        ttk.Button(self._gguf_frame, text="Browse…",
                   command=lambda: self._browse_gguf(self._t1_gguf_var)).pack(side="left", padx=(0, 10))
        ttk.Label(self._gguf_frame, text="T2 GGUF:").pack(side="left")
        self._t2_gguf_var = tk.StringVar(value=self.cfg.get("transformer_2_gguf", ""))
        _t2_entry = ttk.Entry(self._gguf_frame, textvariable=self._t2_gguf_var, width=38)
        _t2_entry.pack(side="left", padx=(3, 2))
        Tooltip(_t2_entry, "Path to GGUF file for the LOW-noise expert (transformer 2).")
        ttk.Button(self._gguf_frame, text="Browse…",
                   command=lambda: self._browse_gguf(self._t2_gguf_var)).pack(side="left", padx=2)
        self._gguf_frame.grid_remove()

        r += 1
        cache_row = ttk.Frame(model_frame)
        cache_row.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(cache_row, text="Quant cache:").pack(side="left")
        self._cache_var = tk.StringVar(value=self.cfg.get("quant_cache_dir", ""))
        _cache_entry = ttk.Entry(cache_row, textvariable=self._cache_var, width=50)
        _cache_entry.pack(side="left", padx=4)
        Tooltip(_cache_entry,
                "Directory for quantization cache files (NF4, SVDQuant, etc.).\n\n"
                "First load computes quantization and saves here; subsequent loads\n"
                "read from cache — much faster startup.\n\n"
                "Leave empty to auto-detect from OneTrainer's active config.")
        ttk.Button(cache_row, text="Browse…", command=self._browse_cache_dir).pack(side="left", padx=2)
        _from_ot_btn = ttk.Button(cache_row, text="From OT", command=self._use_ot_cache_dir)
        _from_ot_btn.pack(side="left", padx=2)
        Tooltip(_from_ot_btn,
                "Read cache_dir from OneTrainer's active training config and set it\n"
                "as the quant cache directory.  Open a config in OT first.")
        ttk.Label(cache_row, text="(empty = auto from OT config)",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="left", padx=4)
        self._text_cache_var = tk.BooleanVar(value=self.cfg.get("text_cache_enabled", False))
        _tc_chk = ttk.Checkbutton(cache_row, text="Cache T5 embeddings",
                                  variable=self._text_cache_var)
        _tc_chk.pack(side="left", padx=(12, 0))
        Tooltip(_tc_chk,
                "Cache UMT5 text encoder embeddings to disk.\n\n"
                "First generation encodes and saves both positive and negative\n"
                "embeddings; subsequent generations with the same prompts load\n"
                "from cache — T5 never moves to GPU VRAM on a cache hit.\n\n"
                "Cache is stored in quant_cache_dir/te_cache/  (or output/ if unset).")

        r += 1
        svd_row = ttk.Frame(model_frame)
        svd_row.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        self._svd_var = tk.BooleanVar(value=self.cfg.get("svd_enabled", False))
        _svd_chk = ttk.Checkbutton(svd_row, text="SVDQuant", variable=self._svd_var)
        _svd_chk.pack(side="left")
        Tooltip(_svd_chk,
                "SVDQuant: extracts a low-rank correction from weights before quantization\n"
                "to improve quality at a given bit-width.\n\n"
                "First load computes full SVD (slow); set Quant cache to save results.")
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
                "First sample after loading will be slow (compile warmup).")
        # Required by _on_dtype_changed in base class
        self._svd_row_widgets = [_svd_chk, _lbl_rank, _svd_spin, _lbl_res, _svd_combo]

        r += 1
        _LAYER_PRESETS = ["full", "blocks", "attn-mlp", "attn-only"]
        _LAYER_PRESET_FILTERS = {
            "full": "", "blocks": "blocks",
            "attn-mlp": "attn,ffn", "attn-only": "attn",
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

        # Resolution row: pixel target + aspect ratio + dims label + frames
        res_row = ttk.Frame(gen_frame)
        res_row.grid(row=r, column=0, columnspan=7, sticky="w", **pad)

        ttk.Label(res_row, text="Pixels:").pack(side="left")
        self._pixel_target_var = tk.StringVar(
            value=str(self.cfg.get("pixel_target", 640)))
        _pixel_cb = ttk.Combobox(
            res_row, textvariable=self._pixel_target_var,
            values=WAN_PIXEL_TARGET_OPTIONS, width=5,
        )
        _pixel_cb.pack(side="left", padx=(3, 12))
        Tooltip(_pixel_cb,
                "Target pixel size for the generated video frame.\n\n"
                "The actual resolution is computed by scaling the chosen aspect ratio\n"
                "to this pixel budget (quantized to 16px), shown in blue to the right.\n\n"
                "Typical 16:9 outputs at quantize=16:\n"
                "  960 → 1280×720 (720p)\n"
                "  720 → 960×544\n"
                "  640 → 848×480\n"
                "  480 → 640×368\n"
                "  320 → 432×240")

        ttk.Label(res_row, text="Aspect:").pack(side="left")
        self._aspect_var = tk.StringVar(value=self.cfg.get("aspect_ratio", "16:9"))
        _aspect_cb = ttk.Combobox(
            res_row, textvariable=self._aspect_var,
            values=ASPECT_RATIO_LABELS, state="readonly", width=7,
        )
        _aspect_cb.pack(side="left", padx=(3, 10))
        Tooltip(_aspect_cb,
                "Aspect ratio of the generated video frame.\n\n"
                "Combined with the Pixels target to determine exact resolution\n"
                "(shown in blue). Wan2.2 was trained on 16:9 and 9:16 content.")

        self._dims_label_var = tk.StringVar(value="")
        ttk.Label(res_row, textvariable=self._dims_label_var,
                  foreground=BLUE, font=("TkDefaultFont", 8)).pack(side="left", padx=(0, 16))

        ttk.Label(res_row, text="Frames:").pack(side="left")
        self._frames_var = tk.IntVar(value=self.cfg.get("frames", 81))
        _frames_entry = ttk.Entry(res_row, textvariable=self._frames_var, width=5)
        _frames_entry.pack(side="left", padx=(3, 12))
        Tooltip(_frames_entry,
                "Number of video frames to generate.\n\n"
                "Wan2.2 was trained on 4×N+1 frame counts: 1, 5, 9, … 81.\n"
                "81 frames at typical resolution produces ~3s video at ~24fps.\n"
                "Set to 1 to generate a single image (PNG output, no sidecar).")

        ttk.Label(res_row, text="Scheduler:").pack(side="left")
        self._scheduler_var = tk.StringVar(value=self.cfg.get("scheduler", "Euler"))
        _sched_cb = ttk.Combobox(
            res_row, textvariable=self._scheduler_var,
            values=["Euler", "Heun"], state="readonly", width=6,
        )
        _sched_cb.pack(side="left", padx=(3, 0))
        Tooltip(_sched_cb,
                "Diffusion scheduler.\n\n"
                "Euler — standard first-order flow matching (default).\n"
                "Heun  — second-order predictor-corrector; higher quality\n"
                "        at the same step count, but 2× NFE per step.\n\n"
                "Wan2.2 calculates sigma shift automatically — no manual\n"
                "shift control is needed.")

        for _v in (self._pixel_target_var, self._aspect_var):
            _v.trace_add("write", lambda *_: self._update_dims_label())
        self._update_dims_label()

        r += 1
        # Steps / CFG row
        params_row = ttk.Frame(gen_frame)
        params_row.grid(row=r, column=0, columnspan=7, sticky="w", **pad)

        self._steps_high_var = tk.IntVar(value=self.cfg.get("steps_high", 20))
        self._steps_low_var  = tk.IntVar(value=self.cfg.get("steps_low", 0))
        self._cfg_var        = tk.DoubleVar(value=self.cfg.get("cfg_scale", 5.0))
        self._cfg2_var       = tk.DoubleVar(value=self.cfg.get("cfg_scale_2", 5.0))
        self._seed_var       = tk.IntVar(value=self.cfg.get("seed", 42))
        self._rnd_var        = tk.BooleanVar(value=self.cfg.get("random_seed", False))

        ttk.Label(params_row, text="Steps (HIGH):").pack(side="left")
        _sh_entry = ttk.Entry(params_row, textvariable=self._steps_high_var, width=5)
        _sh_entry.pack(side="left", padx=(3, 8))
        Tooltip(_sh_entry,
                "Diffusion steps for the HIGH-noise expert (first phase).\n\n"
                "Controls how much the HIGH expert refines the initial latent.\n"
                "Typical range: 10–25. Higher = better quality, slower generation.")

        ttk.Label(params_row, text="Steps (LOW):").pack(side="left")
        _sl_entry = ttk.Entry(params_row, textvariable=self._steps_low_var, width=5)
        _sl_entry.pack(side="left", padx=(3, 8))
        Tooltip(_sl_entry,
                "Diffusion steps for the LOW-noise expert (second phase).\n\n"
                "Controls how much the LOW expert refines the denoised latent.\n"
                "Set to 0 to skip the LOW expert entirely.")

        ttk.Separator(params_row, orient="vertical").pack(
            side="left", fill="y", padx=8, pady=2)

        ttk.Label(params_row, text="CFG:").pack(side="left")
        _cfg_entry = ttk.Entry(params_row, textvariable=self._cfg_var, width=5)
        _cfg_entry.pack(side="left", padx=(3, 8))
        Tooltip(_cfg_entry,
                "Classifier-Free Guidance scale for the HIGH-noise expert.\n\n"
                "Controls how strongly the output follows the prompt.\n"
                "Typical range: 3.0–7.0.")

        ttk.Label(params_row, text="CFG2:").pack(side="left")
        _cfg2_entry = ttk.Entry(params_row, textvariable=self._cfg2_var, width=5)
        _cfg2_entry.pack(side="left", padx=(3, 8))
        Tooltip(_cfg2_entry,
                "CFG scale for the LOW-noise expert.\n\n"
                "Ignored when Steps (LOW) is 0.\n"
                "Typical range: 3.0–7.0.")

        ttk.Separator(params_row, orient="vertical").pack(
            side="left", fill="y", padx=8, pady=2)

        ttk.Label(params_row, text="Seed:").pack(side="left")
        _seed_entry = ttk.Entry(params_row, textvariable=self._seed_var, width=8)
        _seed_entry.pack(side="left", padx=(3, 4))
        Tooltip(_seed_entry,
                "Random seed for the initial noise tensor.\n\n"
                "Same seed + same settings = same video (reproducible).\n"
                "Ignored when Rnd is checked.")
        _rnd_chk = ttk.Checkbutton(params_row, text="Rnd", variable=self._rnd_var)
        _rnd_chk.pack(side="left", padx=(0, 2))
        Tooltip(_rnd_chk,
                "Random seed — generates a new random seed for each video.\n\n"
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
        w, h = compute_dims(target, self._aspect_var.get(), quantize=16)
        self._dims_label_var.set(f"→  {w} × {h}")

    def _get_total_steps(self, cfg: dict) -> int:
        return cfg.get("steps_high", 20) + cfg.get("steps_low", 0)

    # ==================================================================
    # Abstract interface — collect / LoRA
    # ==================================================================

    def _collect_cfg(self) -> dict:
        try:
            pixel = int(self._pixel_target_var.get())
        except (ValueError, AttributeError):
            pixel = 640
        try:
            aspect = self._aspect_var.get()
        except AttributeError:
            aspect = "16:9"
        w, h = compute_dims(pixel, aspect, quantize=16)
        return {
            "base_model_path":    self._base_model_var.get(),
            "transformer_1_gguf": self._t1_gguf_var.get(),
            "transformer_2_gguf": self._t2_gguf_var.get(),
            "weight_dtype":       self._dtype_var.get(),
            "text_enc_dtype":     self._te_dtype_var.get(),
            "svd_enabled":        self._svd_var.get(),
            "svd_rank":           self._svd_rank_var.get(),
            "svd_dtype":          self._svd_dtype_var.get(),
            "quant_layer_filter":        self._qf_entry_var.get(),
            "quant_layer_filter_preset": self._qf_preset_var.get(),
            "quant_cache_dir":    self._cache_var.get(),
            "use_compile":        self._compile_var.get(),
            "compute_dtype":      self._compute_dtype_var.get(),
            "fast_fp16_accum":    self._fast_fp16_var.get(),
            "offload_enabled":    self._offload_enabled_var.get(),
            "offload_fraction":   self._offload_fraction_var.get(),
            "attn_backend":       self._attn_var.get(),
            "scheduler":          self._scheduler_var.get(),
            "text_cache_enabled": self._text_cache_var.get(),
            "loras":              self._get_lora_list(),
            "prompt":             self._prompt_text.get("1.0", "end-1c"),
            "negative_prompt":    self._neg_text.get("1.0", "end-1c"),
            "pixel_target":       pixel,
            "aspect_ratio":       aspect,
            "width":              w,
            "height":             h,
            "frames":             self._frames_var.get(),
            "cfg_scale":          self._cfg_var.get(),
            "cfg_scale_2":        self._cfg2_var.get(),
            "steps_high":         self._steps_high_var.get(),
            "steps_low":          self._steps_low_var.get(),
            "seed":               self._seed_var.get(),
            "random_seed":        self._rnd_var.get(),
            "output_dir":         self._outdir_var.get(),
        }

    def _add_lora(self, path: str = "", weight: float = 1.0, **kwargs) -> None:
        enabled = kwargs.get("enabled", True)
        expert  = kwargs.get("expert", "AUTO")

        if path == "":
            path = filedialog.askopenfilename(
                title="Select LoRA",
                filetypes=[("Safetensors", "*.safetensors"), ("All", "*.*")],
            )
            if not path:
                return

        row_frame   = ttk.Frame(self._lora_inner)
        weight_var  = tk.DoubleVar(value=weight)
        expert_var  = tk.StringVar(value=expert)
        enabled_var = tk.BooleanVar(value=enabled)

        row = {
            "path": path, "weight_var": weight_var,
            "expert_var": expert_var, "enabled_var": enabled_var,
            "frame": row_frame,
        }

        short = os.path.basename(path)
        lbl = ttk.Label(row_frame, text=short, width=38, anchor="w")
        lbl.pack(side="left", padx=2)
        lbl.bind("<Double-Button-1>", lambda e, p=path: messagebox.showinfo("Full path", p))

        _w_entry = ttk.Entry(row_frame, textvariable=weight_var, width=6)
        _w_entry.pack(side="left", padx=2)
        Tooltip(_w_entry, "LoRA weight multiplier (1.0 = full strength).")
        _expert_cb = ttk.Combobox(
            row_frame, textvariable=expert_var,
            values=["AUTO", "HIGH", "LOW", "BOTH"], state="readonly", width=6,
        )
        _expert_cb.pack(side="left", padx=2)
        Tooltip(_expert_cb,
                "Which Wan2.2 expert to apply this LoRA to.\n\n"
                "AUTO — infer from filename (high/low/highnoise/lownoise).\n"
                "HIGH — apply to the HIGH-noise expert (transformer 1).\n"
                "LOW  — apply to the LOW-noise expert (transformer 2).\n"
                "BOTH — apply to both experts with the same weights.")
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
                "expert":  row["expert_var"].get(),
                "enabled": row["enabled_var"].get(),
            }
            for row in self._lora_rows
        ]

    def _load_loras_from_config(self) -> None:
        for entry in self.cfg.get("loras", []):
            self._add_lora(
                path=entry.get("path", ""),
                weight=entry.get("weight", 1.0),
                expert=entry.get("expert", "AUTO"),
                enabled=entry.get("enabled", True),
            )

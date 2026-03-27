"""ChromaBackend — Chroma (CHROMA_1) sampler backend.

Implements BaseSamplerBackend.  load_model() reads all settings from the cfg
dict so the GUI can pass a plain snapshot without keyword explosion.
"""
import hashlib
import os
from contextlib import nullcontext

import torch
import sampler_core  # noqa: F401 — inject OT path

from sampler_core.backend.base import BaseSamplerBackend, Cancelled
from sampler_core.util.dtype_maps import (
    DTYPE_MAP, COMPUTE_TORCH_DTYPE, COMPUTE_DATATYPE,
    COMPUTE_DTYPE_OVERRIDE, SVD_DTYPE_MAP,
)
from sampler_core.util.gguf_util import gguf_type_summary
from sampler_core.util.ot_bridge import find_ot_quant_cache
from sampler_core.util.png_meta import write_png_metadata
from sampler_core.util.tokenizer_patch import patch_tokenizer_no_truncate
from sampler_core.util.resolution import ATTN_BACKEND_ENUM_NAME, check_attn_backends
from sampler_core.lora.hooks import apply_lora_hooks
from chroma.lora_keys import make_chroma_translator, expand_lora_unet_fused, expand_diffusion_model_fused

from modules.model.ChromaModel import ChromaModel
from modules.modelLoader.chroma.ChromaModelLoader import ChromaModelLoader
from modules.modelSampler.ChromaSampler import ChromaSampler
from modules.util.checkpointing_util import enable_checkpointing_for_chroma_transformer
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.ModelType import ModelType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import quantize_layers as _ot_quantize_layers


def _te_cache_key(pos_prompt: str, neg_prompt: str, te_dtype: str) -> str:
    """
    Stable 24-char hex key for one (pos, neg, dtype) text-encoder cache entry.
    Includes te_dtype so BF16 and INT8 encodings don't collide.
    """
    data = f"{pos_prompt}\x00{neg_prompt}\x00{te_dtype}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:24]


class _OffloadConfig:
    """Minimal duck-type for OT's LayerOffloadConductor constructor."""
    def __init__(self, train_device, temp_device, fraction, use_compile=False):
        self.train_device                = str(train_device)
        self.temp_device                 = str(temp_device)
        self.layer_offload_fraction      = float(fraction)
        self.gradient_checkpointing      = GradientCheckpointingMethod.CPU_OFFLOADED
        self.enable_activation_offloading = False
        self.enable_async_offloading     = True
        self.compile                     = use_compile


class ChromaBackend(BaseSamplerBackend):
    """Sampler backend for Chroma (CHROMA_1)."""

    MODEL_IDENTITY_KEYS = (
        "base_model", "weight_dtype", "text_enc_dtype",
        "svd_enabled", "svd_rank", "svd_dtype", "transformer_gguf",
        "quant_cache_dir", "use_compile", "compute_dtype",
        "fast_fp16_accum", "offload_enabled", "offload_fraction",
    )

    # ------------------------------------------------------------------
    def _ensure_blocks_compiled(self):
        """Compile transformer blocks if deferred compilation was requested.

        Called AFTER LoRA injection so dynamo's initial trace includes the
        patched forwards.  Uses module.compile() (in-place) rather than
        torch.compile() (OptimizedModule wrapper) so the block type is
        preserved — critical for the offload path where _kwargs_to_args
        introspects the block's forward signature.

        fullgraph selection:
          - GGUF: fullgraph=False — dequant is a torch.compiler.disable
            graph break, incompatible with fullgraph=True.
          - Everything else (NF4, INT8, FP8, W8A8*, BF16, FP16, FP32):
            fullgraph=True — matches OneTrainer; traces INTO custom
            autograd Functions so dynamo sees only standard torch ops.
        """
        if not getattr(self, '_compile_deferred', False):
            return
        if self.model is None:
            return

        cfg = self._loaded_cfg or {}
        is_gguf = cfg.get("weight_dtype", "") in {"GGUF", "GGUF_A8I", "GGUF_A8F"}
        fg = not is_gguf

        for blk_list in (self.model.transformer.transformer_blocks,
                         self.model.transformer.single_transformer_blocks):
            for i in range(len(blk_list)):
                block = blk_list[i]

                # Offload path: compile the INNER block, not the wrapper.
                if hasattr(block, 'checkpoint') and \
                        isinstance(block.checkpoint, torch.nn.Module):
                    if hasattr(block.checkpoint, '_compiled_call_impl'):
                        continue  # already compiled; cache was reset by remove_loras()
                    block.checkpoint.compile(fullgraph=fg)
                    continue

                # No-offload path: compile the block directly (in-place).
                if hasattr(block, '_compiled_call_impl'):
                    continue
                block.compile(fullgraph=fg)

    # ------------------------------------------------------------------
    def load_model(self, cfg: dict, on_status) -> None:
        self._compile_deferred = False  # reset; set below if use_compile
        # Unload any currently loaded model first so we never hold two copies
        # in RAM simultaneously (old model must be released before new one loads).
        if self.model is not None:
            self.unload_model(on_status)

        weight_dtype_str      = cfg.get("weight_dtype", "NF4")
        te_dtype_str          = cfg.get("text_enc_dtype", "BF16")
        svd_enabled           = cfg.get("svd_enabled", False)
        svd_rank              = int(cfg.get("svd_rank", 16))
        svd_dtype_str         = cfg.get("svd_dtype", "BF16")
        quant_cache_dir       = cfg.get("quant_cache_dir", "") or ""
        use_compile           = cfg.get("use_compile", False)
        offload_enabled       = cfg.get("offload_enabled", False)
        offload_fraction      = cfg.get("offload_fraction", 50)
        transformer_gguf      = cfg.get("transformer_gguf", "") or ""
        compute_dtype_override = cfg.get("compute_dtype", "Auto")
        fast_fp16_accum       = cfg.get("fast_fp16_accum", False)
        base_model_path       = cfg.get("base_model", "")

        on_status("Loading model …")

        if weight_dtype_str == "GGUF":
            if not os.path.isfile(transformer_gguf):
                raise ValueError("Transformer GGUF path not set or file not found")
            if svd_enabled:
                raise ValueError("SVDQuant is incompatible with GGUF — uncheck SVDQuant first")
            summary = gguf_type_summary(transformer_gguf)
            on_status(f"GGUF: {os.path.basename(transformer_gguf)}  [{summary}]")

        if use_compile:
            from modules.util.compile_util import init_compile
            init_compile()

            # ---- inductor tuning for faster compilation ------------------
            # fx_graph_cache: persist fully-compiled Triton binaries to disk
            # so the SECOND run with the same model+LoRA+resolution skips
            # all 2000+ Triton kernel compilations (~seconds vs ~8 minutes).
            # max_autotune: when False, skip ~1000 GPU benchmarks that pick
            # optimal GEMM tile sizes.  Uses sensible defaults instead —
            # ~5-10% slower kernels but saves ~2 minutes on cold compile.
            try:
                import torch._inductor.config as _ind_cfg
                _ind_cfg.fx_graph_cache = True
                _ind_cfg.max_autotune = False
                _ind_cfg.max_autotune_gemm = False
            except (ImportError, AttributeError):
                pass  # older PyTorch — inductor config not available
            # --------------------------------------------------------------

        weight_dtype  = DTYPE_MAP.get(weight_dtype_str, DataType.NFLOAT_4)
        te_dtype      = DTYPE_MAP.get(te_dtype_str, DataType.BFLOAT_16)
        compute_dtype = COMPUTE_DATATYPE.get(weight_dtype_str, DataType.BFLOAT_16)
        torch_compute = COMPUTE_TORCH_DTYPE.get(weight_dtype_str, torch.bfloat16)

        if compute_dtype_override in COMPUTE_DTYPE_OVERRIDE:
            torch_compute, compute_dtype = COMPUTE_DTYPE_OVERRIDE[compute_dtype_override]
            if compute_dtype_override == "FP16" and weight_dtype_str not in ("FP16",):
                on_status(
                    "⚠ WARNING: FP16 compute with non-FP16 weights — "
                    "norm weights stay BF16, expect dtype mismatches. "
                    "Use FP16 weight dtype or set Compute to Auto."
                )

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = fast_fp16_accum
        if fast_fp16_accum:
            on_status("Fast FP16 accumulation enabled")

        weight_dtypes = ModelWeightDtypes(
            train_dtype=compute_dtype,
            fallback_train_dtype=DataType.BFLOAT_16,
            unet=DataType.NONE,
            prior=DataType.NONE,
            transformer=weight_dtype,
            text_encoder=te_dtype,
            text_encoder_2=DataType.NONE,
            text_encoder_3=DataType.NONE,
            text_encoder_4=DataType.NONE,
            vae=DataType.FLOAT_32,
            effnet_encoder=DataType.NONE,
            decoder=DataType.NONE,
            decoder_text_encoder=DataType.NONE,
            decoder_vqgan=DataType.NONE,
            lora=DataType.NONE,
            embedding=DataType.NONE,
        )

        from modules.util.config.TrainConfig import QuantizationConfig
        quantization = QuantizationConfig.default_values()
        if svd_enabled:
            quantization.svd_dtype = SVD_DTYPE_MAP.get(svd_dtype_str, DataType.BFLOAT_16)
            quantization.svd_rank  = svd_rank
        resolved_cache = quant_cache_dir.strip() if quant_cache_dir.strip() else find_ot_quant_cache()
        if resolved_cache:
            os.makedirs(resolved_cache, exist_ok=True)
            quantization.cache_dir = resolved_cache
            on_status(f"Quant cache: {resolved_cache}")

        is_gguf = weight_dtype_str in {"GGUF", "GGUF_A8I", "GGUF_A8F"}
        model_names = ModelNames(
            base_model=base_model_path,
            transformer_model=transformer_gguf if is_gguf else "",
        )

        model  = ChromaModel(ModelType.CHROMA_1)
        loader = ChromaModelLoader()
        loader.load(model, ModelType.CHROMA_1, model_names, weight_dtypes, quantization)

        # For GGUF_A8I/F: the loader already replaced GGUFLinear → LinearGGUFA8 internally.
        # quantize_layers sets compute_dtype on those layers (left None by the factory).
        # It does NOT call .quantize() for non-QuantizedModuleMixin modules.
        if weight_dtype_str in {"GGUF_A8I", "GGUF_A8F"}:
            _ot_quantize_layers(model.transformer, self.train_device, compute_dtype, None)

        # Quantize transformer layer-by-layer with status updates
        if weight_dtype.is_quantized():
            from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin as _QMM
            quant_layers = [(n, m) for n, m in model.transformer.named_modules()
                            if isinstance(m, _QMM)]
            total_q = len(quant_layers)
            for idx, (name, child) in enumerate(quant_layers):
                child.compute_dtype = compute_dtype.torch_dtype()
                label = name.rsplit(".", 1)[-1] if "." in name else name
                if svd_enabled:
                    on_status(f"SVDQuant {idx + 1}/{total_q}: {label} …")
                elif idx % 20 == 0:
                    on_status(f"Quantizing transformer {idx + 1}/{total_q} …")
                child.quantize(device=self.train_device)
            on_status(f"Transformer quantization done ({total_q} layers).")
        if te_dtype.is_quantized():
            on_status("Finalizing quantized text encoder weights …")
            _ot_quantize_layers(model.text_encoder, self.train_device, compute_dtype, None)

        fraction = max(0.0, min(0.99, offload_fraction / 100.0)) if offload_enabled else 0.0
        if fraction > 0:
            # Offload enabled: pass use_compile so OT's create_checkpoint()
            # returns proper OffloadCheckpointLayer wrappers with .checkpoint
            # pointing to the inner block.  OT also calls
            # orig_module.compile(fullgraph=True) which we strip immediately
            # below — we recompile after LoRA with the correct fullgraph
            # setting for the weight type.
            on_status(f"Setting up layer offload ({fraction:.0%}) …")
            oc = _OffloadConfig(self.train_device, self.temp_device, fraction,
                                use_compile=use_compile)
            model.transformer_offload_conductor = \
                enable_checkpointing_for_chroma_transformer(model.transformer, oc)
            if use_compile:
                # Strip OT's premature fullgraph=True compile so we can
                # recompile after LoRA injection with the right settings.
                for blk_list in (model.transformer.transformer_blocks,
                                 model.transformer.single_transformer_blocks):
                    for blk in blk_list:
                        if hasattr(blk, 'checkpoint') and \
                                hasattr(blk.checkpoint, '_compiled_call_impl'):
                            del blk.checkpoint._compiled_call_impl
                self._compile_deferred = True
        elif use_compile:
            # No offload: defer compilation to sample() so it happens AFTER
            # LoRA injection.  Compiling here (before LoRA) causes per-step
            # recompilation: dynamo guards on child module .forward identity
            # at trace time, and LoRA patches change those identities, so
            # guards fail every step and dynamo retraces until it hits the
            # 256-recompile limit and falls back to full eager.
            # By compiling after LoRA, dynamo's initial trace includes the
            # patched forwards and guards are stable between diffusion steps.
            on_status("torch.compile enabled (deferred to first sample)")
            self._compile_deferred = True

        model.autocast_context             = torch.autocast("cuda", dtype=torch_compute)
        model.text_encoder_autocast_context = torch.autocast("cuda", dtype=torch_compute)
        model.train_dtype                  = compute_dtype

        model.eval()
        model.vae_to(self.temp_device)
        model.text_encoder_to(self.temp_device)
        try:
            model.transformer_to(self.temp_device)
        except RuntimeError as exc:
            if "meta tensor" not in str(exc).lower():
                raise

        self.model = model
        self._loaded_cfg = cfg
        patch_tokenizer_no_truncate(model)

        dtype_label  = f"GGUF [{os.path.basename(transformer_gguf)}]" if is_gguf else weight_dtype_str
        offload_str  = f"offload {fraction:.0%}" if fraction > 0 else "offload off"
        compile_note = "  compile:deferred" if use_compile else ""
        on_status(f"Loaded [{dtype_label}]  |  {offload_str}{compile_note}")

    # ------------------------------------------------------------------
    def _inject_lora(self, state_dict: dict, weight: float, entry: dict) -> list:
        state_dict = expand_lora_unet_fused(state_dict)
        state_dict = expand_diffusion_model_fused(state_dict)
        translator = make_chroma_translator(self.model.transformer, self.model.text_encoder)
        on_log = getattr(self, "_on_log", None)
        
        compile_friendly = getattr(self, '_compile_deferred', False)
            
        return apply_lora_hooks(
            self.model.transformer,
            self.model.text_encoder,
            state_dict,
            weight,
            translator,
            on_log=on_log,
            hint_device=self.train_device,
            compile_friendly=compile_friendly,
        )

    # ------------------------------------------------------------------
    def sample(self, cfg: dict, on_progress, on_done, on_error) -> None:
        if self.model is None:
            on_error("Model not loaded")
            return

        attn_str = cfg.get("attn_backend", "Auto")
        if attn_str != "Auto":
            avail = check_attn_backends()
            if not avail.get(attn_str, False):
                _install = {
                    "Flash": (
                        "pip install <wheel> from:\n"
                        "https://github.com/zzlol63/flash-attention-prebuild-wheels/releases"
                    ),
                    "SageAttn": (
                        "pip install triton-windows, then wheel from:\n"
                        "https://github.com/woct0rdho/SageAttention/releases"
                    ),
                }
                on_error(f"{attn_str} is not installed.\n{_install.get(attn_str, '')}")
                return

        self._cancel_event.clear()

        import datetime
        import random as _random
        os.makedirs(cfg["output_dir"], exist_ok=True)
        seed = cfg["seed"]
        if cfg.get("random_seed"):
            seed = _random.randint(0, 2 ** 31 - 1)
        ts       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_base = os.path.join(cfg["output_dir"], f"{ts}-sample-{seed}")

        try:
            sample_config = SampleConfig.default_values(ModelType.CHROMA_1)
        except TypeError:
            sample_config = SampleConfig.default_values()
        sample_config.prompt           = cfg.get("prompt", "")
        sample_config.negative_prompt  = cfg.get("negative_prompt", "")
        sample_config.width            = int(cfg.get("width", 1024))
        sample_config.height           = int(cfg.get("height", 1024))
        sample_config.cfg_scale        = float(cfg.get("cfg_scale", 3.5))
        sample_config.seed             = seed
        sample_config.random_seed      = False
        sample_config.diffusion_steps  = int(cfg.get("steps", 30))

        scheduler_type = cfg.get("scheduler", "Euler")
        sigma_shift    = float(cfg.get("sigma_shift", 3.0))

        cancel_event  = self._cancel_event
        _heun_sched   = (scheduler_type == "Heun")

        def _progress(step, total):
            if cancel_event.is_set():
                raise Cancelled()
            if _heun_sched:
                # Heun exposes 2*steps-1 NFE ticks; map back to user-facing steps
                on_progress((step + 1) // 2, (total + 1) // 2)
            else:
                on_progress(step, total)

        try:
            _enum_name = ATTN_BACKEND_ENUM_NAME.get(attn_str)
            from diffusers.models.attention_dispatch import (
                attention_backend as _attn_ctx,
                AttentionBackendName as _AttnName,
            )
            _ctx = (
                _attn_ctx(getattr(_AttnName, _enum_name))
                if _enum_name is not None
                else nullcontext()
            )

            # ---- scheduler swap ------------------------------------------
            _orig_scheduler = self.model.noise_scheduler
            _base_cfg = dict(_orig_scheduler.config)
            _model_shift = float(_base_cfg.get("shift", 3.0))
            _use_shift   = sigma_shift if abs(sigma_shift - _model_shift) > 1e-4 else _model_shift

            if scheduler_type == "Heun":
                from diffusers import FlowMatchHeunDiscreteScheduler
                self.model.noise_scheduler = FlowMatchHeunDiscreteScheduler(
                    num_train_timesteps=_base_cfg.get("num_train_timesteps", 1000),
                    shift=_use_shift,
                )
            elif abs(_use_shift - _model_shift) > 1e-4:
                # Euler with overridden shift — rebuild from existing config
                from diffusers import FlowMatchEulerDiscreteScheduler
                self.model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                    {**_base_cfg, "shift": _use_shift}
                )
            # else: Euler + same shift — keep model scheduler as-is
            # --------------------------------------------------------------

            # ---- deferred compile ----------------------------------------
            # Compile inner blocks AFTER LoRA injection so dynamo's trace
            # captures the patched forwards.  With compile-after-LoRA and
            # method-patching (not hooks), the compiled graph includes LoRA
            # — no need to swap to eager.
            self._ensure_blocks_compiled()
            # --------------------------------------------------------------

            sampler = ChromaSampler(
                self.train_device, self.temp_device,
                self.model, ModelType.CHROMA_1,
            )

            _transformer      = self.model.transformer
            _needs_mask_patch = attn_str in ("Flash", "SageAttn")
            if _needs_mask_patch:
                _orig_tr_fwd = _transformer.forward
                def _fwd_no_attn_mask(*args, **kwargs):
                    kwargs.pop("attention_mask", None)
                    return _orig_tr_fwd(*args, **kwargs)
                _transformer.forward = _fwd_no_attn_mask

            # ---- text-encoder embedding cache ----------------------------
            # Patch model.encode_text (not text_encoder.forward) so we cache
            # the two final tensors it returns — (embedding, mask) — rather
            # than the intermediate BaseModelOutput whose hidden_states tuple
            # contains CPU tensors that cause a device mismatch on replay.
            _et_patched    = False   # True if we monkey-patched encode_text
            _et_save_file  = None   # cache file path on miss (save after success)
            _et_captured   = [None] # [0] filled by miss wrapper
            _te_to_orig    = None   # original text_encoder_to (patched on cache hit)

            if cfg.get("text_cache_enabled", False):
                _cache_dir = (cfg.get("quant_cache_dir", "") or "").strip()
                if not _cache_dir:
                    _cache_dir = find_ot_quant_cache()
                if not _cache_dir:
                    _cache_dir = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)), "output")
                _te_cache_dir  = os.path.join(_cache_dir, "te_cache")
                _te_key        = _te_cache_key(
                    cfg.get("prompt", ""),
                    cfg.get("negative_prompt", ""),
                    cfg.get("text_enc_dtype", "BF16"),
                )
                _te_cache_file = os.path.join(_te_cache_dir, f"te_{_te_key}.pt")
                _train_dev     = self.train_device

                if os.path.isfile(_te_cache_file):
                    # Cache hit — return the two saved tensors, skip TE entirely.
                    # Also no-op text_encoder_to so T5 is never moved to GPU VRAM.
                    try:
                        _cached = torch.load(
                            _te_cache_file, map_location="cpu", weights_only=True)
                        if not isinstance(_cached, dict) or "embedding" not in _cached:
                            raise ValueError("stale cache format")
                        _emb_cpu  = _cached["embedding"]
                        _mask_cpu = _cached["mask"]

                        _et_orig    = self.model.encode_text
                        _te_to_orig = self.model.text_encoder_to
                        def _et_hit(*args, **kwargs):
                            dev = kwargs.get("train_device", _train_dev)
                            return (_emb_cpu.to(dev), _mask_cpu.to(dev))
                        self.model.encode_text      = _et_hit
                        self.model.text_encoder_to  = lambda *a, **kw: None
                        _et_patched = True
                        print(f"[T5 cache] hit  {_te_key}  (T5 stays off-GPU)")
                    except Exception as _cache_exc:
                        print(f"[T5 cache] load failed ({_cache_exc}) — re-encoding")
                        try:
                            os.remove(_te_cache_file)
                        except OSError:
                            pass

                else:
                    # Cache miss — wrap encode_text to capture outputs, save after success
                    os.makedirs(_te_cache_dir, exist_ok=True)
                    _et_orig = self.model.encode_text
                    def _et_miss(*args, **kwargs):
                        result = _et_orig(*args, **kwargs)
                        _et_captured[0] = (result[0].detach().cpu(),
                                           result[1].detach().cpu())
                        return result
                    self.model.encode_text = _et_miss
                    _et_patched   = True
                    _et_save_file = _te_cache_file
                    print(f"[T5 cache] miss {_te_key} — encoding and saving")
            # --------------------------------------------------------------

            # Suppress ChromaSampler's tqdm console output — we have the GUI bar.
            import modules.modelSampler.ChromaSampler as _cs_mod
            import tqdm as _tqdm_lib
            _orig_cs_tqdm = _cs_mod.tqdm
            _cs_mod.tqdm = lambda *a, **kw: _tqdm_lib.tqdm(*a, **{**kw, "disable": True})

            try:
                with _ctx:
                    sampler.sample(
                        sample_config=sample_config,
                        destination=out_base,
                        image_format=ImageFormat.PNG,
                        on_update_progress=_progress,
                    )
            finally:
                _cs_mod.tqdm = _orig_cs_tqdm
                self.model.noise_scheduler = _orig_scheduler
                if _needs_mask_patch:
                    _transformer.forward = _orig_tr_fwd
                if _et_patched:
                    self.model.encode_text = _et_orig
                if _te_to_orig is not None:
                    self.model.text_encoder_to = _te_to_orig

            # Persist captured embeddings (only reached on clean success)
            if _et_save_file and _et_captured[0] is not None:
                _emb, _mask = _et_captured[0]
                torch.save({"embedding": _emb, "mask": _mask}, _et_save_file)
                print(f"[T5 cache] saved → {os.path.basename(_et_save_file)}")

            out_path = out_base + ImageFormat.PNG.extension()
            lora_entries = [
                {"path": e.get("path", ""), "weight": e.get("weight", 1.0)}
                for e in cfg.get("loras", [])
                if e.get("enabled", True) and e.get("path", "").strip()
            ]
            meta = {
                "model":            cfg.get("base_model", ""),
                "transformer_gguf": cfg.get("transformer_gguf") or None,
                "weight_dtype":     cfg.get("weight_dtype", ""),
                "text_enc_dtype":   cfg.get("text_enc_dtype", ""),
                "compute_dtype":    cfg.get("compute_dtype", "Auto"),
                "svd_enabled":      cfg.get("svd_enabled", False),
                "svd_rank":         cfg.get("svd_rank", 16) if cfg.get("svd_enabled") else None,
                "svd_dtype":        cfg.get("svd_dtype", "BF16") if cfg.get("svd_enabled") else None,
                "prompt":           cfg.get("prompt", ""),
                "negative_prompt":  cfg.get("negative_prompt", ""),
                "seed":             seed,
                "scheduler":        scheduler_type,
                "sigma_shift":      _use_shift,
                "steps":            cfg.get("steps", 30),
                "cfg_scale":        cfg.get("cfg_scale", 3.5),
                "width":            cfg.get("width", 1024),
                "height":           cfg.get("height", 1024),
                "pixel_target":     cfg.get("pixel_target", 1024),
                "aspect_ratio":     cfg.get("aspect_ratio", "1:1"),
                "attn_backend":     attn_str,
                "loras":            lora_entries,
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            write_png_metadata(out_path, "chroma_sampler", meta)
            on_done(out_path, seed)
        except Cancelled:
            on_done(None, None)
        except Exception as exc:
            import traceback as _tb
            full = _tb.format_exc()
            on_error(str(exc) + "\n\n" + full)

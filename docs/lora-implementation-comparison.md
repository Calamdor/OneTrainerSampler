# LoRA Implementation Comparison: OneTrainer / ComfyUI / city96 / Kijai

## Overview

Four implementations, each with different tradeoffs for LoRA application
during inference with quantized models and offloading.

---

## 1. OneTrainer (Training Framework)

**Pattern**: Persistent `nn.Module` LoRA layers with forward hooking.

**Flow**:
```
safetensors → convert_to_diffusers() → model.lora_state_dict
    → LoRAModuleWrapper creates LoRAModule per layer
    → lora_down = nn.Linear(in_dim, rank), lora_up = nn.Linear(rank, out_dim)
    → load_state_dict() populates weights
    → hook_to_module() replaces orig_module.forward with LoRA forward
```

**Forward (LoRAModule.forward)**:
```python
ld = self.lora_up(self.dropout(self.lora_down(x)))
return self.orig_forward(x) + ld * (self.alpha / self.rank)
```

**Quantized (SVDQuant + W8A8)**:
```python
# forward_with_lora merges LoRA and SVD projections before matmul:
down_merged = cat([lora_down.weight, svd_down], dim=0)
up_merged = cat([lora_up_scaled, svd_up], dim=1)
return linear(linear(x, down_merged), up_merged) + quantized_base(x)
```

**Key Properties**:
- LoRA modules are proper nn.Module → participate in .to(), state_dict, etc.
- Only ONE LoRA at a time (training context)
- hook_to_module replaces forward → traceable by dynamo
- Compile: fullgraph=True works (traces through LoRA forward)
- Device: LoRA weights move with transformer_lora.to(device)

---

## 2. ComfyUI Core (Model Patcher)

**Pattern**: Patch dictionary with lazy or eager application.

**Flow**:
```
safetensors → load_lora() → patch_dict with WeightAdapterBase instances
    → model.add_patches(patch_dict, strength) → self.patches[key].append(...)
    → Multiple LoRAs: list grows per key
```

**Merge mode (patch_weight_to_device)**:
```python
# Backup original weight
self.backup[key] = weight.clone()
# Apply ALL patches in sequence
temp_weight = weight.to(compute_dtype)
out_weight = calculate_weight(self.patches[key], temp_weight, key)
# Write back permanently
copy_to_param(self.model, key, out_weight)
```

**LowVRAM mode (runtime)**:
```python
# Wrap patches in callable, defer to forward time
m.weight_function = [LowVramPatch(key, self.patches)]
# During forward (ops.py cast_bias_weight):
weight = s.weight.to(device)    # move to GPU
for f in s.weight_function:
    weight = f(weight)           # apply LoRA on-the-fly
x = F.linear(input, weight)     # compute
# weight freed after use
```

**LoRA delta computation (WeightAdapterBase.calculate_weight)**:
```python
delta = torch.mm(up.flatten(1), down.flatten(1)).reshape(weight.shape)
weight += (strength * alpha / rank) * delta
```

**Key Properties**:
- Patches stored on CPU, moved to GPU on demand
- Multiple LoRAs: sequential application in calculate_weight loop
- Pinned memory + non_blocking transfers + async streams
- No torch.compile interaction (weight loading is @torch_compiler_disable)
- Merge mode: one-time cost, zero per-step overhead
- LowVRAM mode: per-layer overhead but minimal VRAM

---

## 3. city96 ComfyUI-GGUF

**Pattern**: Extends ComfyUI's patcher for GGUF quantized tensors.

**Flow**:
```
GGUF file → GGMLTensor (stores quant_type + tensor_shape + patches)
    → Patches attached to the tensor itself: tensor.patches = [...]
    → During forward: get_weight() dequantizes + applies patches
```

**Forward (GGMLLayer.get_weight)**:
```python
# Move patches to GPU async
patch_list = move_patch_to_device(tensor.patches, device)
# Dequantize GGUF tensor
weight = dequantize_tensor(tensor, dtype)
# Apply LoRA patches
weight = comfy.lora.calculate_weight(patch_list, weight, key)
return weight
```

**cast_bias_weight is @torch_compiler_disable()**:
- Weight loading + dequant + LoRA patching ALL happen outside compiled graph
- Only the matmul itself is compiled
- Patches moved with non_blocking=True

**Key Properties**:
- Patches on CPU, dequant on GPU, LoRA applied to dequantized float
- No re-quantization (uses dequantized weight directly)
- @torch_compiler_disable avoids dynamo tracing the dequant+patch
- Per-layer device movement with non_blocking=True

---

## 4. Kijai ComfyUI-WanVideoWrapper

**Pattern**: CustomLinear modules with registered buffers, dual merge/unmerge paths.

**Merged path (default)**:
```python
# Uses ComfyUI's apply_lora → patch_weight_to_device → calculate_weight
# ONE pass: dequant + apply all LoRAs + write back
# Zero per-step overhead, zero LoRA VRAM after merge
patcher.patches.clear()  # LoRA data freed after merge
```

**Unmerged path (CustomLinear)**:
```python
class CustomLinear(nn.Linear):
    def __init__(self, ...):
        self.lora_diffs = []       # registered buffers (move with .to())
        self.lora_strengths = []

    def forward(self, input):
        weight = self._prepare_weight(input)  # dequant if GGUF
        weight = self._get_weight_with_lora(weight)  # apply LoRAs
        return F.linear(input, weight)
```

**LoRA application (custom_op for torch.compile compat)**:
```python
@torch.library.custom_op("wanvideo::apply_lora", mutates_args=())
def apply_lora(weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
    patch_diff = torch.mm(lora_diff_0.flatten(1), lora_diff_1.flatten(1)).reshape(weight.shape)
    alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
    return weight + patch_diff * (lora_strength * alpha)
```

**Key Properties**:
- Merged = default for non-GGUF (zero per-step cost)
- FP8: merge REQUIRED ("FP8 matmul with unmerged LoRAs is not supported")
- GGUF: unmerged only (dequant per-forward, no re-quantization)
- LoRA factors as registered buffers → move with block swapping
- @torch.library.custom_op for torch.compile compatibility
- Multiple LoRAs: sequential in _get_weight_with_lora loop

---

## Comparison Matrix

| Aspect | OneTrainer | ComfyUI Core | city96 GGUF | Kijai Wan |
|--------|-----------|-------------|------------|----------|
| **LoRA storage** | nn.Module params | Patch dict (CPU) | On tensor.patches | Registered buffers |
| **Multi-LoRA** | 1 only (training) | Sequential in list | Sequential | Sequential |
| **Merge into weight** | No (hook only) | Yes (merge mode) | No (dequant+patch) | Yes (default) |
| **Per-step cost** | 2 F.linear per layer | 0 (merged) or calc_weight | dequant+patch | 0 (merged) or apply_lora |
| **VRAM for LoRA** | Moves with model | 0 (merged) or CPU | CPU (on-demand) | 0 (merged) or buffers |
| **torch.compile** | fullgraph=True | N/A (merge) | @compiler_disable | @custom_op |
| **Quantized merge** | No (additive only) | Yes | No (stays float) | Yes (FP8 required) |
| **Offload aware** | Via transformer_lora.to() | Via patcher | non_blocking .to() | Via block swapping |
| **Apply speed** | Fast (hook only) | One pass per key | Per-forward | One pass (merge) |

---

## Lessons for OneTrainerSampler

### What to take from each:

**From OneTrainer**:
- Use OT's infrastructure (model loading, quantization, offloading)
- fullgraph=True compilation strategy
- Forward hooking pattern (method patching)

**From ComfyUI Core**:
- Merge mode as default (zero per-step overhead)
- Backup/restore pattern for weight merging
- Pinned memory + non_blocking for transfers
- offload_quantized integration for device management

**From city96 GGUF**:
- GGUF: dequant + patch per-forward (no re-quantization)
- @torch_compiler_disable for weight loading (avoids dynamo overhead)
- non_blocking .to() for async transfers
- Patches stored on CPU, moved on demand

**From Kijai WanVideoWrapper**:
- Merge LoRA into quantized weights as DEFAULT strategy
- FP8: merge REQUIRED (don't even offer unmerged)
- @torch.library.custom_op for compile-compatible LoRA ops
- Registered buffers for block-swapping compatibility
- Batch all LoRAs, apply in one pass per module

### Current OneTrainerSampler dispatch:
1. Float merge (BF16/FP16/FP32) — direct weight add ✓
2. Quantized merge (W8A8I/W8A8F/FP8) — dequant+add+requant ✓
3. GGUF compile-friendly — dequant+merge per-forward ✓
4. Forward-patch fallback — closure-based (compile off) ✓

### Remaining optimization:
- **Batch the quantized merge**: currently dequant→merge→requant per LoRA.
  Should be: dequant once → merge ALL LoRAs → requant once.
- **Consider @torch.library.custom_op** for the GGUF forward LoRA merge
  (avoids fullgraph issues with device movement).

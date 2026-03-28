"""Shared dtype option lists and lookup maps used by all sampler backends."""
import sampler_core  # noqa: F401 — ensures OT is on sys.path

import torch
from modules.util.enum.DataType import DataType

DTYPE_MAP = {
    "NF4":      DataType.NFLOAT_4,
    "FP8":      DataType.FLOAT_8,
    "W8A8F":    DataType.FLOAT_W8A8,
    "W8A8I":    DataType.INT_W8A8,
    "BF16":     DataType.BFLOAT_16,
    "FP16":     DataType.FLOAT_16,
    "FP32":     DataType.FLOAT_32,
    "GGUF":     DataType.GGUF,
    "GGUF_A8I": DataType.GGUF_A8_INT,
    "GGUF_A8F": DataType.GGUF_A8_FLOAT,
}

COMPUTE_TORCH_DTYPE = {
    "NF4":      torch.bfloat16,
    "FP8":      torch.bfloat16,
    "W8A8F":    torch.bfloat16,
    "W8A8I":    torch.bfloat16,
    "BF16":     torch.bfloat16,
    "FP16":     torch.float16,
    "FP32":     torch.float32,
    "GGUF":     torch.bfloat16,
    "GGUF_A8I": torch.bfloat16,
    "GGUF_A8F": torch.bfloat16,
}

COMPUTE_DATATYPE = {
    "NF4":      DataType.BFLOAT_16,
    "FP8":      DataType.BFLOAT_16,
    "W8A8F":    DataType.BFLOAT_16,
    "W8A8I":    DataType.BFLOAT_16,
    "BF16":     DataType.BFLOAT_16,
    "FP16":     DataType.FLOAT_16,
    "FP32":     DataType.FLOAT_32,
    "GGUF":     DataType.BFLOAT_16,
    "GGUF_A8I": DataType.BFLOAT_16,
    "GGUF_A8F": DataType.BFLOAT_16,
}

COMPUTE_DTYPE_OPTIONS  = ["Auto", "BF16", "FP16"]
COMPUTE_DTYPE_OVERRIDE = {
    "BF16": (torch.bfloat16, DataType.BFLOAT_16),
    "FP16": (torch.float16,  DataType.FLOAT_16),
}

TRANSFORMER_DTYPE_OPTIONS = ["NF4", "FP8", "W8A8F", "W8A8I", "BF16", "FP16", "FP32", "GGUF", "GGUF_A8I", "GGUF_A8F"]
TEXT_ENC_DTYPE_OPTIONS    = ["BF16", "FP16", "FP32", "NF4"]

# SVD residual dtype — BF16/FP32 only (OT UI excludes FP16 for Chroma).
# Models that allow FP16 (e.g. Wan) define their own _SVD_DTYPE_OPTIONS locally.
SVD_DTYPE_OPTIONS = ["BF16", "FP32"]
SVD_DTYPE_MAP     = {
    "BF16": DataType.BFLOAT_16,
    "FP16": DataType.FLOAT_16,
    "FP32": DataType.FLOAT_32,
}

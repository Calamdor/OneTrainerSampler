"""
sampler_core — shared infrastructure for OT-based standalone samplers.

Importing this package (or any sub-module) injects the adjacent OneTrainer
directory into sys.path so that OT modules are importable.
"""
import os
import sys

# sampler_core/ lives inside OneTrainerSamplers/
# OneTrainer/ is adjacent to OneTrainerSamplers/ under the same parent
SAMPLERS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OT_DIR = os.path.join(os.path.dirname(SAMPLERS_DIR), "OneTrainer")

if OT_DIR not in sys.path:
    sys.path.insert(0, OT_DIR)

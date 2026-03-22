"""Helpers for reading OneTrainer's live config and locating OT directories."""
import json
import os

from sampler_core import OT_DIR


def read_ot_config() -> dict:
    """Load OT's config.json, returning {} on failure."""
    try:
        with open(os.path.join(OT_DIR, "config.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_ot_workspace() -> str:
    """Return absolute path of OT's workspace_dir, or '' if unreadable."""
    cfg = read_ot_config()
    ws = cfg.get("workspace_dir", "")
    if not ws:
        return ""
    return os.path.normpath(os.path.join(OT_DIR, ws))


def find_ot_quant_cache() -> str:
    """
    Return the absolute path to use for SVDQuant / quantization cache.
    Priority:
      1. config.json → quantization.cache_dir
      2. config.json → cache_dir  (general OT cache)
      3. '' (caching disabled)
    """
    cfg = read_ot_config()
    quant_cache = (cfg.get("quantization") or {}).get("cache_dir", "") or ""
    if quant_cache:
        return os.path.normpath(os.path.join(OT_DIR, quant_cache))
    general_cache = cfg.get("cache_dir", "") or ""
    if general_cache:
        return os.path.normpath(os.path.join(OT_DIR, general_cache, "quantization"))
    return ""

"""Extract generation parameters from sampler outputs and ComfyUI files.

Supported formats:
  - Our own sampler PNG  (tEXt chunk "chroma_sampler" or "wan_sampler")
  - Our own sampler MP4  (looks for a sidecar .png at the same stem)
  - ComfyUI PNG          (tEXt chunk "prompt" containing a workflow JSON graph)
  - ComfyUI JSON         (.json file containing a workflow graph)

No tkinter dependency — pure data extraction.
"""
from __future__ import annotations

import json
import os

_OUR_TOOL_KEYS = ("chroma_sampler", "wan_sampler")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sampler_file(path: str) -> dict | None:
    """Try to extract generation info from *path*.

    Returns a dict with keys:
      ``display_path`` (str | None)  — path to show in the preview panel
                                       (may differ from *path* for MP4)
      ``is_video``     (bool)        — True when display_path is an MP4
      ``params``       (dict | None) — extracted generation parameters, or
                                       None if no recognised metadata found
      ``source``       (str)         — one of "our_png", "our_mp4",
                                       "comfyui_png", "comfyui_json",
                                       "unknown_image"

    Returns *None* if the file extension is not recognised at all.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".png":
        # Try our format first (most specific)
        params = _read_our_png(path)
        if params is not None:
            return {"display_path": path, "is_video": False,
                    "params": params, "source": "our_png"}
        # Try ComfyUI workflow embedded in PNG
        params = _read_comfyui_png(path)
        if params is not None:
            return {"display_path": path, "is_video": False,
                    "params": params, "source": "comfyui_png"}
        # Unknown PNG — show image but no metadata
        return {"display_path": path, "is_video": False,
                "params": None, "source": "unknown_image"}

    if ext == ".mp4":
        # 1. Try embedded MP4 metadata (our own tool key)
        mp4_chunks = _read_mp4_chunks(path)
        for key in _OUR_TOOL_KEYS:
            raw = mp4_chunks.get(key)
            if raw:
                try:
                    return {"display_path": path, "is_video": True,
                            "params": json.loads(raw), "source": "our_mp4"}
                except Exception:
                    pass

        # 2. Try ComfyUI embedded MP4 metadata ("prompt" key = workflow JSON)
        if "prompt" in mp4_chunks:
            params = _parse_comfyui_workflow(mp4_chunks["prompt"])
            if params is not None:
                return {"display_path": path, "is_video": True,
                        "params": params, "source": "comfyui_mp4"}

        # 3. Fall back to our sidecar PNG
        sidecar = os.path.splitext(path)[0] + ".png"
        if os.path.isfile(sidecar):
            params = _read_our_png(sidecar)
            if params is not None:
                return {"display_path": path, "is_video": True,
                        "params": params, "source": "our_mp4"}

        # 4. Video with no recognised metadata — show it anyway
        return {"display_path": path, "is_video": True,
                "params": None, "source": "unknown_video"}

    if ext == ".json":
        params = _read_comfyui_json(path)
        if params is not None:
            return {"display_path": None, "is_video": False,
                    "params": params, "source": "comfyui_json"}
        return None

    return None


# ---------------------------------------------------------------------------
# Internal helpers — our own format
# ---------------------------------------------------------------------------

def _read_our_png(path: str) -> dict | None:
    """Return the full params dict if *path* has one of our tool-key tEXt
    chunks; otherwise return None."""
    from sampler_core.util.png_meta import read_png_metadata
    chunks = read_png_metadata(path)
    for key in _OUR_TOOL_KEYS:
        raw = chunks.get(key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Internal helpers — ComfyUI format
# ---------------------------------------------------------------------------

def _read_comfyui_png(path: str) -> dict | None:
    """Parse a ComfyUI-style "prompt" tEXt chunk from *path*."""
    from sampler_core.util.png_meta import read_png_metadata
    chunks = read_png_metadata(path)
    raw = chunks.get("prompt", "")
    return _parse_comfyui_workflow(raw)


def _read_comfyui_json(path: str) -> dict | None:
    """Read *path* as a JSON file and attempt ComfyUI workflow parsing."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        return _parse_comfyui_workflow(raw)
    except Exception:
        return None


def _read_mp4_chunks(path: str) -> dict:
    """Return the ``mdta`` key→value dict from an MP4 file's ``udta`` box."""
    from sampler_core.util.png_meta import read_mp4_metadata
    return read_mp4_metadata(path)


def _parse_comfyui_workflow(raw: str) -> dict | None:
    """Parse *raw* as a ComfyUI workflow JSON string.

    Returns ``{"prompt": pos, "negative_prompt": neg}`` or *None* when the
    string is not a recognised ComfyUI workflow.
    """
    try:
        graph = json.loads(raw)
    except Exception:
        return None
    if not isinstance(graph, dict):
        return None

    # ComfyUI exports may nest the node map under a "nodes" key.
    nodes: dict = graph
    if "nodes" in graph and isinstance(graph["nodes"], dict):
        nodes = graph["nodes"]

    # Must look like a node graph: dict of {id: {class_type, inputs}}
    # Quick sanity check — at least one node must have class_type.
    if not any(
        isinstance(v, dict) and "class_type" in v
        for v in nodes.values()
    ):
        return None

    pos, neg = _extract_comfyui_prompts(nodes)
    if pos is None and neg is None:
        return None
    return {"prompt": pos or "", "negative_prompt": neg or ""}


def _extract_comfyui_prompts(nodes: dict) -> tuple[str | None, str | None]:
    """Walk the ComfyUI node graph to extract positive/negative prompt text.

    Strategy:
    1. Build a map of all CLIPTextEncode nodes → their text value.
    2. Find ANY node that has both a ``"positive"`` and a ``"negative"``
       input that are node-link references (i.e. ``[node_id, slot]``).
       This catches KSampler, KSamplerAdvanced, CFGGuider, and any
       custom sampler/guider node regardless of class name.
    3. Resolve those references through the graph (up to depth 4) until a
       CLIPTextEncode text string is reached.
    4. Fallback (only if no such node exists at all): return the two
       CLIPTextEncode texts sorted by node ID numerically.  This is
       unreliable for workflows where the negative encode has a lower ID
       than the positive, so it is only used as a last resort.
    """
    # --- Build CLIPTextEncode map ---
    clip_texts: dict[str, str] = {}
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "CLIPTextEncode":
            text = node.get("inputs", {}).get("text", "")
            # "text" can itself be a [node_id, slot] link — skip those
            if isinstance(text, str):
                clip_texts[str(nid)] = text

    # --- Find any node with explicit positive + negative conditioning refs ---
    def _is_link(v) -> bool:
        return isinstance(v, (list, tuple)) and len(v) >= 1

    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        inputs  = node.get("inputs", {})
        pos_ref = inputs.get("positive")
        neg_ref = inputs.get("negative")
        if _is_link(pos_ref) and _is_link(neg_ref):
            pos = _resolve_clip_text(pos_ref, nodes, clip_texts, depth=4)
            neg = _resolve_clip_text(neg_ref, nodes, clip_texts, depth=4)
            if pos is not None or neg is not None:
                return pos, neg

    # --- Fallback: sort by node ID and take first two CLIPTextEncode ---
    if not clip_texts:
        return None, None
    try:
        sorted_ids = sorted(clip_texts.keys(), key=lambda x: int(x))
    except ValueError:
        sorted_ids = sorted(clip_texts.keys())
    pos = clip_texts[sorted_ids[0]] if len(sorted_ids) > 0 else None
    neg = clip_texts[sorted_ids[1]] if len(sorted_ids) > 1 else None
    return pos, neg


def _resolve_clip_text(
    ref,
    nodes: dict,
    clip_texts: dict[str, str],
    depth: int,
) -> str | None:
    """Recursively follow a node-link reference until a CLIPTextEncode text
    string is found, stopping after *depth* hops to avoid infinite loops.

    *ref* is either ``[node_id, output_slot]`` (a ComfyUI link) or a plain
    string (already resolved).  Returns the text string or None.
    """
    if depth <= 0:
        return None
    if isinstance(ref, str):
        return ref
    if not (isinstance(ref, (list, tuple)) and len(ref) >= 1):
        return None

    nid = str(ref[0])

    # Direct CLIPTextEncode hit
    if nid in clip_texts:
        return clip_texts[nid]

    # Follow intermediate conditioning node
    node = nodes.get(nid)
    if not isinstance(node, dict):
        return None

    inputs = node.get("inputs", {})
    # Try common pass-through conditioning inputs in order of preference
    for key in ("conditioning", "positive", "cond_1", "cond1", "cond"):
        child_ref = inputs.get(key)
        if child_ref is not None:
            result = _resolve_clip_text(child_ref, nodes, clip_texts, depth - 1)
            if result is not None:
                return result

    return None

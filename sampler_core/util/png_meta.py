"""Read/write metadata for PNG and MP4 sampler outputs.

PNG:  standard tEXt chunks (ComfyUI-compatible).
MP4:  Apple/QuickTime ``udta → meta → keys / ilst`` atom structure,
      the same format used by ComfyUI for video outputs.  Pure Python,
      no extra dependencies beyond the stdlib ``struct`` module.
"""
import json
import os
import struct


def write_png_metadata(path: str, tool_key: str, params: dict) -> None:
    """
    Embed generation parameters into `path` as two PNG tEXt chunks:
      "prompt"    — plain positive prompt text (read by most image viewers)
      `tool_key`  — JSON dict with full generation parameters

    Non-fatal: if the write fails the image is still valid without metadata.
    `params` should already have None values removed by the caller if desired.
    """
    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        img = Image.open(path)
        pnginfo = PngInfo()
        pnginfo.add_text("prompt", params.get("prompt", ""))
        pnginfo.add_text(tool_key, json.dumps(params, ensure_ascii=False))
        img.save(path, pnginfo=pnginfo)
    except Exception as exc:
        print(f"[metadata] Warning: could not write PNG metadata: {exc}")


def _read_first_frame(video_path: str):
    """
    Return the first frame of a video file as a PIL Image, or None on failure.
    Tries PyAV first, then imageio (ffmpeg plugin), then cv2.
    """
    try:
        import av
        container = av.open(video_path)
        for frame in container.decode(video=0):
            img = frame.to_image()
            container.close()
            return img
        container.close()
    except Exception:
        pass

    try:
        import imageio
        reader = imageio.get_reader(video_path, format="ffmpeg")
        frame = reader.get_data(0)   # numpy HWC uint8
        reader.close()
        from PIL import Image
        return Image.fromarray(frame)
    except Exception:
        pass

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            from PIL import Image
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        pass

    return None


def write_png_sidecar(video_path: str, tool_key: str, params: dict) -> str | None:
    """
    Write a PNG alongside `video_path` carrying generation metadata as tEXt
    chunks (same format as write_png_metadata).  Used for video outputs (e.g.
    MP4) where the container has no standard metadata slot.

    The image is frame 0 of the video.  If the video cannot be read, falls
    back to a 1×1 black placeholder so metadata is never silently lost.

    The sidecar is saved as `<video_path_without_ext>.png`.
    Returns the sidecar path on success, None on failure (non-fatal).
    """
    sidecar_path = os.path.splitext(video_path)[0] + ".png"
    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        img = _read_first_frame(video_path)
        if img is None:
            img = Image.new("RGB", (1, 1), color=(0, 0, 0))

        pnginfo = PngInfo()
        pnginfo.add_text("prompt", params.get("prompt", ""))
        pnginfo.add_text(tool_key, json.dumps(params, ensure_ascii=False))
        img.save(sidecar_path, pnginfo=pnginfo)
        return sidecar_path
    except Exception as exc:
        print(f"[metadata] Warning: could not write PNG sidecar: {exc}")
        return None


def read_png_metadata(path: str) -> dict:
    """Return a dict of all PNG tEXt chunks for the file at `path`.

    Values are raw strings (not JSON-parsed).  Returns an empty dict on any
    failure (missing file, non-PNG, corrupt header, etc.).
    """
    try:
        from PIL import Image
        img = Image.open(path)
        return dict(img.text) if hasattr(img, "text") else {}
    except Exception:
        return {}


# ===========================================================================
# MP4 metadata — Apple/QuickTime udta / meta / keys / ilst
# ===========================================================================

def write_mp4_metadata(path: str, tool_key: str, params: dict) -> None:
    """Embed generation parameters into an MP4 file.

    Uses the Apple/QuickTime ``udta → meta → keys / ilst`` atom structure
    (the same format ComfyUI uses).  Three ``mdta`` keys are written:

      ``tool_key``   — JSON dict with full generation parameters
      ``"prompt"``   — plain positive prompt text
      ``"encoder"``  — ``"OneTrainer Sampler"``

    The function modifies the file in place by rewriting the ``moov`` box.
    Non-fatal: logs a warning and leaves the file unchanged on any error.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()

        moov_offset, moov_size = _find_box(data, b"moov", 0, len(data))
        if moov_offset < 0:
            print(f"[metadata] Warning: no moov box found in {path}")
            return

        # Build replacement moov: strip any existing udta, append ours.
        moov_inner    = data[moov_offset + 8 : moov_offset + moov_size]
        stripped      = _strip_box(moov_inner, b"udta")
        new_udta      = _build_mdta_udta({
            tool_key:  json.dumps(params, ensure_ascii=False),
            "prompt":  params.get("prompt", ""),
            "encoder": "OneTrainer Sampler",
        })
        new_moov_inner = stripped + new_udta
        new_moov       = (struct.pack(">I", 8 + len(new_moov_inner))
                          + b"moov" + new_moov_inner)

        new_data = data[:moov_offset] + new_moov + data[moov_offset + moov_size:]
        with open(path, "wb") as fh:
            fh.write(new_data)

    except Exception as exc:
        print(f"[metadata] Warning: could not write MP4 metadata: {exc}")


def read_mp4_metadata(path: str) -> dict:
    """Read Apple/QuickTime ``udta`` metadata from an MP4 file.

    Returns a ``{key_name: value_string}`` dict for all UTF-8 ``mdta`` keys
    found in the file's ``moov → udta → meta → keys / ilst`` structure.
    Returns ``{}`` on any failure or if no metadata is present.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        moov_off, moov_sz = _find_box(data, b"moov", 0, len(data))
        if moov_off < 0:
            return {}
        return _parse_moov_metadata(data, moov_off, moov_off + moov_sz)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_box(data: bytes, target: bytes,
              start: int, end: int) -> tuple[int, int]:
    """Return (offset, size) of the first top-level box named *target*
    in the byte range [start, end).  Returns (-1, 0) if not found."""
    offset = start
    while offset < end - 8:
        size = struct.unpack_from(">I", data, offset)[0]
        btype = data[offset + 4 : offset + 8]
        if size == 0:
            size = end - offset
        if size < 8:
            break
        if btype == target:
            return offset, size
        offset += size
    return -1, 0


def _strip_box(data: bytes, box_type: bytes) -> bytes:
    """Return *data* with every top-level box of *box_type* removed."""
    result = bytearray()
    offset = 0
    while offset < len(data) - 8:
        size = struct.unpack_from(">I", data, offset)[0]
        btype = data[offset + 4 : offset + 8]
        if size < 8:
            result += data[offset:]
            break
        if btype != box_type:
            result += data[offset : offset + size]
        offset += size
    return bytes(result)


def _build_mdta_udta(kv: dict) -> bytes:
    """Build a ``udta`` box containing Apple ``mdta``-style string metadata."""
    keys_list = list(kv.keys())

    # ── hdlr (handler reference — FullBox) ──────────────────────────────
    hdlr_payload = (
        b"\x00\x00\x00\x00"   # version + flags
        b"\x00\x00\x00\x00"   # pre_defined
        b"mdta"               # handler_type
        + bytes(12)            # reserved
        + b"\x00"              # name (null-terminated empty string)
    )
    hdlr = struct.pack(">I", 8 + len(hdlr_payload)) + b"hdlr" + hdlr_payload

    # ── keys (FullBox) ───────────────────────────────────────────────────
    key_entries = b""
    for key in keys_list:
        name_bytes  = key.encode("utf-8")
        entry_size  = 8 + len(name_bytes)      # 4 (size) + 4 (namespace) + name
        key_entries += struct.pack(">I", entry_size) + b"mdta" + name_bytes

    keys_payload = (
        b"\x00\x00\x00\x00"                      # version + flags
        + struct.pack(">I", len(keys_list))        # entry_count
        + key_entries
    )
    keys_box = struct.pack(">I", 8 + len(keys_payload)) + b"keys" + keys_payload

    # ── ilst (item list) ─────────────────────────────────────────────────
    ilst_items = b""
    for i, key in enumerate(keys_list, 1):
        val_bytes = kv[key].encode("utf-8")
        data_box  = (
            struct.pack(">I", 16 + len(val_bytes)) + b"data"
            + struct.pack(">I", 1)    # type 1 = UTF-8
            + struct.pack(">I", 0)    # locale = 0
            + val_bytes
        )
        item = struct.pack(">I", 8 + len(data_box)) + struct.pack(">I", i) + data_box
        ilst_items += item

    ilst = struct.pack(">I", 8 + len(ilst_items)) + b"ilst" + ilst_items

    # ── meta (FullBox) ───────────────────────────────────────────────────
    meta_inner = hdlr + keys_box + ilst
    meta = (
        struct.pack(">I", 12 + len(meta_inner)) + b"meta"
        + b"\x00\x00\x00\x00"   # version + flags
        + meta_inner
    )

    # ── udta ─────────────────────────────────────────────────────────────
    return struct.pack(">I", 8 + len(meta)) + b"udta" + meta


def _parse_moov_metadata(data: bytes, moov_start: int, moov_end: int) -> dict:
    udta_off, udta_sz = _find_box(data, b"udta", moov_start + 8, moov_end)
    if udta_off < 0:
        return {}
    return _parse_udta_metadata(data, udta_off, udta_off + udta_sz)


def _parse_udta_metadata(data: bytes, udta_start: int, udta_end: int) -> dict:
    meta_off, meta_sz = _find_box(data, b"meta", udta_start + 8, udta_end)
    if meta_off < 0:
        return {}
    # meta is a FullBox: 4-byte version/flags after the 8-byte header
    return _parse_meta_metadata(data, meta_off + 12, meta_off + meta_sz)


def _parse_meta_metadata(data: bytes, inner_start: int, inner_end: int) -> dict:
    """Parse ``keys`` and ``ilst`` sub-boxes from inside a ``meta`` FullBox."""
    key_names:   list[str] = []
    ilst_start = ilst_end = -1

    offset = inner_start
    while offset < inner_end - 8:
        size  = struct.unpack_from(">I", data, offset)[0]
        btype = data[offset + 4 : offset + 8]
        if size < 8:
            break

        if btype == b"keys":
            # FullBox: 4 version/flags + 4 entry_count
            entry_count = struct.unpack_from(">I", data, offset + 12)[0]
            k_off = offset + 16
            for _ in range(entry_count):
                if k_off + 8 > inner_end:
                    break
                k_size = struct.unpack_from(">I", data, k_off)[0]
                if k_size < 8:
                    break
                # key entry: 4 size + 4 namespace + name bytes
                name = data[k_off + 8 : k_off + k_size].decode("utf-8", errors="replace")
                key_names.append(name)
                k_off += k_size

        elif btype == b"ilst":
            ilst_start = offset + 8
            ilst_end   = offset + size

        offset += size

    if not key_names or ilst_start < 0:
        return {}

    result: dict = {}
    offset = ilst_start
    while offset < ilst_end - 8:
        item_size    = struct.unpack_from(">I", data, offset)[0]
        item_key_idx = struct.unpack_from(">I", data, offset + 4)[0]
        if item_size < 16 or not (1 <= item_key_idx <= len(key_names)):
            break

        key_name = key_names[item_key_idx - 1]
        d_off    = offset + 8
        if d_off + 16 <= offset + item_size:
            d_size = struct.unpack_from(">I", data, d_off)[0]
            d_type = data[d_off + 4 : d_off + 8]
            if d_type == b"data" and d_size >= 16:
                type_indicator = struct.unpack_from(">I", data, d_off + 8)[0]
                if type_indicator == 1:   # UTF-8
                    value = data[d_off + 16 : d_off + d_size].decode("utf-8", errors="replace")
                    result[key_name] = value

        offset += item_size

    return result

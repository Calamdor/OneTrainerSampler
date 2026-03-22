"""Write ComfyUI-compatible tEXt metadata chunks into a saved PNG."""
import json
import os


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
    Tries imageio (ffmpeg plugin) first, then cv2.
    """
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

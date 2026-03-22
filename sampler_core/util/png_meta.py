"""Write ComfyUI-compatible tEXt metadata chunks into a saved PNG."""
import json


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

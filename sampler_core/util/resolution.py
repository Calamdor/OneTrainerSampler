import math

# (label, width_ratio, height_ratio)
ASPECT_RATIOS = [
    ("1:1",   1,  1),
    ("5:4",   5,  4),
    ("4:5",   4,  5),
    ("4:3",   4,  3),
    ("3:4",   3,  4),
    ("3:2",   3,  2),
    ("2:3",   2,  3),
    ("16:10", 16, 10),
    ("10:16", 10, 16),
    ("16:9",  16,  9),
    ("9:16",   9, 16),
    ("21:9",  21,  9),
    ("9:21",   9, 21),
]

ASPECT_RATIO_LABELS = [r[0] for r in ASPECT_RATIOS]
ASPECT_RATIO_MAP    = {r[0]: (r[1], r[2]) for r in ASPECT_RATIOS}
PIXEL_TARGET_OPTIONS = ["512", "768", "896", "1024", "1152", "1280", "1536", "2048"]

ATTN_BACKEND_OPTIONS   = ["Auto", "Flash", "SageAttn"]
ATTN_BACKEND_ENUM_NAME = {"Flash": "FLASH", "SageAttn": "SAGE"}


def compute_dims(pixel_target: int, aspect_label: str, quantize: int = 64) -> tuple[int, int]:
    """
    Return (width, height) such that:
      - w / h == ratio_w / ratio_h
      - w * h ≈ pixel_target²  (constant total-pixel budget)
      - both are multiples of `quantize`
    """
    rw, rh = ASPECT_RATIO_MAP.get(aspect_label, (1, 1))
    r = rw / rh
    w = max(quantize, round(pixel_target * math.sqrt(r) / quantize) * quantize)
    h = max(quantize, round(pixel_target / math.sqrt(r) / quantize) * quantize)
    return w, h


def check_attn_backends() -> dict[str, bool]:
    avail = {"Auto": True}
    try:
        import flash_attn  # noqa: F401
        avail["Flash"] = True
    except ImportError:
        avail["Flash"] = False
    try:
        import sageattention  # noqa: F401
        avail["SageAttn"] = True
    except ImportError:
        avail["SageAttn"] = False
    return avail

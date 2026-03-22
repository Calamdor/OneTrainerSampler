import struct

_GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 14: "Q2_K", 15: "Q3_K_S", 16: "Q4_K_S",
    17: "Q5_K_S", 18: "Q6_K", 19: "Q8_K", 30: "BF16",
}


def read_gguf_tensor_types(path: str) -> dict[str, int]:
    """Parse GGUF header and return {type_name: count}, or {} on error."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                return {}
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def _read_str():
                n = struct.unpack("<Q", f.read(8))[0]
                f.read(n)

            def _skip(typ):
                _SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                if typ in _SIZES:
                    f.read(_SIZES[typ])
                elif typ == 8:
                    _read_str()
                elif typ == 9:
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_count = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(arr_count):
                        _skip(arr_type)

            for _ in range(n_kv):
                _read_str()
                _skip(struct.unpack("<I", f.read(4))[0])

            counts: dict[str, int] = {}
            for _ in range(n_tensors):
                _read_str()
                n_dims = struct.unpack("<I", f.read(4))[0]
                f.read(n_dims * 8)
                ggml_type = struct.unpack("<I", f.read(4))[0]
                f.read(8)
                name = _GGML_TYPE_NAMES.get(ggml_type, f"type{ggml_type}")
                counts[name] = counts.get(name, 0) + 1
        return counts
    except Exception:
        return {}


def gguf_type_summary(path: str) -> str:
    counts = read_gguf_tensor_types(path)
    if not counts:
        return "unknown"
    total = sum(counts.values())
    parts = [
        f"{typ}×{cnt} ({cnt * 100 // total}%)"
        for typ, cnt in sorted(counts.items(), key=lambda x: -x[1])
    ]
    return "  ".join(parts)

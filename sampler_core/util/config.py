import json
import os


def load_config(defaults: dict, path: str) -> dict:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = dict(defaults)
            cfg.update(data)
            return cfg
        except Exception:
            pass
    return dict(defaults)


def save_config(cfg: dict, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as exc:
        print(f"[config] save failed: {exc}")

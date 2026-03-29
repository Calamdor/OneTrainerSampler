"""
OneTrainerLauncher — top-level model selector + dynamic GUI builder.

A model dropdown sits above a content frame.  Switching models tears down the
content frame, instantiates a fresh BaseSamplerApp subclass inside a new frame,
and saves the last-used model to sampler_launcher_config.json.

Adding a new model:
  1. Add its display name to _MODELS.
  2. Add a builder branch in _build_model_ui().
  Per-model settings are saved to their own JSON files (e.g. chroma_sampler_config.json)
  so switching models never corrupts another model's settings.
"""
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

from sampler_core.gui.theme import apply_dark_theme, FG

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_SAMPLERS_DIR    = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_LAUNCHER_CFG    = os.path.join(_SAMPLERS_DIR, "config", "sampler_launcher_config.json")

# Ordered list of available models shown in the dropdown.
_MODELS = [
    "Chroma",
    "Wan 2.2 T2V-A14B",
]


class OneTrainerLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("OneTrainer Sampler")
        root.resizable(True, True)
        apply_dark_theme(root)

        self._current_app  = None
        self._current_name: str | None = None
        self._content_frame: ttk.Frame | None = None

        # ── Model selector row ────────────────────────────────────────────
        sel_row = ttk.Frame(root)
        sel_row.pack(fill="x", padx=8, pady=(6, 4))

        ttk.Label(sel_row, text="Model:",
                  font=("TkDefaultFont", 9, "bold")).pack(side="left")

        last = self._load_last_model()
        self._model_var = tk.StringVar(value=last)
        self._model_combo = ttk.Combobox(
            sel_row, textvariable=self._model_var,
            values=_MODELS, state="readonly", width=30,
        )
        self._model_combo.pack(side="left", padx=6)

        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=6, pady=(0, 2))

        # Build initial model UI, then bind so the trace doesn't fire during init.
        self._build_model_ui(last)
        self._model_var.trace_add("write", self._on_model_changed)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Model switching ───────────────────────────────────────────────────

    def _on_model_changed(self, *_) -> None:
        new_name = self._model_var.get()
        if new_name == self._current_name:
            return

        if self._current_app is not None and self._current_app.backend.model is not None:
            if not messagebox.askyesno(
                "Switch model",
                f"Switching to {new_name} will unload the current model.\n\nContinue?",
                parent=self.root,
            ):
                # Revert dropdown without re-triggering the trace.
                cbname = self._model_var.trace_info()[0][1]
                self._model_var.trace_remove("write", cbname)
                self._model_var.set(self._current_name)
                self._model_var.trace_add("write", self._on_model_changed)
                return

        if self._current_app is not None:
            self._current_app.cleanup()
            self._current_app.backend.unload_model()
        self._save_last_model(new_name)
        self._build_model_ui(new_name)

    def _build_model_ui(self, model_name: str) -> None:
        if self._content_frame is not None:
            self._content_frame.destroy()

        self._content_frame = ttk.Frame(self.root)
        self._content_frame.pack(fill="both", expand=True)
        self._current_name  = model_name

        if model_name == "Chroma":
            import sampler_core  # noqa: F401 — injects OT path
            from chroma.gui import ChromaSamplerApp
            self._current_app = ChromaSamplerApp(self.root,
                                                 container=self._content_frame)

        elif model_name == "Wan 2.2 T2V-A14B":
            import sampler_core  # noqa: F401
            from wan.gui import WanSamplerApp
            self._current_app = WanSamplerApp(self.root,
                                              container=self._content_frame)

        else:
            ttk.Label(
                self._content_frame,
                text=f'"{model_name}" is not yet implemented.',
                foreground=FG,
            ).pack(padx=20, pady=30)
            self._current_app = None

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_last_model(self) -> str:
        try:
            if os.path.isfile(_LAUNCHER_CFG):
                with open(_LAUNCHER_CFG, encoding="utf-8") as f:
                    name = json.load(f).get("model", _MODELS[0])
                    if name in _MODELS:
                        return name
        except Exception:
            pass
        return _MODELS[0]

    def _save_last_model(self, model: str) -> None:
        try:
            with open(_LAUNCHER_CFG, "w", encoding="utf-8") as f:
                json.dump({"model": model}, f)
        except Exception:
            pass

    # ── Close ─────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._current_app is not None:
            self._current_app.cleanup()
        self.root.destroy()

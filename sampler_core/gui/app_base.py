"""
BaseSamplerApp — abstract tkinter application base for all standalone samplers.

Subclasses implement:
  _build_model_frame(pad)       — model settings panel
  _build_gen_params(gen_frame, r) -> int  — model-specific generation params
  _collect_cfg() -> dict        — snapshot of all current settings
  _add_lora(path, weight, **kw) — add one LoRA row (model-specific extra columns)
  _get_lora_list() -> list[dict]
  _load_loras_from_config()

Everything else (blink, busy, queue, lora panel infrastructure, output panel,
prompt/negative, generate/abort, token counter, cfg save/load) is shared.
"""
import os
import time
import threading
from abc import ABC, abstractmethod

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from sampler_core.util.config import load_config, save_config
from sampler_core.util.ot_bridge import find_ot_workspace, find_ot_quant_cache
from sampler_core.util.resolution import check_attn_backends
from sampler_core.gui.tooltip import Tooltip  # noqa: F401 — re-exported for subclasses
from sampler_core.gui.theme import apply_dark_theme, BG, BG_INPUT, FG, BG_WIDGET, FG_DIM


class BaseSamplerApp(ABC):
    def __init__(self, root: tk.Tk, backend, defaults: dict, config_path: str, title: str,
                 container: tk.Frame | None = None):
        self.root = root
        # All widgets pack into self.frame; inside a launcher this is the
        # content frame, standalone it IS the root window.
        self._standalone = container is None
        self.frame = container if container is not None else root

        if self._standalone:
            root.title(title)
            root.resizable(True, True)
            apply_dark_theme(root)

        self.cfg          = load_config(defaults, config_path)
        self._config_path = config_path
        self.backend      = backend

        self._lora_rows: list[dict] = []
        self._last_output: str | None = None
        self._busy        = False
        self._blink_id: str | None = None
        self._blink_state = False

        self._queue: list[dict] = []
        self._queue_running        = False
        self._queue_stop_requested = False

        self._build_ui()
        self._load_loras_from_config()

    # ==================================================================
    # Abstract interface
    # ==================================================================

    @abstractmethod
    def _build_model_frame(self, pad: dict) -> None:
        """Build and pack the Model Settings LabelFrame.
        Must set self._model_status_label (tk.Label) and self._model_status_var.
        Should set self._gguf_frame, self._svd_var, self._svd_row_widgets,
        self._attn_var, self._attn_avail_var for shared helpers to work."""

    @abstractmethod
    def _build_gen_params(self, gen_frame: ttk.LabelFrame, r: int) -> int:
        """Build model-specific generation widgets (dims, steps, etc.) inside
        gen_frame using grid layout starting at row r.  Return the next row."""

    @abstractmethod
    def _collect_cfg(self) -> dict:
        """Return a complete snapshot of all current UI settings."""

    @abstractmethod
    def _add_lora(self, path: str = "", weight: float = 1.0, **kwargs) -> None:
        """Add one LoRA row to the scrollable list."""

    @abstractmethod
    def _get_lora_list(self) -> list[dict]:
        """Return the current LoRA list as serialisable dicts."""

    @abstractmethod
    def _load_loras_from_config(self) -> None:
        """Restore LoRA rows from self.cfg."""

    # ==================================================================
    # UI build
    # ==================================================================

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 3}

        # Two-column layout: controls on the left, preview + details on the right.
        # self.frame is reassigned to the left pane so all existing panel builders
        # below continue to work without modification.
        _paned = tk.PanedWindow(
            self.frame, orient=tk.HORIZONTAL,
            sashwidth=5, sashrelief="flat",
            background=BG, bd=0,
        )
        _paned.pack(fill="both", expand=True)

        self._left_frame  = ttk.Frame(_paned)
        self._right_frame = ttk.Frame(_paned)
        _paned.add(self._left_frame,  minsize=480, stretch="never")
        _paned.add(self._right_frame, minsize=340, stretch="always")

        # Redirect self.frame → left column so every builder below targets it.
        self.frame = self._left_frame

        self._build_model_frame(pad)
        self._build_lora_panel(pad)

        # ---- Generation frame ----------------------------------------
        gen_frame = ttk.LabelFrame(self.frame, text="Generation")
        gen_frame.pack(fill="x", **pad)

        r = 0
        r = self._build_prompt_widgets(gen_frame, r, pad)
        r = self._build_gen_params(gen_frame, r)
        r = self._build_gen_controls(gen_frame, r, pad)
        gen_frame.columnconfigure(1, weight=1)

        self._build_output_panel(pad)
        self._build_queue_panel(pad)
        self._build_right_panel(pad)

        # Token counter bindings (prompt text widget created in _build_prompt_widgets)
        self._prompt_text.bind("<KeyRelease>", self._schedule_token_count)
        self._prompt_text.bind("<<Paste>>",    self._schedule_token_count)

        if self._standalone:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_attn_avail()

    # ------------------------------------------------------------------
    def _build_prompt_widgets(self, gen_frame, r: int, pad: dict) -> int:
        ttk.Label(gen_frame, text="Prompt:").grid(row=r, column=0, sticky="nw", **pad)
        _pf = ttk.Frame(gen_frame)
        _pf.grid(row=r, column=1, columnspan=5, sticky="ew", **pad)
        self._prompt_text = tk.Text(
            _pf, height=3, wrap="word", undo=True,
            font=("TkDefaultFont", 9), relief="flat", borderwidth=1,
            background=BG_INPUT, foreground=FG, insertbackground=FG,
            selectbackground="#264f78", selectforeground=FG,
        )
        _ps = ttk.Scrollbar(_pf, orient="vertical", command=self._prompt_text.yview)
        self._prompt_text.configure(yscrollcommand=_ps.set)
        self._prompt_text.pack(side="left", fill="both", expand=True)
        _ps.pack(side="right", fill="y")
        self._prompt_text.insert("1.0", self.cfg.get("prompt", ""))
        Tooltip(self._prompt_text,
                "Positive prompt — describe what you want in the image.\n\n"
                "T5 token limit: 512 tokens (counter shown to the right).\n"
                "Text beyond 512 tokens is silently truncated.")

        self._token_count_var = tk.StringVar(value="")
        self._token_label = ttk.Label(
            gen_frame, textvariable=self._token_count_var,
            foreground="gray", font=("TkDefaultFont", 8), width=10, anchor="nw",
        )
        self._token_label.grid(row=r, column=6, sticky="nw", padx=(0, 6))
        Tooltip(self._token_label,
                "T5 token count for the positive prompt.\n"
                "Turns red when over the 512-token limit.\n\n"
                "Counted using the loaded model's tokenizer;\n"
                "falls back to downloading the T5 vocab if no model is loaded.")

        r += 1
        ttk.Label(gen_frame, text="Negative:").grid(row=r, column=0, sticky="nw", **pad)
        _nf = ttk.Frame(gen_frame)
        _nf.grid(row=r, column=1, columnspan=6, sticky="ew", **pad)
        self._neg_text = tk.Text(
            _nf, height=2, wrap="word", undo=True,
            font=("TkDefaultFont", 9), relief="flat", borderwidth=1,
            background=BG_INPUT, foreground=FG, insertbackground=FG,
            selectbackground="#264f78", selectforeground=FG,
        )
        _ns = ttk.Scrollbar(_nf, orient="vertical", command=self._neg_text.yview)
        self._neg_text.configure(yscrollcommand=_ns.set)
        self._neg_text.pack(side="left", fill="both", expand=True)
        _ns.pack(side="right", fill="y")
        self._neg_text.insert("1.0", self.cfg.get("negative_prompt", ""))
        Tooltip(self._neg_text,
                "Negative prompt — describe what to avoid in the image.\n\n"
                "Applied via classifier-free guidance (CFG). At CFG=1.0 the\n"
                "negative prompt has no effect. Higher CFG increases its influence.")
        return r + 1

    # ------------------------------------------------------------------
    def _build_gen_controls(self, gen_frame, r: int, pad: dict) -> int:
        """Output-dir row only.  Action buttons live in the Queue panel."""
        ctrl_row = ttk.Frame(gen_frame)
        ctrl_row.grid(row=r, column=0, columnspan=8, sticky="w", **pad)
        ttk.Label(ctrl_row, text="Output dir:").pack(side="left")
        self._outdir_var = tk.StringVar(value=self.cfg.get("output_dir", "output"))
        _outdir_entry = ttk.Entry(ctrl_row, textvariable=self._outdir_var, width=44)
        _outdir_entry.pack(side="left", padx=4)
        Tooltip(_outdir_entry, "Directory where generated images are saved.\n"
                "Images are named: YYYY-MM-DD_HH-MM-SS-sample-<seed>.png")
        ttk.Button(ctrl_row, text="Browse…",        command=self._browse_outdir).pack(side="left", padx=2)
        _ws_btn = ttk.Button(ctrl_row, text="From workspace", command=self._use_workspace_dir)
        _ws_btn.pack(side="left", padx=2)
        Tooltip(_ws_btn, "Set output dir to the workspace_dir from OneTrainer's active config.\n"
                "Keeps generated samples alongside your training workspace.")
        return r + 1

    # ------------------------------------------------------------------
    def _build_lora_panel(self, pad: dict) -> None:
        lora_frame = ttk.LabelFrame(self.frame, text="LoRA List")
        lora_frame.pack(fill="both", expand=False, **pad)

        add_row = ttk.Frame(lora_frame)
        add_row.pack(fill="x", pady=2)
        _add_lora_btn = ttk.Button(add_row, text="+ Add LoRA", command=self._add_lora)
        _add_lora_btn.pack(side="left", padx=2)
        Tooltip(_add_lora_btn, "Add a LoRA (.safetensors) to the list.\n\n"
                "LoRAs are not applied until you click Apply LoRAs.\n"
                "Accepted formats: OT internal, ai-toolkit, kohya, musubi-tuner.")

        canvas_frame = ttk.Frame(lora_frame)
        canvas_frame.pack(fill="both", expand=True)
        self._lora_canvas = tk.Canvas(canvas_frame, height=110, highlightthickness=0,
                                       background=BG)
        lora_scroll = ttk.Scrollbar(canvas_frame, orient="vertical",
                                    command=self._lora_canvas.yview)
        self._lora_canvas.configure(yscrollcommand=lora_scroll.set)
        self._lora_canvas.pack(side="left", fill="both", expand=True)
        lora_scroll.pack(side="right", fill="y")

        self._lora_inner = ttk.Frame(self._lora_canvas)
        self._lora_canvas_window = self._lora_canvas.create_window(
            (0, 0), window=self._lora_inner, anchor="nw",
        )
        self._lora_inner.bind("<Configure>", self._on_lora_frame_configure)
        self._lora_canvas.bind("<Configure>", self._on_canvas_configure)

        self._build_lora_header(self._lora_inner)

        apply_row = ttk.Frame(lora_frame)
        apply_row.pack(fill="x", pady=2)
        _apply_btn = ttk.Button(apply_row, text="Apply LoRAs", command=self._apply_loras)
        _apply_btn.pack(side="left", padx=2)
        Tooltip(_apply_btn,
                "Apply all enabled LoRAs to the loaded model as forward hooks.\n\n"
                "LoRAs stay applied until you click Clear LoRAs or reload the model.\n"
                "Weight column controls each LoRA's strength (1.0 = full strength).\n\n"
                "Note: applying LoRAs while torch.compile is active will run inference\n"
                "in eager mode for that session (avoids a 5-min recompile).")
        _clear_btn = ttk.Button(apply_row, text="Clear LoRAs", command=self._clear_loras)
        _clear_btn.pack(side="left", padx=2)
        Tooltip(_clear_btn, "Remove all applied LoRA hooks from the model.\n"
                "The model reverts to its base weights. No reload needed.")
        self._lora_status_var = tk.StringVar(value="No LoRAs applied")
        ttk.Label(apply_row, textvariable=self._lora_status_var,
                  foreground="gray").pack(side="left", padx=8)

    def _build_lora_header(self, parent) -> None:
        """Default header: Path / Weight / En.  Subclass overrides to add columns."""
        hdr = ttk.Frame(parent)
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Path",   width=46, anchor="w").pack(side="left", padx=2)
        ttk.Label(hdr, text="Weight", width=7,  anchor="w").pack(side="left", padx=2)
        ttk.Label(hdr, text="En",     width=3,  anchor="w").pack(side="left", padx=2)
        ttk.Label(hdr, text="",       width=3).pack(side="left")

    # ------------------------------------------------------------------
    def _build_output_panel(self, pad: dict) -> None:
        out_frame = ttk.LabelFrame(self.frame, text="Output")
        out_frame.pack(fill="x", **pad)
        self._out_path_var = tk.StringVar(value="—")
        ttk.Label(out_frame, textvariable=self._out_path_var,
                  width=70, anchor="w").pack(side="left", padx=4)
        _open_btn = ttk.Button(out_frame, text="Open",   command=self._open_output)
        _open_btn.pack(side="left", padx=2)
        Tooltip(_open_btn, "Open the last generated image in the system viewer.")
        _folder_btn = ttk.Button(out_frame, text="Folder", command=self._open_folder)
        _folder_btn.pack(side="left", padx=2)
        Tooltip(_folder_btn, "Open the output directory in Windows Explorer.")

    # ------------------------------------------------------------------
    def _build_queue_panel(self, pad: dict) -> None:
        queue_frame = ttk.LabelFrame(self.frame, text="Queue")
        queue_frame.pack(fill="both", expand=False, **pad)

        # --- button row ---
        btn_row = ttk.Frame(queue_frame)
        btn_row.pack(fill="x", pady=(2, 0))
        self._add_queue_btn = ttk.Button(
            btn_row, text="+ Add to Queue", command=self._queue_add)
        self._add_queue_btn.pack(side="left", padx=2)
        Tooltip(self._add_queue_btn,
                "Snapshot the current settings and add a job to the queue.\n\n"
                "The queue starts automatically once the model is loaded.\n"
                "Each job runs with the settings captured at the time it was added,\n"
                "so you can queue multiple jobs with different prompts/seeds.")
        _remove_btn = ttk.Button(btn_row, text="Remove", command=self._queue_remove_selected)
        _remove_btn.pack(side="left", padx=2)
        Tooltip(_remove_btn, "Remove the selected job(s) from the queue.\n"
                "Running jobs cannot be removed — use Abort first.")
        _clear_btn = ttk.Button(btn_row, text="Clear",  command=self._queue_clear)
        _clear_btn.pack(side="left", padx=2)
        Tooltip(_clear_btn, "Remove all pending jobs from the queue.\n"
                "Any currently running job is not affected.")

        # --- progress row ---
        prog_row = ttk.Frame(queue_frame)
        prog_row.pack(fill="x", pady=(2, 2))
        self._abort_btn = tk.Button(
            prog_row, text="Abort", fg="white", bg="#cc3333",
            activebackground="#ff4444", activeforeground="white",
            disabledforeground="#888888",
            relief="flat", padx=6, pady=3, state="disabled",
            command=self._abort,
        )
        self._abort_btn.pack(side="left", padx=(2, 4))
        Tooltip(self._abort_btn,
                "Abort the current job and stop the queue.\n\n"
                "The current diffusion step finishes before stopping.\n"
                "Pending jobs remain in the queue and can be restarted.")
        self._progress = ttk.Progressbar(prog_row, mode="determinate", length=280)
        self._progress.pack(side="left")
        self._step_label_var = tk.StringVar(value="")
        ttk.Label(prog_row, textvariable=self._step_label_var,
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="left", padx=(4, 8))
        self._queue_status_var = tk.StringVar(value="")
        ttk.Label(prog_row, textvariable=self._queue_status_var,
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="left")

        # --- tree ---
        tree_frame = ttk.Frame(queue_frame)
        tree_frame.pack(fill="both", expand=True, pady=(0, 2))
        _qcols = ("prompt", "resolution", "steps", "seed", "time", "spit", "status")
        self._queue_tree = ttk.Treeview(tree_frame, columns=_qcols, show="headings", height=4)
        for _col, _lbl, _w, _anchor in [
            ("prompt",     "Prompt",     300, "w"),
            ("resolution", "Resolution",  90, "center"),
            ("steps",      "Steps",       50, "center"),
            ("seed",       "Seed",        80, "center"),
            ("time",       "Time",        55, "center"),
            ("spit",       "s/it",        45, "center"),
            ("status",     "Status",      80, "center"),
        ]:
            self._queue_tree.heading(_col, text=_lbl)
            self._queue_tree.column(_col, width=_w, anchor=_anchor,
                                    stretch=(_col == "prompt"))
        _qs = ttk.Scrollbar(tree_frame, orient="vertical", command=self._queue_tree.yview)
        self._queue_tree.configure(yscrollcommand=_qs.set)
        self._queue_tree.pack(side="left", fill="both", expand=True)
        _qs.pack(side="right", fill="y")
        self._queue_tree.bind("<<TreeviewSelect>>", self._on_queue_select)

    # ------------------------------------------------------------------
    def _build_right_panel(self, pad: dict) -> None:
        """Build the image preview and run-details panel (right column)."""

        # ---- Preview -----------------------------------------------
        preview_frame = ttk.LabelFrame(self._right_frame, text="Preview")
        preview_frame.pack(fill="both", expand=True, **pad)

        self._preview_container = tk.Frame(preview_frame, background=BG)
        self._preview_container.pack(fill="both", expand=True, padx=2, pady=2)

        self._preview_label = tk.Label(
            self._preview_container,
            text="No image yet",
            background=BG,
            foreground=FG_DIM,
            font=("TkDefaultFont", 10),
            cursor="hand2",
        )
        self._preview_label.pack(fill="both", expand=True)
        self._preview_label.bind("<Button-1>", lambda _: self._open_output())

        self._preview_pil_img = None   # PIL.Image kept for rescaling
        self._preview_photo   = None   # ImageTk reference kept to prevent GC

        self._preview_container.bind("<Configure>", self._on_preview_resize)

        # ---- Run details -------------------------------------------
        details_frame = ttk.LabelFrame(self._right_frame, text="Run Details")
        details_frame.pack(fill="x", **pad)

        self._detail_vars: dict[str, tk.StringVar] = {}
        _scalar_fields = [
            ("file",    "File"),
            ("seed",    "Seed"),
            ("res",     "Resolution"),
            ("steps",   "Steps"),
            ("cfg",     "CFG"),
            ("sampler", "Sampler"),
        ]
        for i, (key, label) in enumerate(_scalar_fields):
            ttk.Label(
                details_frame, text=label + ":", anchor="e", width=10,
            ).grid(row=i, column=0, sticky="e", padx=(6, 2), pady=1)
            var = tk.StringVar(value="—")
            self._detail_vars[key] = var
            ttk.Label(details_frame, textvariable=var, anchor="w").grid(
                row=i, column=1, sticky="ew", padx=(2, 6), pady=1,
            )

        r = len(_scalar_fields)
        ttk.Label(details_frame, text="Prompt:", anchor="ne").grid(
            row=r, column=0, sticky="ne", padx=(6, 2), pady=(4, 1),
        )
        self._detail_prompt_text = tk.Text(
            details_frame, height=3, wrap="word", state="disabled",
            font=("TkDefaultFont", 8), relief="flat", borderwidth=1,
            background=BG_INPUT, foreground=FG, insertbackground=FG,
            selectbackground="#264f78", selectforeground=FG,
        )
        self._detail_prompt_text.grid(
            row=r, column=1, sticky="ew", padx=(2, 6), pady=(4, 1),
        )

        r += 1
        ttk.Label(details_frame, text="Negative:", anchor="ne").grid(
            row=r, column=0, sticky="ne", padx=(6, 2), pady=1,
        )
        self._detail_neg_text = tk.Text(
            details_frame, height=2, wrap="word", state="disabled",
            font=("TkDefaultFont", 8), relief="flat", borderwidth=1,
            background=BG_INPUT, foreground=FG, insertbackground=FG,
            selectbackground="#264f78", selectforeground=FG,
        )
        self._detail_neg_text.grid(
            row=r, column=1, sticky="ew", padx=(2, 6), pady=1,
        )

        details_frame.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    def _on_preview_resize(self, event=None) -> None:
        if hasattr(self, "_preview_resize_id"):
            self.root.after_cancel(self._preview_resize_id)
        self._preview_resize_id = self.root.after(80, self._redraw_preview)

    def _redraw_preview(self) -> None:
        if self._preview_pil_img is None:
            return
        w = self._preview_container.winfo_width() - 8
        h = self._preview_container.winfo_height() - 8
        if w < 10 or h < 10:
            return
        try:
            from PIL import Image, ImageTk
            img = self._preview_pil_img.copy()
            img.thumbnail((w, h), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(img)
            self._preview_label.config(image=self._preview_photo, text="")
        except Exception:
            pass

    def _update_right_panel(self, path: str, cfg: dict | None = None) -> None:
        """Load the finished image into the preview and populate run details."""
        # Image
        try:
            from PIL import Image
            self._preview_pil_img = Image.open(path)
            self._redraw_preview()
        except Exception:
            pass

        # Details
        if cfg is None:
            return

        self._detail_vars["file"].set(os.path.basename(path))
        self._detail_vars["seed"].set(str(cfg.get("seed", "—")))

        w, h = cfg.get("width", 0), cfg.get("height", 0)
        self._detail_vars["res"].set(f"{w} × {h}" if w and h else "—")

        self._detail_vars["steps"].set(str(cfg.get("steps", "—")))
        self._detail_vars["cfg"].set(str(cfg.get("cfg_scale", "—")))

        sched = cfg.get("scheduler", "")
        shift = cfg.get("sigma_shift", "")
        if sched and shift != "":
            self._detail_vars["sampler"].set(f"{sched}  σ={shift}")
        else:
            self._detail_vars["sampler"].set(sched or "—")

        for widget, key in (
            (self._detail_prompt_text, "prompt"),
            (self._detail_neg_text,    "negative_prompt"),
        ):
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", cfg.get(key, ""))
            widget.config(state="disabled")

    # ==================================================================
    # Shared blink / status / busy
    # ==================================================================

    def _set_model_status(self, text: str, warn: bool = False) -> None:
        self._stop_blink()
        self._model_status_var.set(text)
        if warn:
            self._start_blink()
        else:
            self._model_status_label.config(foreground="gray")

    def _start_blink(self):
        self._stop_blink()
        self._blink_state = False
        self._do_blink()

    def _stop_blink(self):
        if self._blink_id is not None:
            self.root.after_cancel(self._blink_id)
            self._blink_id = None

    def _do_blink(self):
        self._blink_state = not self._blink_state
        self._model_status_label.config(
            foreground="#cc3333" if self._blink_state else "#888888"
        )
        self._blink_id = self.root.after(500, self._do_blink)

    def _on_model_setting_changed(self, *_):
        if self.backend.model is not None:
            self._set_model_status("⚠ Settings changed — reload required", warn=True)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._add_queue_btn.config(state="disabled" if busy else "normal")
        if not busy:
            # Model may just have finished loading — kick off pending jobs if any
            self._auto_start_queue()

    def _run_in_thread(self, fn) -> None:
        self._set_busy(True)
        def _work():
            try:
                fn()
            finally:
                self.root.after(0, self._set_busy, False)
        threading.Thread(target=_work, daemon=True).start()

    # ==================================================================
    # Shared LoRA panel logic
    # ==================================================================

    def _on_lora_frame_configure(self, _event=None):
        self._lora_canvas.configure(scrollregion=self._lora_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._lora_canvas.itemconfig(self._lora_canvas_window, width=event.width)

    def _remove_lora_row(self, row: dict) -> None:
        if row in self._lora_rows:
            self._lora_rows.remove(row)
        row["frame"].destroy()
        self._on_lora_frame_configure()
        self._save_cfg()

    def _apply_loras(self) -> None:
        if self._busy or self._queue_running:
            return
        loras = self._get_lora_list()
        self._run_in_thread(
            lambda: self.backend.apply_loras(
                loras,
                on_status=lambda s: self.root.after(0, self._lora_status_var.set, s),
            )
        )

    def _clear_loras(self) -> None:
        if self._busy or self._queue_running:
            return
        self.backend.remove_loras()
        self._lora_status_var.set("LoRAs cleared")

    # ==================================================================
    # Shared generate / abort
    # ==================================================================

    def _get_total_steps(self, cfg: dict) -> int:
        """Override in subclass if the model uses a non-standard steps key."""
        return cfg.get("steps", 30)

    def _abort(self) -> None:
        self._queue_stop_requested = True
        self.backend.cancel()
        self._abort_btn.config(state="disabled")

    # ==================================================================
    # Queue
    # ==================================================================

    def _queue_add(self) -> None:
        self._save_cfg()
        cfg = self._collect_cfg()
        prompt_short = cfg.get("prompt", "")[:45].replace("\n", " ")
        if len(cfg.get("prompt", "")) > 45:
            prompt_short += "…"
        seed_str = "rnd" if cfg.get("random_seed") else str(cfg.get("seed", 0))
        total    = self._get_total_steps(cfg)
        w, h     = cfg.get("width", 0), cfg.get("height", 0)
        res_str  = f"{w}×{h}" if w and h else cfg.get("aspect_ratio", "")
        iid = self._queue_tree.insert("", "end", values=(
            prompt_short, res_str, total, seed_str, "", "", "Pending",
        ))
        self._queue.append({"cfg": cfg, "status": "Pending", "iid": iid})
        self._auto_start_queue()

    def _queue_remove_selected(self) -> None:
        for iid in self._queue_tree.selection():
            job = next((j for j in self._queue if j["iid"] == iid), None)
            if job and job["status"] != "Running":
                self._queue.remove(job)
                self._queue_tree.delete(iid)
        if not self._queue_running:
            self._queue_status_var.set("")

    def _queue_clear(self) -> None:
        for job in list(self._queue):
            if job["status"] != "Running":
                self._queue_tree.delete(job["iid"])
                self._queue.remove(job)
        if not self._queue_running:
            self._queue_status_var.set("")

    def _queue_update_job(self, job: dict) -> None:
        iid = job["iid"]
        if self._queue_tree.exists(iid):
            vals = self._queue_tree.item(iid, "values")
            self._queue_tree.item(iid, values=(*vals[:-1], job["status"]))

    def _queue_update_timing(self, iid: str, time_str: str, spi_str: str) -> None:
        """Fill the Time and s/it columns (indices 4 & 5) for a completed job."""
        if self._queue_tree.exists(iid):
            vals = list(self._queue_tree.item(iid, "values"))
            vals[4] = time_str
            vals[5] = spi_str
            self._queue_tree.item(iid, values=vals)

    def _on_queue_select(self, event=None) -> None:
        """When a completed queue row is clicked, restore its image and details."""
        sel = self._queue_tree.selection()
        if not sel:
            return
        iid = sel[0]
        job = next((j for j in self._queue if j["iid"] == iid), None)
        if job and job.get("status") == "Done" and job.get("output_path"):
            self._update_right_panel(job["output_path"], job.get("cfg"))

    def _auto_start_queue(self) -> None:
        """Start the queue loop if idle and a pending job exists.

        Called automatically after every Add-to-Queue and after every
        _set_busy(False) (e.g. model just finished loading).
        """
        if self._queue_running or self._busy:
            return
        if not any(j["status"] == "Pending" for j in self._queue):
            return
        if self.backend.model is None:
            self._queue_status_var.set("Load model to run")
            return
        self._queue_running        = True
        self._queue_stop_requested = False
        self._abort_btn.config(state="normal")
        threading.Thread(target=self._run_queue_loop, daemon=True).start()

    def _run_queue_loop(self) -> None:
        try:
            while True:
                if self._queue_stop_requested:
                    self.root.after(0, self._queue_status_var.set, "Queue stopped")
                    break
                job = next((j for j in self._queue if j["status"] == "Pending"), None)
                if job is None:
                    self.root.after(0, self._queue_status_var.set, "Queue finished")
                    break

                job["status"] = "Running"
                self.root.after(0, self._queue_update_job, job)
                cfg = job["cfg"]
                remaining = sum(1 for j in self._queue if j["status"] == "Pending")
                prompt_short = cfg.get("prompt", "")[:28].replace("\n", " ")
                status_str = f"{prompt_short}…" + (f"  ({remaining} more)" if remaining else "")
                self.root.after(0, self._queue_status_var.set, status_str)

                loras = [e for e in cfg.get("loras", [])
                         if e.get("enabled", True) and e.get("path", "").strip()]
                if not self.backend.loras_current(loras):
                    self.backend.remove_loras()
                    if loras:
                        self.backend.apply_loras(
                            loras,
                            on_status=lambda s: self.root.after(0, self._lora_status_var.set, s),
                        )
                else:
                    self.root.after(0, self._lora_status_var.set,
                                    f"LoRAs unchanged ({len(loras)} active)")

                total = self._get_total_steps(cfg)
                self.root.after(0, lambda t=total: (
                    self._progress.configure(maximum=t, value=0),
                    self._step_label_var.set(""),
                ))

                done_path = [None]
                done_seed = [None]
                error_msg = [None]

                _t0      = [None]   # time of first callback
                _s0      = [None]   # step value of first callback

                def _prog(step, t):
                    now = time.monotonic()
                    if _t0[0] is None:
                        _t0[0] = now
                        _s0[0] = step
                        label = f"{step}/{t}"
                    else:
                        steps_done = step - _s0[0]
                        if steps_done > 0:
                            spi = (now - _t0[0]) / steps_done
                            remaining = int((t - step) * spi)
                            m, s_rem = divmod(remaining, 60)
                            label = f"{step}/{t}  {spi:.1f}s/it  ETA {m}:{s_rem:02d}"
                        else:
                            label = f"{step}/{t}"
                    self.root.after(0, lambda s=step, tt=t, lbl=label: (
                        self._progress.configure(value=s),
                        self._step_label_var.set(lbl),
                    ))
                def _done(path, actual_seed=None):
                    done_path[0] = path
                    done_seed[0] = actual_seed
                def _error(msg):
                    error_msg[0] = msg

                _sample_t0 = time.monotonic()
                self.backend.sample(cfg, _prog, _done, _error)
                _sample_elapsed = time.monotonic() - _sample_t0

                if error_msg[0]:
                    job["status"] = "Error"
                    self.root.after(0, self._queue_status_var.set,
                                    f"Error: {error_msg[0][:60]}")
                elif done_path[0] is None:
                    job["status"] = "Aborted"
                    self._queue_stop_requested = True
                else:
                    job["status"] = "Done"
                    job["output_path"] = done_path[0]
                    # Patch the actual rolled seed into the cfg snapshot so
                    # Run Details shows the real seed instead of the default.
                    if done_seed[0] is not None and cfg.get("random_seed"):
                        cfg["seed"] = done_seed[0]   # in-place — job["cfg"] is this dict
                        # Also update the queue tree seed column from "rnd" → actual value
                        _iid = job["iid"]
                        def _update_tree_seed(iid=_iid, s=done_seed[0]):
                            if self._queue_tree.exists(iid):
                                vals = list(self._queue_tree.item(iid, "values"))
                                vals[3] = str(s)
                                self._queue_tree.item(iid, values=vals)
                        self.root.after(0, _update_tree_seed)
                    # Post elapsed time and average s/it to the tree entry.
                    _m, _s_rem = divmod(int(_sample_elapsed), 60)
                    _time_str = f"{_m}:{_s_rem:02d}" if _m else f"{int(_sample_elapsed)}s"
                    _spi_str  = f"{_sample_elapsed / total:.1f}" if total > 0 else ""
                    self.root.after(0, self._queue_update_timing,
                                    job["iid"], _time_str, _spi_str)
                    self.root.after(0, self._on_queue_job_done, done_path[0], cfg)

                self.root.after(0, self._queue_update_job, job)
                if job["status"] == "Aborted":
                    self.root.after(0, self._queue_status_var.set, "Queue stopped")
                    break
        finally:
            self._queue_running = False
            self.root.after(0, self._abort_btn.config, {"state": "disabled"})
            self.root.after(0, self._step_label_var.set, "")
            self.root.after(0, self._progress.configure, {"value": 0})

    def _on_queue_job_done(self, path: str, cfg: dict | None = None) -> None:
        self._last_output = path
        self._out_path_var.set(path)
        self._progress["value"] = self._progress["maximum"]
        self._update_right_panel(path, cfg)

    # ==================================================================
    # Token counter
    # ==================================================================

    def _schedule_token_count(self, *_) -> None:
        if hasattr(self, "_token_after_id"):
            self.root.after_cancel(self._token_after_id)
        self._token_after_id = self.root.after(1500, self._update_token_count)

    def _update_token_count(self) -> None:
        prompt = self._prompt_text.get("1.0", "end-1c")
        if not prompt.strip():
            self._token_count_var.set("")
            self._token_label.config(foreground="gray")
            return
        # Run tokenizer lookup off the main thread — from_pretrained can block on
        # disk/network I/O (HF hub download on first use without a loaded model).
        def _count():
            try:
                if self.backend.model is not None and \
                        getattr(self.backend.model, "tokenizer", None) is not None:
                    tokenizer = self.backend.model.tokenizer
                else:
                    from transformers import T5Tokenizer
                    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
                count = len(tokenizer(prompt, truncation=False).input_ids)
                limit = 512
                color = "#cc3333" if count > limit else "gray"
                self.root.after(0, self._token_count_var.set, f"{count}/{limit} tok")
                self.root.after(0, self._token_label.config, {"foreground": color})
            except Exception:
                self.root.after(0, self._token_count_var.set, "")
        threading.Thread(target=_count, daemon=True).start()

    # ==================================================================
    # Shared misc handlers
    # ==================================================================

    def _refresh_attn_avail(self) -> None:
        avail = check_attn_backends()
        parts = [f"{n}:{'✓' if avail.get(n) else '✗'}" for n in ("Flash", "SageAttn")]
        self._attn_avail_var.set("  ".join(parts))

    def _on_dtype_changed(self, *_) -> None:
        is_gguf = (self._dtype_var.get() == "GGUF")
        if is_gguf:
            self._gguf_frame.grid()
            self._svd_var.set(False)
            for w in self._svd_row_widgets:
                try:
                    w.config(state="disabled")
                except Exception:
                    pass
        else:
            self._gguf_frame.grid_remove()
            for w in self._svd_row_widgets:
                try:
                    w.config(state="normal")
                except Exception:
                    pass

    def _browse_model(self) -> None:
        path = filedialog.askdirectory(title="Select base model directory")
        if path:
            self._base_model_var.set(path)
            self._save_cfg()

    def _browse_gguf(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select GGUF file",
            filetypes=[("GGUF / Safetensors", "*.gguf *.safetensors"), ("All files", "*.*")],
        )
        if path:
            var.set(path)
            self._save_cfg()

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self._outdir_var.set(path)
            self._save_cfg()

    def _browse_cache_dir(self) -> None:
        path = filedialog.askdirectory(title="Select quantization cache directory")
        if path:
            self._cache_var.set(path)
            self._save_cfg()

    def _use_ot_cache_dir(self) -> None:
        path = find_ot_quant_cache()
        if path:
            self._cache_var.set(path)
            self._save_cfg()
        else:
            messagebox.showinfo(
                "Cache not found",
                "Could not read cache_dir from OneTrainer/config.json.\n"
                "Open a training config in OT first, or set the path manually.",
            )

    def _use_workspace_dir(self) -> None:
        ws = find_ot_workspace()
        if ws:
            self._outdir_var.set(ws)
            self._save_cfg()
        else:
            messagebox.showinfo(
                "Workspace not found",
                "Could not read workspace_dir from OneTrainer/config.json.\n"
                "Open a training config in OT first.",
            )

    def _open_output(self) -> None:
        if self._last_output and os.path.isfile(self._last_output):
            os.startfile(self._last_output)

    def _open_folder(self) -> None:
        outdir = self._outdir_var.get()
        if os.path.isdir(outdir):
            os.startfile(outdir)

    def _save_cfg(self) -> None:
        try:
            save_config(self._collect_cfg(), self._config_path)
        except Exception:
            pass

    def cleanup(self) -> None:
        """Save config, stop queue, cancel backend.  Called by launcher on model switch."""
        self._queue_stop_requested = True
        self.backend.cancel()
        self._stop_blink()
        self._save_cfg()

    def _on_close(self) -> None:
        self.cleanup()
        self.root.destroy()

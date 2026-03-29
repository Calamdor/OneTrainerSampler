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
from tkinter.scrolledtext import ScrolledText

from sampler_core.util.config import load_config, save_config
from sampler_core.util.ot_bridge import find_ot_workspace, find_ot_quant_cache
from sampler_core.util.resolution import check_attn_backends
from sampler_core.gui.tooltip import Tooltip  # noqa: F401 — re-exported for subclasses
from sampler_core.gui.theme import apply_dark_theme, BG, BG_INPUT, FG, BG_WIDGET, FG_DIM

# Optional drag-and-drop support (requires tkinterdnd2 and a TkinterDnD.Tk root).
try:
    import tkinterdnd2 as _tkdnd
    _DND_AVAILABLE = True
except ImportError:
    _tkdnd = None
    _DND_AVAILABLE = False


def _load_video_frames(path: str) -> tuple[list, float]:
    """
    Decode all frames of an MP4 as PIL Images.  Returns (frames, fps).
    Tries PyAV first, then imageio (ffmpeg plugin), then cv2.
    Called from a background thread — no tkinter calls inside.
    """
    from PIL import Image

    # ---- PyAV -----------------------------------------------------------
    try:
        import av
        container = av.open(path)
        stream    = container.streams.video[0]
        fps       = float(stream.average_rate) if stream.average_rate else 24.0
        frames    = [f.to_image() for f in container.decode(video=0)]
        container.close()
        if frames:
            return frames, fps
    except Exception:
        pass

    # ---- imageio / ffmpeg -----------------------------------------------
    try:
        import imageio
        reader = imageio.get_reader(path, plugin="ffmpeg")
        fps    = float(reader.get_meta_data().get("fps", 24.0))
        frames = [Image.fromarray(f) for f in reader]
        reader.close()
        if frames:
            return frames, fps
    except Exception:
        pass

    # ---- cv2 fallback ---------------------------------------------------
    try:
        import cv2
        cap    = cv2.VideoCapture(path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        if frames:
            return frames, float(fps)
    except Exception:
        pass

    return [], 24.0


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
        self._blink_id: str | None = None
        self._blink_state = False

        # Session log
        self._log_lines: list[str] = []
        self._log_win:  tk.Toplevel | None = None
        self._log_text: ScrolledText | None = None
        
        # Redirect print() to log window for debug output
        import sys
        class _LogRedirect:
            def __init__(self, logger):
                self.logger = logger
            def write(self, msg):
                if msg.strip():
                    self.logger._append_log(msg.rstrip())
            def flush(self):
                pass
        
        # Save original stdout/stderr for error traceback printing
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        
        sys.stdout = _LogRedirect(self)
        sys.stderr = _LogRedirect(self)

        # Prompt Library window (singleton)
        self._lib_win: tk.Toplevel | None = None

        self._queue: list[dict] = []
        self._queue_running        = False
        self._queue_stop_requested = False

        # Video player state
        self._video_frames:    list      = []
        self._video_frame_idx: int       = 0
        self._video_fps:       float     = 24.0
        self._video_playing:   bool      = False
        self._video_after_id:  str | None = None

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
                "T5 token count shown to the right.\n"
                "There is no hard limit — VRAM is the only constraint.")

        self._token_count_var = tk.StringVar(value="")
        self._token_label = ttk.Label(
            gen_frame, textvariable=self._token_count_var,
            foreground="gray", font=("TkDefaultFont", 8), width=10, anchor="nw",
        )
        self._token_label.grid(row=r, column=6, sticky="nw", padx=(0, 6))
        Tooltip(self._token_label,
                "T5 token count for the positive prompt.\n\n"
                "Counted using the loaded model's tokenizer;\n"
                "falls back to downloading the T5 vocab if no model is loaded.")

        self._lib_btn = ttk.Button(gen_frame, text="Library…", width=9,
                                   command=self._open_library_window)
        self._lib_btn.grid(row=r, column=7, sticky="nw", padx=(0, 6))
        Tooltip(self._lib_btn,
                "Open the Prompt Library window.\n\n"
                "Save, browse, search, and reuse positive/negative prompt pairs.\n"
                "Entries can be filtered by model.")

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
        lora_frame.pack(fill="both", expand=True, **pad)

        add_row = ttk.Frame(lora_frame)
        add_row.pack(fill="x", pady=2)
        _add_lora_btn = ttk.Button(add_row, text="+ Add LoRA", command=self._add_lora)
        _add_lora_btn.pack(side="left", padx=2)
        Tooltip(_add_lora_btn, "Add a LoRA (.safetensors) to the list.\n\n"
                "LoRAs are not applied until you click Apply LoRAs.\n"
                "Accepted formats: OT internal, ai-toolkit, kohya, musubi-tuner.")

        canvas_frame = ttk.Frame(lora_frame)
        canvas_frame.pack(fill="both", expand=True)
        self._lora_canvas = tk.Canvas(canvas_frame, highlightthickness=0,
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

        # Mousewheel scroll for LoRA list
        def _on_mousewheel(event):
            self._lora_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._lora_canvas.bind("<MouseWheel>", _on_mousewheel)
        self._lora_inner.bind("<MouseWheel>", _on_mousewheel)
        # Also bind to children as they're added (handled in _add_lora_row)
        self._lora_mousewheel_handler = _on_mousewheel

        self._build_lora_header(self._lora_inner)

        apply_row = ttk.Frame(lora_frame)
        apply_row.pack(fill="x", pady=2)
        _clear_btn = ttk.Button(apply_row, text="Clear LoRAs", command=self._clear_loras)
        _clear_btn.pack(side="left", padx=2)
        Tooltip(_clear_btn, "Remove all applied LoRA hooks from the model.\n"
                "The model reverts to its base weights. No reload needed.\n\n"
                "LoRAs are applied automatically by the queue before each job.")
        self._lora_status_var = tk.StringVar(value="No LoRAs applied")
        self._lora_status_var.trace_add("write",
            lambda *_: self._append_log(f"[lora]  {self._lora_status_var.get()}"))
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
            btn_row, text="＋ Add to Queue", command=self._queue_add,
            style="Accent.TButton")
        self._add_queue_btn.pack(side="left", padx=2)
        Tooltip(self._add_queue_btn,
                "Snapshot the current settings and add a job to the queue.\n\n"
                "The model loads automatically when the job runs.\n"
                "Consecutive jobs with the same model settings skip the reload.\n"
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
        _log_btn = ttk.Button(btn_row, text="Log", command=self._open_log_window)
        _log_btn.pack(side="right", padx=2)
        Tooltip(_log_btn, "Open the session log window.\n\n"
                "Shows all model load, LoRA, and queue status messages\n"
                "with timestamps. Useful for diagnosing errors.")

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
        self._queue_status_var.trace_add("write",
            lambda *_: self._append_log(f"[queue] {self._queue_status_var.get()}"))
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

        # Vertical paned window so preview and details both stay visible
        # regardless of image aspect ratio.  User can drag the sash.
        _vpaned = tk.PanedWindow(
            self._right_frame, orient=tk.VERTICAL,
            sashwidth=5, sashrelief="flat",
            background=BG, bd=0,
        )
        _vpaned.pack(fill="both", expand=True)

        # ---- Preview -----------------------------------------------
        preview_frame = ttk.LabelFrame(_vpaned, text="Preview")
        _vpaned.add(preview_frame, stretch="always", minsize=120,
                    padx=pad["padx"], pady=pad["pady"])

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
        # Load File / DnD toolbar — packed first so it appears above the image
        _lf_row = tk.Frame(self._preview_container, background=BG)
        _lf_row.pack(side="top", fill="x")
        _lf_btn = ttk.Button(_lf_row, text="Load File…",
                             command=self._load_file_dialog)
        _lf_btn.pack(side="left", padx=4, pady=2)
        Tooltip(_lf_btn,
                "Load a PNG, MP4, or ComfyUI JSON to preview and extract prompts.\n\n"
                "Supports: OneTrainer sampler outputs (our metadata), ComfyUI PNGs,\n"
                "and ComfyUI JSON workflow files.\n"
                "Drag-and-drop also works if tkinterdnd2 is installed.")
        if not _DND_AVAILABLE:
            tk.Label(
                _lf_row,
                text="(install tkinterdnd2 for drag-and-drop)",
                background=BG, foreground=FG_DIM,
                font=("TkDefaultFont", 7),
            ).pack(side="left", padx=(0, 4))

        self._preview_label.pack(fill="both", expand=True)
        self._preview_label.bind("<Button-1>", lambda _: self._open_output())

        self._preview_pil_img = None   # PIL.Image kept for rescaling
        self._preview_photo   = None   # ImageTk reference kept to prevent GC

        self._preview_container.bind("<Configure>", self._on_preview_resize)

        # Drag-and-drop registration (requires TkinterDnD.Tk root)
        if _DND_AVAILABLE:
            self._preview_container.drop_target_register(_tkdnd.DND_FILES)
            self._preview_container.dnd_bind("<<Drop>>", self._on_file_drop)
            self._preview_label.drop_target_register(_tkdnd.DND_FILES)
            self._preview_label.dnd_bind("<<Drop>>", self._on_file_drop)

        # ---- Video player controls (hidden until a video is loaded) ----
        self._video_ctrl_bar = tk.Frame(preview_frame, background=BG)
        self._video_ctrl_bar.pack(fill="x", padx=4, pady=(0, 4))

        self._video_play_btn = tk.Button(
            self._video_ctrl_bar,
            text="⏸ Pause", width=8,
            command=self._toggle_video_playback,
            background="#2a4a6a", foreground="white",
            activebackground="#3a6a8a", activeforeground="white",
            relief="flat", padx=4, pady=2,
        )
        self._video_play_btn.pack(side="left", padx=(0, 8))

        self._video_frame_label_var = tk.StringVar(value="")
        tk.Label(
            self._video_ctrl_bar,
            textvariable=self._video_frame_label_var,
            background=BG, foreground=FG_DIM,
            font=("TkDefaultFont", 8),
        ).pack(side="left")

        self._video_ctrl_bar.pack_forget()   # shown only for video output

        # ---- Run details -------------------------------------------
        details_frame = ttk.LabelFrame(_vpaned, text="Run Details")
        _vpaned.add(details_frame, stretch="never", minsize=210,
                    padx=pad["padx"], pady=pad["pady"])

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

        r += 1
        use_row = ttk.Frame(details_frame)
        use_row.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 4))
        self._use_pos_btn = ttk.Button(use_row, text="← Use Positive",
                                       command=self._use_detail_positive,
                                       state="disabled")
        self._use_pos_btn.pack(side="left", padx=(0, 4))
        Tooltip(self._use_pos_btn,
                "Copy the positive prompt shown in Run Details\nto the main Prompt field.")
        self._use_neg_btn = ttk.Button(use_row, text="← Use Negative",
                                       command=self._use_detail_negative,
                                       state="disabled")
        self._use_neg_btn.pack(side="left", padx=(0, 4))
        Tooltip(self._use_neg_btn,
                "Copy the negative prompt shown in Run Details\nto the main Negative field.")
        self._use_both_btn = ttk.Button(use_row, text="← Use Both",
                                        command=self._use_detail_both,
                                        state="disabled")
        self._use_both_btn.pack(side="left")
        Tooltip(self._use_both_btn,
                "Copy both prompts from Run Details to the main prompt fields.")

    # ------------------------------------------------------------------
    def _on_preview_resize(self, event=None) -> None:
        if hasattr(self, "_preview_resize_id"):
            self.root.after_cancel(self._preview_resize_id)
        self._preview_resize_id = self.root.after(80, self._redraw_preview)

    def _redraw_preview(self) -> None:
        if self._video_playing and self._video_frames:
            return   # video tick handles its own redraws
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

    # ------------------------------------------------------------------
    # Video playback helpers
    # ------------------------------------------------------------------

    def _start_video_playback(self, frames: list, fps: float) -> None:
        """Called on the main thread after background frame loading completes."""
        if not frames:
            self._preview_label.config(image="", text="Could not read video")
            return
        self._video_frames    = frames
        self._video_frame_idx = 0
        self._video_fps       = max(1.0, fps)
        self._video_playing   = True
        self._video_play_btn.config(text="⏸ Pause")
        self._video_tick()

    def _video_tick(self) -> None:
        """Advance one frame and schedule the next tick."""
        if not self._video_playing or not self._video_frames:
            return
        frame = self._video_frames[self._video_frame_idx]
        total = len(self._video_frames)
        w = self._preview_container.winfo_width() - 8
        h = self._preview_container.winfo_height() - 8
        if w > 10 and h > 10:
            try:
                from PIL import Image, ImageTk
                img = frame.copy()
                img.thumbnail((w, h), Image.LANCZOS)
                self._preview_photo = ImageTk.PhotoImage(img)
                self._preview_label.config(image=self._preview_photo, text="")
            except Exception:
                pass
        self._video_frame_label_var.set(
            f"{self._video_frame_idx + 1} / {total}  ({self._video_fps:.1f} fps)")
        self._video_frame_idx = (self._video_frame_idx + 1) % total
        delay_ms = max(1, int(1000.0 / self._video_fps))
        self._video_after_id = self.root.after(delay_ms, self._video_tick)

    def _stop_video_playback(self) -> None:
        """Cancel animation loop and reset state."""
        self._video_playing = False
        if self._video_after_id is not None:
            self.root.after_cancel(self._video_after_id)
            self._video_after_id = None
        self._video_frames    = []
        self._video_frame_idx = 0

    def _toggle_video_playback(self) -> None:
        if not self._video_frames:
            return
        if self._video_playing:
            self._video_playing = False
            if self._video_after_id is not None:
                self.root.after_cancel(self._video_after_id)
                self._video_after_id = None
            self._video_play_btn.config(text="▶ Play")
        else:
            self._video_playing = True
            self._video_play_btn.config(text="⏸ Pause")
            self._video_tick()

    def _update_right_panel(self, path: str, cfg: dict | None = None) -> None:
        """Load the finished output into the preview and populate run details."""
        self._stop_video_playback()

        if path and path.lower().endswith(".mp4"):
            # Hide static image, show controls bar, load frames in background
            self._preview_pil_img = None
            self._preview_label.config(image="", text="Loading video …")
            self._video_ctrl_bar.pack(fill="x", padx=4, pady=(0, 4))

            def _load():
                frames, fps = _load_video_frames(path)
                self.root.after(0, self._start_video_playback, frames, fps)

            threading.Thread(target=_load, daemon=True).start()
        else:
            # Static image — hide controls bar
            self._video_ctrl_bar.pack_forget()
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

        if "steps_high" in cfg:
            sh = cfg.get("steps_high", 0)
            sl = cfg.get("steps_low", 0)
            steps_str = f"{sh}+{sl}" if sl else str(sh)
        else:
            steps_str = str(cfg.get("steps", "—"))
        self._detail_vars["steps"].set(steps_str)
        cfg_val = cfg.get("cfg_scale", "—")
        cfg2    = cfg.get("cfg_scale_2")
        self._detail_vars["cfg"].set(f"{cfg_val}/{cfg2}" if cfg2 is not None else str(cfg_val))

        sched = cfg.get("scheduler", "")
        shift = cfg.get("sigma_shift", None)
        if sched and shift is not None:
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

        has_prompt = bool(
            cfg.get("prompt", "").strip() or cfg.get("negative_prompt", "").strip()
        )
        _state = "normal" if has_prompt else "disabled"
        for btn in (self._use_pos_btn, self._use_neg_btn, self._use_both_btn):
            btn.config(state=_state)

    # ------------------------------------------------------------------
    # Prompt Library
    # ------------------------------------------------------------------

    def _open_library_window(self) -> None:
        """Open (or raise) the Prompt Library window."""
        if self._lib_win is not None:
            try:
                self._lib_win.lift()
                self._lib_win.focus_force()
                return
            except tk.TclError:
                pass
        from sampler_core.gui.prompt_library import PromptLibraryWindow
        lib = PromptLibraryWindow(self.root, self)
        self._lib_win = lib.window

    # ------------------------------------------------------------------
    # File load / drag-and-drop
    # ------------------------------------------------------------------

    def _on_file_drop(self, event) -> None:
        """Handle a tkinterdnd2 ``<<Drop>>`` event on the preview panel."""
        import re
        # tkinterdnd2 wraps paths with spaces in {braces}; multiple files are
        # space-separated.  We process only the first dropped file.
        raw = event.data.strip()
        paths = re.findall(r'\{[^}]+\}|[^\s]+', raw)
        paths = [p.strip("{}") for p in paths]
        if paths:
            self._load_file(paths[0])

    def _load_file_dialog(self) -> None:
        """Open a file-chooser dialog and load the selected file."""
        path = filedialog.askopenfilename(
            title="Load file for preview",
            filetypes=[
                ("Supported files", "*.png *.mp4 *.json"),
                ("PNG images",      "*.png"),
                ("MP4 video",       "*.mp4"),
                ("ComfyUI JSON",    "*.json"),
                ("All files",       "*.*"),
            ],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        """Parse *path* and display its content in the preview + run details."""
        from sampler_core.util.file_import import load_sampler_file
        result = load_sampler_file(path)
        if result is None:
            messagebox.showwarning(
                "Unsupported file",
                f"Could not read metadata from:\n{path}",
            )
            return
        display = result.get("display_path")
        params  = result.get("params")
        if display:
            self._update_right_panel(display, params)
        elif params:
            # JSON-only (ComfyUI workflow — no image to display)
            self._stop_video_playback()
            self._video_ctrl_bar.pack_forget()
            self._preview_pil_img = None
            self._preview_label.config(image="", text="(JSON workflow — no image)")
            self._populate_detail_from_params(params)

    def _populate_detail_from_params(self, params: dict) -> None:
        """Fill the Run Details panel from a params dict (no file path needed)."""
        self._detail_vars["file"].set("(imported)")
        self._detail_vars["seed"].set(str(params.get("seed", "—")))
        w, h = params.get("width", 0), params.get("height", 0)
        self._detail_vars["res"].set(f"{w} × {h}" if w and h else "—")
        steps = params.get("steps", "—")
        self._detail_vars["steps"].set(str(steps))
        self._detail_vars["cfg"].set(str(params.get("cfg_scale", "—")))
        sched = params.get("scheduler", "—")
        shift = params.get("sigma_shift")
        self._detail_vars["sampler"].set(
            f"{sched}  σ={shift}" if sched and shift is not None else str(sched)
        )
        for widget, key in (
            (self._detail_prompt_text, "prompt"),
            (self._detail_neg_text,    "negative_prompt"),
        ):
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", params.get(key, ""))
            widget.config(state="disabled")
        has_prompt = bool(
            params.get("prompt", "").strip() or params.get("negative_prompt", "").strip()
        )
        _state = "normal" if has_prompt else "disabled"
        for btn in (self._use_pos_btn, self._use_neg_btn, self._use_both_btn):
            btn.config(state=_state)

    # ------------------------------------------------------------------
    # Use-prompt helpers (copy Run Details → main prompt fields)
    # ------------------------------------------------------------------

    def _use_detail_positive(self) -> None:
        text = self._detail_prompt_text.get("1.0", "end-1c")
        self._prompt_text.delete("1.0", "end")
        self._prompt_text.insert("1.0", text)
        self._schedule_token_count()

    def _use_detail_negative(self) -> None:
        text = self._detail_neg_text.get("1.0", "end-1c")
        self._neg_text.delete("1.0", "end")
        self._neg_text.insert("1.0", text)

    def _use_detail_both(self) -> None:
        self._use_detail_positive()
        self._use_detail_negative()

    # ==================================================================
    # Shared blink / status / busy
    # ==================================================================

    def _set_model_status(self, text: str, warn: bool = False) -> None:
        self._stop_blink()
        self._model_status_var.set(text)
        self._append_log(f"[model] {text}")
        if warn:
            self._start_blink()
        else:
            self._model_status_label.config(foreground="gray")

    # ==================================================================
    # Session log
    # ==================================================================

    def _append_log(self, msg: str) -> None:
        if not msg:
            return
        ts   = time.strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        self._log_lines.append(line)
        if self._log_text is not None:
            try:
                self._log_text.configure(state="normal")
                self._log_text.insert("end", line + "\n")
                self._log_text.configure(state="disabled")
                self._log_text.see("end")
            except tk.TclError:
                pass  # window was destroyed between check and write

    def _open_log_window(self) -> None:
        if self._log_win is not None:
            try:
                self._log_win.lift()
                self._log_win.focus_force()
                return
            except tk.TclError:
                pass  # window was destroyed; recreate below

        win = tk.Toplevel(self.root)
        win.title("Session Log")
        win.geometry("900x380")
        win.configure(background="#1e1e1e")

        btn_row = tk.Frame(win, background="#1e1e1e")
        btn_row.pack(fill="x", padx=4, pady=4)

        def _clear_log():
            self._log_lines.clear()
            if self._log_text is not None:
                self._log_text.configure(state="normal")
                self._log_text.delete("1.0", "end")
                self._log_text.configure(state="disabled")

        def _copy_log():
            win.clipboard_clear()
            win.clipboard_append("\n".join(self._log_lines))

        def _dump_dynamo():
            try:
                import torch._dynamo.utils as _du
                counters = dict(_du.counters)
                self._append_log("── dynamo counters ──────────────────")
                if counters:
                    for k, v in sorted(counters.items()):
                        if isinstance(v, dict):
                            for k2, v2 in sorted(v.items()):
                                self._append_log(f"[dynamo] {k}/{k2}: {v2}")
                        else:
                            self._append_log(f"[dynamo] {k}: {v}")
                else:
                    self._append_log("[dynamo] no counters (no compiled graphs yet?)")
            except Exception as exc:
                self._append_log(f"[dynamo] error reading counters: {exc}")
            try:
                import torch._dynamo.utils as _du
                self._append_log(f"[dynamo] compile_id={getattr(_du, 'compile_id', 'n/a')}")
            except Exception:
                pass
            self._append_log("─────────────────────────────────────")

        ttk.Button(btn_row, text="Clear",        command=_clear_log).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Copy All",     command=_copy_log).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Dynamo Stats", command=_dump_dynamo).pack(side="left", padx=2)

        txt = ScrolledText(
            win, state="disabled", wrap="word",
            background="#1e1e1e", foreground="#cccccc",
            font=("Consolas", 9), insertbackground="#cccccc",
            relief="flat", borderwidth=0,
        )
        txt.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        txt.configure(state="normal")
        for line in self._log_lines:
            txt.insert("end", line + "\n")
        txt.configure(state="disabled")
        txt.see("end")

        self._log_win  = win
        self._log_text = txt

        def _on_close():
            self._log_win  = None
            self._log_text = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

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

    # ==================================================================
    # Shared LoRA panel logic
    # ==================================================================

    def _bind_mousewheel_recursive(self, widget) -> None:
        """Bind mousewheel scroll to a widget and all its descendants."""
        handler = getattr(self, '_lora_mousewheel_handler', None)
        if handler is None:
            return
        widget.bind("<MouseWheel>", handler)
        for child in widget.winfo_children():
            self._bind_mousewheel_recursive(child)

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

    def _clear_loras(self) -> None:
        if self._queue_running:
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
        """Start the queue loop if idle and a pending job exists."""
        if self._queue_running:
            return
        if not any(j["status"] == "Pending" for j in self._queue):
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
                cfg = job["cfg"]
                self.root.after(0, self._queue_update_job, job)

                self.backend._cancel_event.clear()

                # ── lazy model load ───────────────────────────────────────────
                if self.backend.model_needs_reload(cfg):
                    self.root.after(0, self._queue_status_var.set, "Loading model …")
                    try:
                        self.backend.load_model(
                            cfg,
                            on_status=lambda s: self.root.after(
                                0, self._set_model_status, s),
                        )
                    except Exception as exc:
                        job["status"] = "Error"
                        self.root.after(0, self._queue_update_job, job)
                        self.root.after(0, self._queue_status_var.set,
                                        f"Load error: {str(exc)[:80]}")
                        self.root.after(0, self._set_model_status,
                                        f"Error: {exc}")
                        continue

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
                _tlast   = [None]   # time of previous callback (per-step timing)
                _baseline = [None]  # median step time after warmup, for recompile detection
                # Wan expert transition: step where high→low expert switch
                # triggers lazy compilation of the second transformer.
                _steps_high = cfg.get("steps_high") if cfg.get("steps_low", 0) > 0 else None

                def _prog(step, t):
                    now = time.monotonic()
                    if _t0[0] is None:
                        _t0[0]    = now
                        _s0[0]    = step
                        _tlast[0] = now
                        label = f"{step}/{t}"
                        self.root.after(0, self._append_log,
                                        f"[step]  {step}/{t}  — first step (compile warmup)")
                    else:
                        step_dt = now - _tlast[0]
                        _tlast[0] = now
                        steps_done = step - _s0[0]
                        if steps_done > 0:
                            spi = (now - _t0[0]) / steps_done
                            remaining = int((t - step) * spi)
                            m, s_rem = divmod(remaining, 60)
                            rate_str = (f"{1/spi:.2f}it/s" if spi < 1
                                        else f"{spi:.1f}s/it")
                            label = f"{step}/{t}  {rate_str}  ETA {m}:{s_rem:02d}"
                        else:
                            label = f"{step}/{t}"
                        # Establish baseline from first post-warmup step (step 2+).
                        # Flag only when the current step is >2× the baseline —
                        # this detects genuine recompile events without triggering
                        # on normal GGUF / quantised-model operation speed.
                        if steps_done == 1:
                            _baseline[0] = step_dt   # first steady-state sample
                        flag = ""
                        if (_baseline[0] is not None and steps_done > 1
                                and step_dt > _baseline[0] * 2.0):
                            if _steps_high is not None and step == _steps_high + 1:
                                flag = "  — expert transition (compile warmup)"
                            else:
                                flag = "  *** SLOW — possible recompile"
                        self.root.after(0, self._append_log,
                                        f"[step]  {step}/{t}  {step_dt:.2f}s{flag}")
                    self.root.after(0, lambda s=step, tt=t, lbl=label: (
                        self._progress.configure(value=s),
                        self._step_label_var.set(lbl),
                    ))
                def _done(path, actual_seed=None):
                    done_path[0] = path
                    done_seed[0] = actual_seed
                def _error(msg):
                    error_msg[0] = msg
                    self.root.after(0, self._append_log, f"[error] {msg}")

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
                    if total > 0:
                        _spi = _sample_elapsed / total
                        _spi_str = (f"{1/_spi:.2f}it/s" if _spi < 1
                                    else f"{_spi:.1f}s/it")
                    else:
                        _spi_str = ""
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
        if cfg and (cfg.get("prompt", "").strip() or cfg.get("negative_prompt", "").strip()):
            cls = type(self).__name__
            model = ("Wan 2.2 T2V-A14B" if "Wan" in cls
                     else "Chroma" if "Chroma" in cls else "All")
            try:
                from sampler_core.gui.prompt_library import auto_save_prompt
                auto_save_prompt(cfg.get("prompt", ""), cfg.get("negative_prompt", ""), model)
            except Exception:
                pass

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
                self.root.after(0, self._token_count_var.set, f"{count} tok")
                self.root.after(0, self._token_label.config, {"foreground": "gray"})
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
        is_gguf = self._dtype_var.get() in {"GGUF", "GGUF_A8I", "GGUF_A8F"}
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
        self._stop_video_playback()
        if self._lib_win is not None:
            try:
                self._lib_win.destroy()
            except tk.TclError:
                pass
            self._lib_win = None
        self._save_cfg()

    def _on_close(self) -> None:
        self.cleanup()
        self.root.destroy()

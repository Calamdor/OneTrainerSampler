"""Prompt Library — persistent history and browser for prompt pairs.

Every completed generation is auto-saved here.  The window groups entries
by date with collapsible sections, and shows a single prompt at a time
(positive or negative) via a toggle button.

Public API used by the rest of the sampler
-------------------------------------------
``auto_save_prompt(positive, negative, model)``
    Save a prompt pair from a completed generation.  Safe to call whether
    or not the library window is currently open.

``PromptLibraryWindow(root, app)``
    Open (or reuse) the library window.  Exposes ``.window`` (the Toplevel)
    for singleton management in BaseSamplerApp.
"""
from __future__ import annotations

import datetime
import json
import os
import tkinter as tk
import uuid
from collections import defaultdict
from tkinter import ttk

from sampler_core.gui.theme import (
    BG, BG_INPUT, BLUE, BLUE_SEL, BORDER, FG,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = os.path.dirname(os.path.abspath(__file__))
_SAMPLERS_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))
_LIBRARY_PATH = os.path.join(_SAMPLERS_DIR, "config", "prompt_library.json")

_MODEL_OPTIONS = ["All", "Chroma", "Wan 2.2 T2V-A14B"]

# Module-level reference to the open window (for live refresh on auto-save).
_open_window: "PromptLibraryWindow | None" = None


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def load_library() -> dict:
    """Load ``prompt_library.json``.  Returns ``{"entries": []}`` on failure."""
    try:
        with open(_LIBRARY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    except Exception:
        pass
    return {"entries": []}


def save_library(data: dict) -> None:
    """Write *data* to ``prompt_library.json``."""
    os.makedirs(os.path.dirname(_LIBRARY_PATH), exist_ok=True)
    with open(_LIBRARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def auto_save_prompt(positive: str, negative: str, model: str) -> None:
    """Append a prompt pair to the library after a successful generation.

    Safe to call from any thread; the library window refresh is scheduled on
    the tkinter main thread if the window happens to be open.
    """
    global _open_window
    if not positive.strip() and not negative.strip():
        return

    snippet = positive.strip()[:50].replace("\n", " ")
    entry = {
        "id":       uuid.uuid4().hex,
        "label":    snippet or "Auto",
        "model":    model,
        "positive": positive,
        "negative": negative,
        "created":  datetime.datetime.now().isoformat(timespec="seconds"),
        "source":   "auto",
    }
    data = load_library()
    entries = data.setdefault("entries", [])
    if any(e.get("positive") == positive and e.get("negative") == negative
           for e in entries):
        return
    entries.append(entry)
    save_library(data)

    if _open_window is not None:
        try:
            # Schedule GUI update on the main thread
            _open_window._data = data
            _open_window.window.after(0, _open_window._refresh_list)
        except Exception:
            _open_window = None


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _date_key(created: str) -> str:
    """Return ``YYYY-MM-DD`` sort key from an ISO datetime string."""
    try:
        return datetime.datetime.fromisoformat(created).strftime("%Y-%m-%d")
    except Exception:
        return "0000-00-00"


def _date_label(key: str) -> str:
    """Human-readable label for a date key."""
    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    try:
        d = datetime.date.fromisoformat(key)
        if d == today:
            return "Today"
        if d == yesterday:
            return "Yesterday"
        return key
    except Exception:
        return key


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class PromptLibraryWindow:
    """Non-modal Toplevel showing the prompt history grouped by date."""

    def __init__(self, root: tk.Tk, app) -> None:
        global _open_window
        self._root = root
        self._app  = app

        self._data: dict           = load_library()
        self._selected_id: str | None = None
        self._show_positive: bool  = True   # toggle state
        self._prev_open_groups: set[str] = set()  # remember collapsed groups

        self.window = tk.Toplevel(root)
        self.window.title("Prompt Library")
        self.window.geometry("920x560")
        self.window.minsize(680, 400)
        self.window.configure(background=BG)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        _open_window = self

        self._build_ui()
        self._refresh_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Top toolbar ──────────────────────────────────────────────
        toolbar = tk.Frame(self.window, background=BG)
        toolbar.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(toolbar, text="Search:", background=BG, foreground=FG).pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh_list())
        _se = tk.Entry(
            toolbar, textvariable=self._search_var, width=26,
            background=BG_INPUT, foreground=FG, insertbackground=FG,
            relief="flat", borderwidth=1,
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=BLUE,
        )
        _se.pack(side="left", padx=(4, 12))

        tk.Label(toolbar, text="Model:", background=BG, foreground=FG).pack(side="left")
        self._model_filter_var = tk.StringVar(value="All")
        self._model_filter_var.trace_add("write", lambda *_: self._refresh_list())
        ttk.Combobox(
            toolbar, textvariable=self._model_filter_var,
            values=_MODEL_OPTIONS, state="readonly", width=18,
        ).pack(side="left", padx=(4, 0))

        # ── Paned window ─────────────────────────────────────────────
        paned = tk.PanedWindow(
            self.window, orient="horizontal",
            background=BG, sashwidth=5, sashrelief="flat", bd=0,
        )
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # ── Left pane — entry list ────────────────────────────────────
        left = tk.Frame(paned, background=BG)
        paned.add(left, minsize=220, width=300)

        tree_frame = tk.Frame(left, background=BG)
        tree_frame.pack(fill="both", expand=True)

        self._tree = ttk.Treeview(
            tree_frame,
            show="tree",          # tree column only — no column headers
            selectmode="browse",
        )
        self._tree.column("#0", width=260, stretch=True)

        # Style group rows with accent colour
        self._tree.tag_configure("group", foreground=BLUE,
                                  font=("TkDefaultFont", 9, "bold"))
        self._tree.tag_configure("entry_auto",   foreground=FG)
        self._tree.tag_configure("entry_manual", foreground=FG)

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical",
                                    command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        self._tree.bind("<<TreeviewSelect>>", self._on_entry_select)

        # ── Right pane — detail view ──────────────────────────────────
        right = tk.Frame(paned, background=BG)
        paned.add(right, minsize=300)

        # Save-current button
        top_row = tk.Frame(right, background=BG)
        top_row.pack(fill="x", pady=(4, 8))
        save_btn = ttk.Button(top_row, text="Save Current Prompts",
                              command=self._save_current_prompts)
        save_btn.pack(side="left")
        _tooltip(save_btn,
                 "Manually save the current main-window prompts to the library.\n"
                 "(Prompts are also saved automatically after every generation.)")

        # Label row
        lbl_row = tk.Frame(right, background=BG)
        lbl_row.pack(fill="x", pady=(0, 4))
        tk.Label(lbl_row, text="Label:", background=BG, foreground=FG,
                 width=8, anchor="e").pack(side="left")
        self._label_var = tk.StringVar()
        self._label_entry = tk.Entry(
            lbl_row, textvariable=self._label_var, width=34,
            background=BG_INPUT, foreground=FG, insertbackground=FG,
            relief="flat", borderwidth=1,
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=BLUE,
            state="disabled",
        )
        self._label_entry.pack(side="left", padx=(4, 4))
        self._label_entry.bind("<FocusOut>", self._on_label_focusout)

        # Positive / Negative toggle
        toggle_row = tk.Frame(right, background=BG)
        toggle_row.pack(fill="x", pady=(0, 4))
        self._pos_btn = ttk.Button(toggle_row, text="▸ Positive",
                                   command=lambda: self._set_toggle(True))
        self._pos_btn.pack(side="left", padx=(0, 2))
        _tooltip(self._pos_btn, "Show the positive prompt.")
        self._neg_btn = ttk.Button(toggle_row, text="  Negative",
                                   command=lambda: self._set_toggle(False))
        self._neg_btn.pack(side="left")
        _tooltip(self._neg_btn, "Show the negative prompt.")

        # Single prompt text area
        txt_frame = tk.Frame(right, background=BG)
        txt_frame.pack(fill="both", expand=True, pady=(0, 6))
        self._detail_text = tk.Text(
            txt_frame, wrap="word", state="disabled",
            font=("TkDefaultFont", 9), relief="flat", borderwidth=1,
            background=BG_INPUT, foreground=FG, insertbackground=FG,
            selectbackground=BLUE_SEL, selectforeground=FG,
        )
        txt_scroll = ttk.Scrollbar(txt_frame, orient="vertical",
                                   command=self._detail_text.yview)
        self._detail_text.configure(yscrollcommand=txt_scroll.set)
        self._detail_text.pack(side="left", fill="both", expand=True)
        txt_scroll.pack(side="right", fill="y")

        # Action buttons
        btn_row = tk.Frame(right, background=BG)
        btn_row.pack(fill="x", pady=(0, 4))

        self._use_btn = ttk.Button(btn_row, text="← Use",
                                   command=self._use_current, state="disabled")
        self._use_btn.pack(side="left", padx=(0, 4))
        _tooltip(self._use_btn,
                 "Copy the currently displayed prompt\n"
                 "(positive or negative, per toggle)\nto the main sampler field.")

        self._use_both_btn = ttk.Button(btn_row, text="← Use Both",
                                        command=self._use_both, state="disabled")
        self._use_both_btn.pack(side="left", padx=(0, 12))
        _tooltip(self._use_both_btn,
                 "Copy both positive and negative prompts\nto the main sampler fields.")

        self._delete_btn = ttk.Button(btn_row, text="Delete",
                                      command=self._delete_entry, state="disabled")
        self._delete_btn.pack(side="left")
        _tooltip(self._delete_btn, "Remove the selected entry from the library.")

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def _set_toggle(self, show_positive: bool) -> None:
        self._show_positive = show_positive
        self._pos_btn.config(text="▸ Positive" if show_positive else "  Positive")
        self._neg_btn.config(text="  Negative" if show_positive else "▸ Negative")
        self._update_prompt_display()

    def _update_prompt_display(self) -> None:
        if self._selected_id is None:
            return
        entry = self._get_entry_by_id(self._selected_id)
        if entry is None:
            return
        key  = "positive" if self._show_positive else "negative"
        text = entry.get(key, "")
        self._detail_text.config(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("1.0", text)
        self._detail_text.config(state="disabled")

    # ------------------------------------------------------------------
    # List refresh (grouped by date)
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        # Remember which date groups are currently open/collapsed
        for iid in self._tree.get_children():
            if self._tree.item(iid, "open"):
                self._prev_open_groups.add(iid)
            else:
                self._prev_open_groups.discard(iid)

        sel_id = self._selected_id
        self._tree.delete(*self._tree.get_children())

        # Apply search + model filter
        search   = self._search_var.get().lower() if hasattr(self, "_search_var") else ""
        model_f  = self._model_filter_var.get() if hasattr(self, "_model_filter_var") else "All"
        filtered = []
        for entry in self._data.get("entries", []):
            if model_f != "All" and entry.get("model", "All") not in ("All", model_f):
                continue
            if search:
                hay = (entry.get("label", "") + " "
                       + entry.get("positive", "") + " "
                       + entry.get("negative", "")).lower()
                if search not in hay:
                    continue
            filtered.append(entry)

        # Group by date, newest first
        groups: dict[str, list] = defaultdict(list)
        for entry in filtered:
            groups[_date_key(entry.get("created", ""))].append(entry)

        yesterday_key = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        for date_key in sorted(groups.keys(), reverse=True):
            entries    = sorted(groups[date_key],
                                key=lambda e: e.get("created", ""), reverse=True)
            group_iid  = f"grp_{date_key}"
            label      = _date_label(date_key)

            # Determine open state: default open for today/yesterday
            if group_iid in self._prev_open_groups:
                should_open = True
            elif group_iid not in {f"grp_{k}" for k in groups}:
                should_open = date_key >= yesterday_key
            else:
                # First time seeing this group
                should_open = date_key >= yesterday_key

            self._tree.insert(
                "", "end", iid=group_iid,
                text=f"  {label}  ({len(entries)})",
                open=should_open,
                tags=("group",),
            )

            for entry in entries:
                model_short = entry.get("model", "All")
                if "Wan" in model_short:
                    model_tag = "[Wan]"
                elif "Chroma" in model_short:
                    model_tag = "[Chroma]"
                else:
                    model_tag = "[All]"
                snippet = entry.get("positive", "").replace("\n", " ")[:60]
                tag = "entry_auto" if entry.get("source") == "auto" else "entry_manual"
                self._tree.insert(
                    group_iid, "end", iid=entry["id"],
                    text=f"    {model_tag} {snippet}",
                    tags=(tag,),
                )

        # Restore selection
        if sel_id and self._tree.exists(sel_id):
            self._tree.selection_set(sel_id)
            self._tree.see(sel_id)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_entry_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("grp_"):
            # Date group header — leave detail pane unchanged
            return
        self._selected_id = iid
        entry = self._get_entry_by_id(iid)
        if entry:
            self._populate_detail(entry)
        else:
            self._clear_detail()

    def _get_entry_by_id(self, entry_id: str) -> dict | None:
        for entry in self._data.get("entries", []):
            if entry.get("id") == entry_id:
                return entry
        return None

    # ------------------------------------------------------------------
    # Detail pane
    # ------------------------------------------------------------------

    def _populate_detail(self, entry: dict) -> None:
        self._label_var.set(entry.get("label", ""))
        self._label_entry.config(state="normal")

        self._update_prompt_display()

        for btn in (self._use_btn, self._use_both_btn, self._delete_btn):
            btn.config(state="normal")

    def _clear_detail(self) -> None:
        self._selected_id = None
        self._label_var.set("")
        self._label_entry.config(state="disabled")
        self._detail_text.config(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.config(state="disabled")
        for btn in (self._use_btn, self._use_both_btn, self._delete_btn):
            btn.config(state="disabled")

    def _on_label_focusout(self, _event=None) -> None:
        if self._selected_id is None:
            return
        entry = self._get_entry_by_id(self._selected_id)
        if entry is None:
            return
        new_label = self._label_var.get().strip()
        if new_label and new_label != entry.get("label", ""):
            entry["label"] = new_label
            save_library(self._data)
            self._refresh_list()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _save_current_prompts(self) -> None:
        try:
            pos = self._app._prompt_text.get("1.0", "end-1c")
            neg = self._app._neg_text.get("1.0", "end-1c")
        except Exception:
            return

        cls = type(self._app).__name__
        if "Wan" in cls:
            model = "Wan 2.2 T2V-A14B"
        elif "Chroma" in cls:
            model = "Chroma"
        else:
            model = "All"

        snippet = pos.strip()[:50].replace("\n", " ")
        entry = {
            "id":       uuid.uuid4().hex,
            "label":    snippet or "Untitled",
            "model":    model,
            "positive": pos,
            "negative": neg,
            "created":  datetime.datetime.now().isoformat(timespec="seconds"),
            "source":   "manual",
        }
        self._data.setdefault("entries", []).append(entry)
        save_library(self._data)
        self._refresh_list()

        if self._tree.exists(entry["id"]):
            self._tree.selection_set(entry["id"])
            self._tree.see(entry["id"])
            self._on_entry_select()

    def _use_current(self) -> None:
        if self._selected_id is None:
            return
        entry = self._get_entry_by_id(self._selected_id)
        if entry is None:
            return
        if self._show_positive:
            try:
                self._app._prompt_text.delete("1.0", "end")
                self._app._prompt_text.insert("1.0", entry.get("positive", ""))
                self._app._schedule_token_count()
            except Exception:
                pass
        else:
            try:
                self._app._neg_text.delete("1.0", "end")
                self._app._neg_text.insert("1.0", entry.get("negative", ""))
            except Exception:
                pass

    def _use_both(self) -> None:
        if self._selected_id is None:
            return
        entry = self._get_entry_by_id(self._selected_id)
        if entry is None:
            return
        try:
            self._app._prompt_text.delete("1.0", "end")
            self._app._prompt_text.insert("1.0", entry.get("positive", ""))
            self._app._schedule_token_count()
            self._app._neg_text.delete("1.0", "end")
            self._app._neg_text.insert("1.0", entry.get("negative", ""))
        except Exception:
            pass

    def _delete_entry(self) -> None:
        if self._selected_id is None:
            return
        self._data["entries"] = [
            e for e in self._data.get("entries", [])
            if e.get("id") != self._selected_id
        ]
        save_library(self._data)
        self._clear_detail()
        self._refresh_list()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        global _open_window
        _open_window = None
        try:
            self._app._lib_win = None
        except Exception:
            pass
        self.window.destroy()


# ---------------------------------------------------------------------------
# Minimal tooltip (avoids circular import)
# ---------------------------------------------------------------------------

class _Tip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._w    = widget
        self._text = text
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _=None) -> None:
        if self._tip:
            return
        x = self._w.winfo_rootx() + 20
        y = self._w.winfo_rooty() + self._w.winfo_height() + 4
        self._tip = tk.Toplevel(self._w)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self._text, justify="left",
                 background="#2d2d30", foreground="#cccccc",
                 relief="solid", borderwidth=1,
                 font=("TkDefaultFont", 8), padx=6, pady=3).pack()

    def _hide(self, _=None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


def _tooltip(widget: tk.Widget, text: str) -> None:
    _Tip(widget, text)

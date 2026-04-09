"""Parameter sweep — dialog, expansion, and HTML grid viewer."""
import datetime
import html as _html
import os
import tkinter as tk
from itertools import product
from tkinter import ttk

from sampler_core.gui.tooltip import Tooltip

# ── Sweepable parameters per model type ──────────────────────────────────────

SWEEP_PARAMS: dict[str, dict] = {
    "chroma": {
        "steps":       {"type": int,   "label": "Steps"},
        "cfg_scale":   {"type": float, "label": "CFG Scale"},
        "scheduler":   {"type": str,   "label": "Scheduler",
                        "options": ["Euler", "Heun"]},
        "sigma_shift": {"type": float, "label": "Sigma Shift"},
        "seed":        {"type": int,   "label": "Seed"},
    },
    "wan": {
        "steps_high":  {"type": int,   "label": "Steps (HIGH)"},
        "steps_low":   {"type": int,   "label": "Steps (LOW)"},
        "cfg_scale":   {"type": float, "label": "CFG Scale"},
        "cfg_scale_2": {"type": float, "label": "CFG Scale 2"},
        "seed":        {"type": int,   "label": "Seed"},
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Sweep Dialog
# ═════════════════════════════════════════════════════════════════════════════

class SweepDialog:
    """Modal dialog for configuring a parameter sweep."""

    MAX_AXES = 4

    def __init__(self, root: tk.Tk, model_type: str, current_cfg: dict):
        self._model_type = model_type
        self._current_cfg = current_cfg
        self._result: dict | None = None
        self._params = SWEEP_PARAMS.get(model_type, SWEEP_PARAMS["chroma"])

        self.window = tk.Toplevel(root)
        self.window.title("Parameter Sweep")
        self.window.transient(root)
        self.window.grab_set()
        self.window.configure(bg="#1e1e1e")
        self.window.resizable(False, False)

        self._axis_rows: list[dict] = []
        self._build_ui()
        self.window.update_idletasks()
        # Centre on parent
        x = root.winfo_x() + (root.winfo_width() - self.window.winfo_reqwidth()) // 2
        y = root.winfo_y() + (root.winfo_height() - self.window.winfo_reqheight()) // 2
        self.window.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    @property
    def result(self) -> dict | None:
        return self._result

    # ── Build ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        w = self.window

        # Title
        ttk.Label(w, text="Configure sweep axes. Each axis specifies "
                         "comma-separated values.",
                  wraplength=460).pack(**pad, anchor="w")

        # Axes container
        self._axes_frame = ttk.Frame(w)
        self._axes_frame.pack(fill="x", **pad)
        self._add_axis_row()  # start with one

        # Add axis button
        self._add_btn = ttk.Button(w, text="+ Add Axis", command=self._add_axis_row)
        self._add_btn.pack(**pad, anchor="w")

        ttk.Separator(w, orient="horizontal").pack(fill="x", **pad)

        # ── Prompt sweep ────────────────────────────────────────────────────
        self._prompt_enabled = tk.BooleanVar(value=False)
        pf = ttk.Frame(w)
        pf.pack(fill="x", **pad)
        ttk.Checkbutton(pf, text="Sweep prompts (separate with  || )",
                        variable=self._prompt_enabled,
                        command=self._update_count).pack(anchor="w")
        self._prompt_text = tk.Text(pf, height=3, width=56, wrap="word",
                                    bg="#3c3f41", fg="#cccccc",
                                    insertbackground="#cccccc",
                                    relief="flat", bd=1,
                                    highlightbackground="#454545",
                                    highlightthickness=1)
        self._prompt_text.pack(fill="x", pady=(2, 0))
        cur_prompt = self._current_cfg.get("prompt", "")
        if cur_prompt.strip():
            self._prompt_text.insert("1.0", cur_prompt)
        Tooltip(self._prompt_text,
                "Enter multiple prompts separated by  ||  (double pipe).\n\n"
                "Example:\n"
                "  A cat sitting on a roof || A dog in a park\n\n"
                "Each prompt will be combined with every parameter\n"
                "combination from the axes above.")

        ttk.Separator(w, orient="horizontal").pack(fill="x", **pad)

        # ── Seed ────────────────────────────────────────────────────────────
        sf = ttk.Frame(w)
        sf.pack(fill="x", **pad)
        self._fixed_seed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="Use fixed seed across sweep:",
                        variable=self._fixed_seed_var).pack(side="left")
        self._seed_entry_var = tk.StringVar(
            value=str(self._current_cfg.get("seed", 42)))
        ttk.Entry(sf, textvariable=self._seed_entry_var,
                  width=12).pack(side="left", padx=4)
        Tooltip(sf,
                "When checked, all sweep images use the same seed,\n"
                "isolating the effect of the swept parameters.\n\n"
                "Uncheck to use whatever seed mode is set in the main UI.")

        ttk.Separator(w, orient="horizontal").pack(fill="x", **pad)

        # ── Summary + buttons ───────────────────────────────────────────────
        self._count_var = tk.StringVar(value="Total: 0 images")
        self._count_label = ttk.Label(w, textvariable=self._count_var,
                                      font=("TkDefaultFont", 10, "bold"))
        self._count_label.pack(**pad, anchor="w")

        self._warn_var = tk.StringVar(value="")
        self._warn_label = ttk.Label(w, textvariable=self._warn_var,
                                     foreground="#cc8833")
        self._warn_label.pack(padx=8, anchor="w")

        bf = ttk.Frame(w)
        bf.pack(fill="x", **pad, pady=(4, 8))
        ttk.Button(bf, text="Cancel", command=self.window.destroy
                   ).pack(side="right", padx=4)
        self._go_btn = ttk.Button(bf, text="Add to Queue",
                                  style="Accent.TButton",
                                  command=self._on_confirm)
        self._go_btn.pack(side="right", padx=4)

        self._update_count()

    # ── Axis rows ────────────────────────────────────────────────────────────

    def _add_axis_row(self):
        if len(self._axis_rows) >= self.MAX_AXES:
            return
        row_frame = ttk.Frame(self._axes_frame)
        row_frame.pack(fill="x", pady=2)

        # Parameter dropdown
        used = {r["combo"].get() for r in self._axis_rows}
        available = [k for k in self._params if k not in used]
        param_var = tk.StringVar(value=available[0] if available else "")
        combo = ttk.Combobox(row_frame, textvariable=param_var,
                             values=available, state="readonly", width=14)
        combo.pack(side="left", padx=(0, 4))

        # Values entry
        default_val = ""
        if param_var.get():
            cur = self._current_cfg.get(param_var.get(), "")
            default_val = str(cur)
        val_var = tk.StringVar(value=default_val)
        entry = ttk.Entry(row_frame, textvariable=val_var, width=36)
        entry.pack(side="left", padx=(0, 4), fill="x", expand=True)

        # Remove button
        row_data: dict = {}
        def _remove():
            row_frame.destroy()
            self._axis_rows.remove(row_data)
            self._refresh_axis_options()
            self._update_count()
            if len(self._axis_rows) < self.MAX_AXES:
                self._add_btn.configure(state="normal")
        rm_btn = ttk.Button(row_frame, text="x", width=2, command=_remove)
        rm_btn.pack(side="left")

        row_data.update({
            "frame": row_frame, "combo": combo, "param_var": param_var,
            "entry": entry, "val_var": val_var,
        })
        self._axis_rows.append(row_data)

        # Update dropdown when selection changes
        combo.bind("<<ComboboxSelected>>", lambda e: (
            self._refresh_axis_options(),
            self._prefill_default(row_data),
            self._update_count(),
        ))
        val_var.trace_add("write", lambda *_: self._update_count())

        if len(self._axis_rows) >= self.MAX_AXES:
            self._add_btn.configure(state="disabled")

    def _prefill_default(self, row_data):
        """Fill the values entry with the current cfg value for the selected param."""
        param = row_data["param_var"].get()
        if param and not row_data["val_var"].get().strip():
            cur = self._current_cfg.get(param, "")
            row_data["val_var"].set(str(cur))

    def _refresh_axis_options(self):
        """Update each axis combo's option list to exclude already-selected params."""
        used = set()
        for r in self._axis_rows:
            v = r["param_var"].get()
            if v:
                used.add(v)
        for r in self._axis_rows:
            own = r["param_var"].get()
            available = [k for k in self._params if k not in used or k == own]
            r["combo"]["values"] = available

    # ── Parsing & validation ─────────────────────────────────────────────────

    def _parse_axes(self) -> list[dict] | None:
        """Parse axis rows into a list of {param, values} dicts. Returns None on error."""
        axes = []
        for row in self._axis_rows:
            param = row["param_var"].get()
            raw = row["val_var"].get().strip()
            if not param or not raw:
                continue
            info = self._params[param]
            try:
                values = [info["type"](v.strip()) for v in raw.split(",")
                          if v.strip()]
            except (ValueError, TypeError):
                return None
            if not values:
                continue
            axes.append({"param": param, "values": values})
        return axes

    def _parse_prompts(self) -> list[str] | None:
        """Parse prompt sweep text. Returns None if disabled or empty."""
        if not self._prompt_enabled.get():
            return None
        raw = self._prompt_text.get("1.0", "end").strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split("||") if p.strip()]
        return parts if parts else None

    def _update_count(self, *_):
        axes = self._parse_axes() or []
        n = 1
        for a in axes:
            n *= len(a["values"])
        prompts = self._parse_prompts()
        if prompts:
            n *= len(prompts)
        self._count_var.set(f"Total: {n} image{'s' if n != 1 else ''}")
        if n > 50:
            self._warn_var.set(f"Warning: {n} images — this may take a while!")
        else:
            self._warn_var.set("")
        self._go_btn.configure(
            state="normal" if (axes or prompts) and n > 0 else "disabled")

    # ── Confirm ──────────────────────────────────────────────────────────────

    def _on_confirm(self):
        axes = self._parse_axes()
        if axes is None:
            self._warn_var.set("Invalid values — check types (int/float).")
            return
        prompts = self._parse_prompts()
        if not axes and not prompts:
            return

        # Seed
        fixed_seed = None
        if self._fixed_seed_var.get():
            try:
                fixed_seed = int(self._seed_entry_var.get())
            except ValueError:
                self._warn_var.set("Invalid seed value.")
                return

        self._result = {
            "axes": axes or [],
            "prompts": prompts,
            "fixed_seed": fixed_seed,
        }
        self.window.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# Sweep expansion
# ═════════════════════════════════════════════════════════════════════════════

def expand_sweep(base_cfg: dict, sweep_result: dict) -> list[dict]:
    """Expand sweep config into a list of individual job cfg dicts.

    Each returned dict is a copy of *base_cfg* with swept parameters
    overridden, ``output_dir`` redirected to a timestamped subfolder,
    and a ``_sweep_meta`` key added for tracking.
    """
    axes = sweep_result["axes"]
    prompts = sweep_result.get("prompts")
    fixed_seed = sweep_result.get("fixed_seed")

    sweep_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    original_outdir = base_cfg.get("output_dir", "output")
    sweep_dir = os.path.join(original_outdir, f"sweep-{sweep_id}")

    # Build dimensions for cartesian product
    dims: list[list[tuple[str, object]]] = []
    axis_names: list[str] = []
    for axis in axes:
        dims.append([(axis["param"], v) for v in axis["values"]])
        axis_names.append(axis["param"])
    if prompts:
        dims.append([("prompt", p) for p in prompts])
        axis_names.append("prompt")

    if not dims:
        return []

    combos = list(product(*dims))

    result = []
    for i, combo in enumerate(combos):
        cfg = dict(base_cfg)
        combo_values: dict = {}
        for param, value in combo:
            cfg[param] = value
            combo_values[param] = value

        if fixed_seed is not None and "seed" not in combo_values:
            cfg["seed"] = fixed_seed
            cfg["random_seed"] = False

        cfg["output_dir"] = sweep_dir
        cfg["_sweep_meta"] = {
            "sweep_id": sweep_id,
            "sweep_dir": sweep_dir,
            "axes": [{"param": n, "values": [v for _, v in d]}
                     for n, d in zip(axis_names, dims)],
            "combo_index": i,
            "combo_values": combo_values,
            "total": len(combos),
        }
        result.append(cfg)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# HTML grid viewer
# ═════════════════════════════════════════════════════════════════════════════

def generate_sweep_html(
    sweep_dir: str,
    axes: list[dict],
    results: list[dict],
    total: int,
) -> str:
    """Generate (or regenerate) an ``index.html`` grid viewer in *sweep_dir*.

    *axes*: list of ``{"param": str, "values": list}``.
    *results*: list of ``{"combo_index": int, "combo_values": dict,
               "output_path": str|None, "status": str, "seed": int|None,
               "error": str|None}``.
    *total*: total expected number of images.

    Returns the path to the generated HTML file.
    """
    os.makedirs(sweep_dir, exist_ok=True)
    html_path = os.path.join(sweep_dir, "index.html")

    # Build a lookup: combo_index → result
    result_map: dict[int, dict] = {}
    for r in results:
        result_map[r["combo_index"]] = r

    # Compute full cartesian product of axis values to map index → values
    if axes:
        all_values = [a["values"] for a in axes]
        combos = list(product(*all_values))
    else:
        combos = [()]

    completed = sum(1 for r in results if r.get("status") == "done")
    errored = sum(1 for r in results if r.get("status") == "error")

    # ── Build HTML ───────────────────────────────────────────────────────
    h = _html.escape
    lines: list[str] = []
    lines.append("<!DOCTYPE html>")
    lines.append('<html lang="en"><head><meta charset="utf-8">')
    lines.append(f"<title>Sweep {os.path.basename(sweep_dir)}</title>")
    lines.append("<style>")
    lines.append(_CSS)
    lines.append("</style></head><body>")

    # Header
    lines.append(f"<h1>Parameter Sweep</h1>")
    axis_desc = " &times; ".join(
        f"<b>{h(a['param'])}</b>: {', '.join(str(v) for v in a['values'])}"
        for a in axes)
    if axis_desc:
        lines.append(f"<p class='axes-desc'>{axis_desc}</p>")
    status_parts = [f"{completed}/{total} completed"]
    if errored:
        status_parts.append(f"{errored} errors")
    pending = total - completed - errored
    if pending > 0:
        status_parts.append(f"{pending} pending")
    lines.append(f"<p class='status'>{' &mdash; '.join(status_parts)}</p>")

    # ── Layout based on axis count ───────────────────────────────────────
    n_axes = len(axes)

    if n_axes == 0:
        # Prompt-only sweep or empty — flat gallery
        lines.append('<div class="gallery">')
        for idx in range(total):
            lines.append(_render_cell(idx, result_map, combos, axes))
        lines.append('</div>')

    elif n_axes == 1:
        # Single row
        lines.append('<div class="grid-1d">')
        for col_idx, val in enumerate(axes[0]["values"]):
            lines.append(f'<div class="col">')
            lines.append(f'<div class="header">{h(str(val))}</div>')
            lines.append(_render_cell(col_idx, result_map, combos, axes))
            lines.append('</div>')
        lines.append('</div>')

    elif n_axes == 2:
        # 2D grid: axis[0] = columns, axis[1] = rows
        _render_grid_2d(lines, axes, 0, 1, result_map, combos, total)

    else:
        # 3+ axes: group by axes[2:], each group is a 2D grid of axes[0]×[1]
        outer_axes = axes[2:]
        outer_values = [a["values"] for a in outer_axes]
        outer_combos = list(product(*outer_values))
        for oc in outer_combos:
            label = ", ".join(f"{h(outer_axes[i]['param'])}={h(str(v))}"
                              for i, v in enumerate(oc))
            lines.append(f'<h2 class="group-label">{label}</h2>')
            _render_grid_2d(lines, axes, 0, 1, result_map, combos, total,
                            filter_outer=dict(zip(
                                [a["param"] for a in outer_axes], oc)))

    # Lightbox
    lines.append('<div class="lightbox" onclick="this.style.display=\'none\'">'
                 '<img src=""></div>')
    lines.append("<script>")
    lines.append(_JS)
    lines.append("</script>")
    lines.append("</body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return html_path


def _render_cell(idx: int, result_map: dict, combos: list,
                 axes: list[dict]) -> str:
    """Render a single image cell (or placeholder) for combo *idx*."""
    h = _html.escape
    r = result_map.get(idx)
    if r and r.get("status") == "done" and r.get("output_path"):
        fname = os.path.basename(r["output_path"])
        label = ", ".join(f"{k}={v}" for k, v in r.get("combo_values", {}).items())
        seed_str = f"seed: {r.get('seed', '?')}" if r.get("seed") is not None else ""
        return (f'<div class="cell done">'
                f'<img src="{h(fname)}" alt="{h(label)}" loading="lazy">'
                f'<div class="meta">{h(label)}</div>'
                f'<div class="meta">{h(seed_str)}</div>'
                f'</div>')
    elif r and r.get("status") == "error":
        err = r.get("error", "Unknown error")[:80]
        return (f'<div class="cell error">'
                f'<div class="error-text">Error</div>'
                f'<div class="meta">{h(err)}</div></div>')
    else:
        return '<div class="cell pending"><div class="pending-text">Pending...</div></div>'


def _render_grid_2d(lines: list[str], axes: list[dict],
                    col_axis: int, row_axis: int,
                    result_map: dict, combos: list, total: int,
                    filter_outer: dict | None = None):
    """Render a 2D grid table for two axes."""
    h = _html.escape
    col_vals = axes[col_axis]["values"]
    row_vals = axes[row_axis]["values"]
    n_cols = len(col_vals)

    lines.append(f'<table class="grid-2d">')
    # Header row
    lines.append('<tr><th class="corner"></th>')
    for cv in col_vals:
        lines.append(f'<th class="header">{h(axes[col_axis]["param"])}='
                     f'{h(str(cv))}</th>')
    lines.append('</tr>')

    for rv in row_vals:
        lines.append(f'<tr><th class="header row-hdr">'
                     f'{h(axes[row_axis]["param"])}={h(str(rv))}</th>')
        for cv in col_vals:
            # Find the combo index matching these axis values
            idx = _find_combo_index(combos, axes, col_axis, cv,
                                    row_axis, rv, filter_outer)
            if idx is not None:
                lines.append(f'<td>{_render_cell(idx, result_map, combos, axes)}</td>')
            else:
                lines.append('<td><div class="cell pending">'
                             '<div class="pending-text">N/A</div></div></td>')
        lines.append('</tr>')
    lines.append('</table>')


def _find_combo_index(combos: list, axes: list[dict],
                      col_axis: int, col_val,
                      row_axis: int, row_val,
                      filter_outer: dict | None = None) -> int | None:
    """Find the index in *combos* matching the given axis values."""
    for idx, combo in enumerate(combos):
        if combo[col_axis] != col_val or combo[row_axis] != row_val:
            continue
        if filter_outer:
            match = True
            for ax_idx, a in enumerate(axes):
                if a["param"] in filter_outer:
                    if combo[ax_idx] != filter_outer[a["param"]]:
                        match = False
                        break
            if not match:
                continue
        return idx
    return None


# ── Static assets ────────────────────────────────────────────────────────────

_CSS = """\
* { box-sizing: border-box; }
body {
    background: #1e1e1e; color: #ccc; font-family: -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px 24px;
}
h1 { color: #4fc3f7; margin: 0 0 8px; font-size: 22px; }
h2.group-label { color: #4fc3f7; font-size: 16px; margin: 24px 0 8px; }
.axes-desc { color: #aaa; font-size: 13px; margin: 4px 0; }
.status { color: #808080; font-size: 13px; margin: 4px 0 16px; }

/* 1-axis: horizontal strip */
.grid-1d {
    display: flex; gap: 8px; overflow-x: auto; padding-bottom: 8px;
}
.grid-1d .col { text-align: center; min-width: 180px; }

/* 2-axis: table grid */
.grid-2d {
    border-collapse: collapse; margin: 8px 0 24px;
}
.grid-2d th, .grid-2d td {
    border: 1px solid #454545; padding: 4px; vertical-align: top;
}
.corner { background: #252526; }
.header {
    background: #252526; color: #4fc3f7; font-size: 12px;
    font-weight: bold; padding: 6px 8px; white-space: nowrap;
}
.row-hdr { text-align: right; }

/* Gallery (0-axis / prompt-only) */
.gallery {
    display: flex; flex-wrap: wrap; gap: 8px;
}

/* Cells */
.cell { text-align: center; min-width: 160px; max-width: 320px; }
.cell img {
    max-width: 100%; border: 1px solid #454545; border-radius: 3px;
    cursor: pointer; display: block; margin: 0 auto;
}
.cell img:hover { border-color: #4fc3f7; }
.cell.pending {
    background: #2d2d30; min-height: 120px; display: flex;
    align-items: center; justify-content: center; border: 1px dashed #454545;
    border-radius: 3px;
}
.cell.error {
    background: #3a1e1e; min-height: 120px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; border: 1px solid #662222;
    border-radius: 3px;
}
.pending-text { color: #666; font-size: 13px; }
.error-text { color: #cc4444; font-weight: bold; font-size: 14px; }
.meta { font-size: 11px; color: #808080; margin-top: 2px; }

/* Lightbox */
.lightbox {
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.92); z-index: 100; cursor: pointer;
}
.lightbox img {
    max-width: 92%; max-height: 92%; position: absolute;
    top: 50%; left: 50%; transform: translate(-50%, -50%);
    border: 1px solid #454545; border-radius: 4px;
}
"""

_JS = """\
document.querySelectorAll('.cell.done img').forEach(function(img) {
    img.addEventListener('click', function(e) {
        e.stopPropagation();
        var lb = document.querySelector('.lightbox');
        lb.querySelector('img').src = img.src;
        lb.style.display = 'block';
    });
});
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        document.querySelector('.lightbox').style.display = 'none';
    }
});
"""

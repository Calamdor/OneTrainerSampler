"""Dark theme with blue highlights for the sampler GUIs."""
import tkinter as tk
from tkinter import ttk

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#1e1e1e"   # root / main background
BG_PANEL  = "#252526"   # label-frame / panel fill
BG_WIDGET = "#2d2d30"   # buttons, headers
BG_INPUT  = "#3c3f41"   # entry / text / combobox fields
FG        = "#cccccc"   # primary text
FG_DIM    = "#808080"   # secondary / grayed text
BLUE      = "#4fc3f7"   # accent (label-frame titles, focus ring)
BLUE_DARK = "#0e639c"   # button hover, progress bar fill
BLUE_SEL  = "#264f78"   # text / treeview selection background
BORDER    = "#454545"   # borders and separators


def apply_dark_theme(root: tk.Tk) -> None:
    """Configure ttk styles and the root window for the dark theme."""
    root.configure(background=BG)

    s = ttk.Style(root)
    s.theme_use("clam")          # clam is the most customisable on all platforms

    # ── Base defaults (inherited by all ttk widgets) ──────────────────────────
    s.configure(".",
        background=BG, foreground=FG,
        bordercolor=BORDER, darkcolor=BG_PANEL, lightcolor=BG_WIDGET,
        troughcolor=BG_PANEL, focuscolor=BLUE,
        selectbackground=BLUE_SEL, selectforeground=FG,
        insertcolor=FG,
    )

    # ── Frames ────────────────────────────────────────────────────────────────
    s.configure("TFrame", background=BG)

    s.configure("TLabelframe",
        background=BG, bordercolor=BORDER, relief="groove")
    s.configure("TLabelframe.Label",
        background=BG, foreground=BLUE,
        font=("TkDefaultFont", 9, "bold"))

    # ── Labels ────────────────────────────────────────────────────────────────
    s.configure("TLabel", background=BG, foreground=FG)

    # ── Buttons ───────────────────────────────────────────────────────────────
    s.configure("TButton",
        background=BG_WIDGET, foreground=FG,
        bordercolor=BORDER, lightcolor=BG_WIDGET, darkcolor=BG_WIDGET,
        padding=(8, 3),
    )
    s.map("TButton",
        background=[("pressed", "#094771"), ("active", BLUE_DARK)],
        foreground=[("pressed", "#ffffff"), ("active", "#ffffff")],
        bordercolor=[("active", BLUE_DARK)],
    )

    # ── Accent button ("＋ Add to Queue") ──────────────────────────────────
    GREEN       = "#1a6b3c"
    GREEN_HOVER = "#22914f"
    s.configure("Accent.TButton",
        background=GREEN, foreground="#ffffff",
        bordercolor=GREEN_HOVER,
        lightcolor=GREEN, darkcolor=GREEN,
        padding=(8, 3),
    )
    s.map("Accent.TButton",
        background=[("pressed", "#0e4a28"), ("active", GREEN_HOVER)],
        foreground=[("pressed", "#ffffff"), ("active", "#ffffff")],
        bordercolor=[("active", GREEN_HOVER)],
    )

    # ── Entry ─────────────────────────────────────────────────────────────────
    s.configure("TEntry",
        fieldbackground=BG_INPUT, foreground=FG,
        insertcolor=FG, bordercolor=BORDER,
        selectbackground=BLUE_SEL, selectforeground=FG,
    )
    s.map("TEntry", bordercolor=[("focus", BLUE)])

    # ── Combobox ──────────────────────────────────────────────────────────────
    s.configure("TCombobox",
        fieldbackground=BG_INPUT, foreground=FG,
        selectbackground=BLUE_SEL, selectforeground=FG,
        arrowcolor=FG, bordercolor=BORDER,
        darkcolor=BG_INPUT, lightcolor=BG_INPUT,
    )
    s.map("TCombobox",
        fieldbackground=[("readonly", BG_INPUT)],
        selectbackground=[("readonly", BLUE_SEL)],
        bordercolor=[("focus", BLUE)],
    )

    # ── Spinbox ───────────────────────────────────────────────────────────────
    s.configure("TSpinbox",
        fieldbackground=BG_INPUT, foreground=FG,
        arrowcolor=FG, bordercolor=BORDER,
        insertcolor=FG,
    )
    s.map("TSpinbox", bordercolor=[("focus", BLUE)])

    # ── Checkbutton ───────────────────────────────────────────────────────────
    s.configure("TCheckbutton", background=BG, foreground=FG)
    s.map("TCheckbutton",
        background=[("active", BG)],
        indicatorcolor=[("selected", BLUE), ("!selected", BG_INPUT)],
        indicatorbackground=[("selected", BLUE_SEL), ("!selected", BG_INPUT)],
    )

    # ── Separator ─────────────────────────────────────────────────────────────
    s.configure("TSeparator", background=BORDER)

    # ── Progressbar ───────────────────────────────────────────────────────────
    s.configure("Horizontal.TProgressbar",
        background=BLUE_DARK, troughcolor=BG_PANEL,
        bordercolor=BORDER, lightcolor=BLUE_DARK, darkcolor=BLUE_DARK,
    )

    # ── Scrollbar ─────────────────────────────────────────────────────────────
    s.configure("TScrollbar",
        background=BG_WIDGET, troughcolor=BG_PANEL,
        arrowcolor=FG, bordercolor=BORDER,
        lightcolor=BG_WIDGET, darkcolor=BG_WIDGET,
    )
    s.map("TScrollbar", background=[("active", BLUE_DARK)])

    # ── Treeview ──────────────────────────────────────────────────────────────
    s.configure("Treeview",
        background=BG_PANEL, foreground=FG,
        fieldbackground=BG_PANEL, bordercolor=BORDER,
        rowheight=22,
    )
    s.configure("Treeview.Heading",
        background=BG_WIDGET, foreground=FG,
        bordercolor=BORDER, lightcolor=BG_WIDGET, darkcolor=BG_WIDGET,
    )
    s.map("Treeview",
        background=[("selected", BLUE_SEL)],
        foreground=[("selected", "#ffffff")],
    )
    s.map("Treeview.Heading",
        background=[("active", BG_INPUT)],
    )

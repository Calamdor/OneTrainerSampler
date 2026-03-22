import tkinter as tk

# Dark theme colours (keep in sync with theme.py)
_TT_BG     = "#2d2d30"
_TT_FG     = "#cccccc"
_TT_BORDER = "#454545"


class Tooltip:
    """Lightweight hover tooltip for any tkinter widget."""
    def __init__(self, widget, text: str):
        self._widget = widget
        self._text   = text
        self._win    = None
        widget.bind("<Enter>",       self._show, add="+")
        widget.bind("<Leave>",       self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, event=None):
        if self._win:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self._text, justify="left",
            background=_TT_BG, foreground=_TT_FG,
            relief="solid", borderwidth=1, highlightbackground=_TT_BORDER,
            font=("TkDefaultFont", 8), wraplength=340, padx=4, pady=3,
        ).pack()

    def _hide(self, event=None):
        if self._win:
            self._win.destroy()
            self._win = None

"""OneTrainer Standalone Sampler — unified entry point.

Select the model from the dropdown at the top; the interface rebuilds
automatically.  Each model stores its settings in its own config file so
switching models never resets your settings.
"""
# Set torch compile cache dirs and enable persistent caching BEFORE any torch import.
# These env vars must be set before torch is imported to take effect.
import os as _os
_os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    _os.path.join(_os.path.expanduser("~"), ".cache", "torchinductor"),
)
_os.environ.setdefault(
    "TRITON_CACHE_DIR",
    _os.path.join(_os.path.expanduser("~"), ".cache", "triton"),
)
# Persistent FX graph cache: stores compiled Triton binaries to disk.
# Helps for non-LoRA compilation; LoRA forward-patch closures have
# per-session Python identity, so cache keys miss across app restarts.
# Still worth enabling for partial hits and non-LoRA use cases.
_os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
_os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")

# ---------------------------------------------------------------------------
# Optional-dependency installer
# Must run BEFORE importing sampler_core so that any newly-installed packages
# (e.g. tkinterdnd2) are importable when app_base.py is first loaded.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REQS_FILE  = _os.path.join(_SCRIPT_DIR, "requirements_sampler.txt")

# Map pip package name → Python import name when they differ.
_IMPORT_NAME = {
    "tkinterdnd2": "tkinterdnd2",
    # add more overrides here if needed in future
}


def _check_optional_deps() -> None:
    """Check requirements_sampler.txt and offer to install any missing packages.

    Runs entirely before the heavy sampler imports so that newly-installed
    packages are picked up when app_base.py is imported later (no restart
    needed for the tkinterdnd2 drag-and-drop feature).
    """
    import importlib.util
    import re
    import subprocess
    import sys
    import tkinter as tk
    from tkinter import messagebox

    if not _os.path.isfile(_REQS_FILE):
        return

    # ── Parse requirements file ──────────────────────────────────────────
    missing: list[tuple[str, str]] = []   # (pip_spec, display_name)
    with open(_REQS_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip version specifier to get the bare package name
            pkg_name = re.split(r"[>=<!;\[]", line)[0].strip()
            if not pkg_name:
                continue
            import_name = _IMPORT_NAME.get(pkg_name.lower(),
                                           pkg_name.replace("-", "_").lower())
            if importlib.util.find_spec(import_name) is None:
                missing.append((line.strip(), pkg_name))

    if not missing:
        return

    # ── Ask the user ─────────────────────────────────────────────────────
    _root = tk.Tk()
    _root.withdraw()

    pkg_lines = "\n".join(f"  \u2022 {name}" for _, name in missing)
    answer = messagebox.askyesno(
        "OneTrainer Sampler \u2014 Optional Dependencies",
        f"The following optional packages are not installed:\n\n"
        f"{pkg_lines}\n\n"
        f"These add drag-and-drop support to the preview panel.\n\n"
        f"Install now into the current Python environment?",
        parent=_root,
    )
    _root.destroy()

    if not answer:
        return

    # ── Show progress window and install ─────────────────────────────────
    _prog = tk.Tk()
    _prog.title("Installing\u2026")
    _prog.geometry("360x90")
    _prog.resizable(False, False)
    tk.Label(
        _prog,
        text="Installing optional dependencies\u2026\nThis may take a moment.",
        padx=24, pady=24,
    ).pack()
    _prog.update()

    failed: list[str] = []
    for pip_spec, pkg_name in missing:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_spec],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failed.append(pkg_name)

    _prog.destroy()

    if failed:
        _err = tk.Tk()
        _err.withdraw()
        messagebox.showwarning(
            "Install Failed",
            f"Could not install: {', '.join(failed)}\n\n"
            f"You can install manually with:\n"
            f"  pip install {' '.join(s for s, _ in missing)}",
            parent=_err,
        )
        _err.destroy()
    # On success: no message needed — the packages are now importable
    # and will be picked up when app_base.py is first imported below.


# Run the check before any sampler imports.
if __name__ == "__main__":
    _check_optional_deps()

# ---------------------------------------------------------------------------
# Main sampler imports (after dep check so newly-installed packages are live)
# ---------------------------------------------------------------------------

import sampler_core  # noqa: F401 — injects OT into sys.path

from sampler_core.gui.launcher import OneTrainerLauncher
import tkinter as tk

if __name__ == "__main__":
    try:
        from tkinterdnd2 import TkinterDnD as _TkDnD
        root = _TkDnD.Tk()
    except ImportError:
        root = tk.Tk()
    root.geometry("1380x860")
    OneTrainerLauncher(root)
    root.mainloop()

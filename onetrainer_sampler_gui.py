"""OneTrainer Standalone Sampler — unified entry point.

Select the model from the dropdown at the top; the interface rebuilds
automatically.  Each model stores its settings in its own config file so
switching models never resets your settings.
"""
# Set torch compile cache dirs before any torch import.
import os as _os
_os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    _os.path.join(_os.path.expanduser("~"), ".cache", "torchinductor"),
)
_os.environ.setdefault(
    "TRITON_CACHE_DIR",
    _os.path.join(_os.path.expanduser("~"), ".cache", "triton"),
)

import sampler_core  # noqa: F401 — injects OT into sys.path

from sampler_core.gui.launcher import OneTrainerLauncher
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1380x860")
    OneTrainerLauncher(root)
    root.mainloop()

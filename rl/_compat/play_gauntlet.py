"""Compatibility for rl.play_gauntlet; implementation: apps.play_gauntlet."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("apps.play_gauntlet", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("apps.play_gauntlet")

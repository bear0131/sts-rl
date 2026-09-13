"""Compatibility for rl.ssh_transport; implementation: rl.training.ssh_transport."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.training.ssh_transport", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.training.ssh_transport")

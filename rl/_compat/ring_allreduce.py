"""Compatibility for rl.ring_allreduce; implementation: rl.training.ring_allreduce."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.training.ring_allreduce", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.training.ring_allreduce")

"""Compatibility for rl.diagnose_rules; implementation: tools.analysis.diagnose_rules."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.diagnose_rules", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.diagnose_rules")

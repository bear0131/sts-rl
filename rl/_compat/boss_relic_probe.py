"""Compatibility for rl.boss_relic_probe; implementation: tools.analysis.boss_relic_probe."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.boss_relic_probe", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.boss_relic_probe")

"""Compatibility for rl.analyze_relic_reward_correlation; implementation: tools.analysis.analyze_relic_reward_correlation."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.analyze_relic_reward_correlation", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.analyze_relic_reward_correlation")

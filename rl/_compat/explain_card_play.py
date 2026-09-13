"""Compatibility for rl.explain_card_play; implementation: rl.evaluation.rules.explain_card_play."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.rules.explain_card_play", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.rules.explain_card_play")

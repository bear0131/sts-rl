"""Compatibility for rl.visualize_vector_rule_card_pca; implementation: tools.analysis.visualize_vector_rule_card_pca."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.visualize_vector_rule_card_pca", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.visualize_vector_rule_card_pca")

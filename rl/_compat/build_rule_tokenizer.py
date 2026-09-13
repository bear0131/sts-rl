"""Compatibility for rl.build_rule_tokenizer; implementation: rl.rules.data.build_rule_tokenizer."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.rules.data.build_rule_tokenizer", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.rules.data.build_rule_tokenizer")

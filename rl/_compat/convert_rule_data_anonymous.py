"""Compatibility for rl.convert_rule_data_anonymous; implementation: rl.rules.data.convert_rule_data_anonymous."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.rules.data.convert_rule_data_anonymous", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.rules.data.convert_rule_data_anonymous")

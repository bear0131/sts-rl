"""Compatibility for rl.audit_vector_rule_v7_contract; implementation: tools.analysis.audit_vector_rule_v7_contract."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.audit_vector_rule_v7_contract", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.audit_vector_rule_v7_contract")

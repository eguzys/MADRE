"""Thin shim — allows `python evaluate_benchmarl.py <args>` from src/.

Forwards to `python -m evaluation.evaluate` (the canonical form).
"""
import runpy, sys
runpy.run_module("evaluation.evaluate", run_name="__main__", alter_sys=True)

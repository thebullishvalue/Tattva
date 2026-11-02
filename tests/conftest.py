"""Tattva test suite — shared setup.

Run from the repository root:  python -m pytest tests -q

Every test here runs on synthetic data, offline: the properties they assert
(causality, sign conventions, the ECM's horizon arithmetic, the edge scorer's
recovery of a planted signal, the conviction chain's rules) are properties of
the code, not of any market, so they must hold without a network.
"""
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "tattva", "numba"))
warnings.filterwarnings("ignore", category=FutureWarning)

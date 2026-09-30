"""Reproduce Tattva's track record on the whole catalogue (research/tattva_v3_study.md §3).

    python research/evaluate_catalogue.py            # uses the cached universe (tattva.data)
    python research/evaluate_catalogue.py --refresh  # re-downloads first
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from tattva import engine, evaluate as ev                      # noqa: E402
from tattva.data import fetch_universe                         # noqa: E402
from tattva.universe import TARGETS_BY_NAME                    # noqa: E402


def main() -> None:
    runs = engine.run_catalogue(fetch_universe(force="--refresh" in sys.argv))
    T = ev.table(runs, {n: TARGETS_BY_NAME[n].pool for n in runs})
    pd.set_option("display.width", 220)
    print(f"{len(runs)} targets\n")
    print(ev.summary(T).round(3).to_string(), "\n")
    print(ev.summary(T, "pool").round(3).to_string())


if __name__ == "__main__":
    main()

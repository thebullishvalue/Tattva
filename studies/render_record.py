"""
Tattva — render the committed call record as the README's table.

    python studies/render_record.py            # print the block
    python studies/render_record.py --write    # replace it in README.md

The block sits between ``<!-- record:start -->`` and ``<!-- record:end -->``.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START, END = "<!-- record:start -->", "<!-- record:end -->"
NAMES = {"commodity": "Commodities", "fx": "Currencies", "crypto": "Crypto",
         "india_index": "Indian equities", "us_index": "US equities"}


def _side(r: dict) -> str:
    if not r or not r.get("n_claims"):
        return "—"
    return f"{r['hit']:.1f}% vs {r['base']:.1f}% · t {r['t']:+.2f}"


def render(rec: dict) -> str:
    rows = [("**All targets**", rec["all"])] + [(NAMES.get(c, c), r) for c, r in rec["classes"].items()]
    out = [START,
           f"Generated {rec['generated']} by `studies/pooled_record.py` from a panel through "
           f"{rec['panel_through']} (fingerprint `{rec['panel_fingerprint']}`). "
           f"{len(rec['targets'])} targets, {rec['horizon']} sessions, entered {rec['lag']} session "
           f"after the claim. *Hit vs base* is the share of claims that made money against the share "
           f"the same direction made on any day; *t* is the mean return in excess of always holding "
           f"that direction, over its standard error.", "",
           "| Asset class | Claims | All claims | Buy claims | Sell claims |",
           "|---|---|---|---|---|"]
    for name, r in rows:
        out.append(f"| {name} | {r['n_claims']:,} | {_side(r)} | {_side(r.get('long'))} | "
                   f"{_side(r.get('short'))} |")
    out += ["", END]
    return "\n".join(out)


def main() -> None:
    with open(os.path.join(ROOT, "data", "priors", "call_record.json"), encoding="utf-8") as f:
        block = render(json.load(f))
    if "--write" not in sys.argv:
        print(block)
        return
    path = os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    a, b = text.index(START), text.index(END) + len(END)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:a] + block + text[b:])
    print("README.md updated")


if __name__ == "__main__":
    main()

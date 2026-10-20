"""
Tattva — the pipeline: from a fetched panel to the issued, graded call.
तत्त्व (Tattva) — "Principle / Essence"

    run.py       the step functions app.py and the offline study share
                 (fetch, prepare, fit Mūla, build Swayam) and run_target,
                 the whole run for one target without Streamlit
    assemble.py  the loop: date-keyed inputs → as-of grades → the call at
                 every session (core.call) → the call graded as issued →
                 the verdict at d*
    evidence.py  admitted evidence about the claim: who closes the gap, and
                 the claim's unit

No Streamlit dependency.
"""

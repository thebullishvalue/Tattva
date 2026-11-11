# Audit working files — Tattva 2.8.2 at `dde6647`

These scripts produced the figures in `docs/RESOLUTION_DIRECTIVE.md`. They
measure the system **as it was audited**: `harness.py` is a copy of the data
preparation `app.py` then contained, kept so the audit's numbers stay
reproducible against that commit.

They are not part of the product and are not maintained. The current
equivalents are `studies/pooled_record.py` (the call as issued, pooled by
asset class) and `studies/run_study.py` (the pre-registered candidates), both
built on `pipeline/`.

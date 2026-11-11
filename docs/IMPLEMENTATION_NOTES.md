# Implementation notes — what was built against the directive

`docs/RESOLUTION_DIRECTIVE.md` is kept as written. This file records where the
implementation departed from it, what it found along the way, and what is left.
Every commit message carries its own before-and-after measurements.

## Departures, and why

| Where | Directive | Built | Why |
|---|---|---|---|
| WP3 | A side's base rate is the majority direction on its own claim rows | The base rate and mean are taken over **every** published session of the target; claims are scored in excess of always holding the direction | Per side, the directive's base equals the hit rate whenever the asset drifts up. The raw buy-side record came out t = +2.88 from drift alone and would have licensed BUY everywhere. |
| WP7a | Give each correlation memory its own factor-level history | One continuous level history; the incoming memory's factors are matched to the existing ones at each switch | Separate histories would make the regressors jump at every switch (12–16 times in eleven years) and put a step in fair value. |
| WP4 | Words `CHEAP — NO GRADED CALL` | The read is the state (`CHEAP`), with `NO GRADED CALL` as its label beside it | One short word fits the card and the tile; the meaning is unchanged. |
| WP1 | `app.py` under 900 lines | 908 | Stopped at the comments worth keeping. |
| WP7b | Delete the drill-down column trimming | Kept (`pipeline.run._slim`) | Frames are ~30 columns now, but the page reads 9; trimming is still free. |
| Order | Stop after WP4 and WP5 for review | Ran straight through | The owner asked for everything to be implemented. WP5's table is in `studies/results/`; nothing was promoted and the holdout is unread. |

## Found while implementing (not in the audit)

- **Seven of fifteen Swayam views rarely voted.** Where a view had no macro
  read, the macro half kept its fixed weight, halving the price signal so it
  seldom reached the oversold / overbought lines. Fixed in WP7c; the share of
  sessions with a breadth lean went from 27% to 65%.
- **Swayam's breadth has no measured edge**, before or after the rewrite: its
  ten-session IC is inside one standard error of zero on all five targets
  measured.
- **The lean's rank IC flatters the call.** USD/INR's lean IC is +0.07 while
  its claims hit 51% against a 48% base; the System page now shows both.
- Results are bit-identical only with single-threaded BLAS (differences of
  1e-12 otherwise). `app.py` and the studies pin it; an embedding caller should.

## What is left

- **No call is licensed.** That is the measured state, not an unfinished task.
  The route to change it is a new pre-registered candidate
  (`studies/PREREGISTRATION.md`), not a threshold.
- The record should be regenerated when the catalogue or the engine changes:
  `python studies/pooled_record.py && python studies/render_record.py --write`.
- Not done: the audit's B2.9 note on futures rolls is documented, not fixed;
  the within-memory eigenbasis compaction (a middle factor dropping out shifts
  later slots) is unchanged; the stylesheet was not audited.
- The UI was checked in a headless browser on Streamlit 1.52.2 in Paper and
  Slate at 1500 px. Phone and tablet widths were not re-checked after the
  Overview tiles and the Universe page were added.

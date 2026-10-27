"""
Tattva — System tab: what the call runs on.
तत्त्व (Tattva) — "Principle / Essence"

INFRASTRUCTURE — everything here guarantees one as-of row and causality, and
none of it is a grade or a reading of the market:

  1 STATE     which history, which session?       d* · h and L · epoch · fingerprint · history rewritten?
  2 TRUST     which pooled priors?                the study's provenance, or 'not yet run'
  3 DETAIL    is the data sound?                  the newest session's coverage · withheld sessions ·
                                                  caches and circuit breakers
  4 RECORD    what can move it, and why?          the KNOBS registry · the ROLES registry

(The gap's stationarity is evidence about the claim and is on the Mūla page,
once; the OU half-life, the θ-stability chip and the duplicate ADF / KPSS
cards that sat here went with the second definition they carried. Every
grade is on the Grades page.)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import core.knobs as knobs
from core.roles import CARDS, CHIPS, ROLES
from ui.components import (render_kpi_strip, render_note, render_section_header,
                           render_table_panel)
from data.cache import all_caches
from data.circuit_breaker import all_circuits, CircuitState


def _history_flag(priors_meta: dict, target: str, fingerprint: str) -> tuple[str, str]:
    """``(value, subtext)``: is this the history the pooled priors were studied on?"""
    if (priors_meta or {}).get("status") != "loaded":
        return "nothing to compare", "no pooled-prior study yet"
    studied = (priors_meta.get("fingerprints") or {}).get(target)
    if not studied:
        return "nothing to compare", f"{target} is not a member of the priors study"
    if studied != fingerprint:
        return ("history rewritten",
                f"history differs from the panel the priors were studied on ({studied})")
    return "same history", f"the panel the priors were studied on ({studied})"


def _render_run(target: str) -> None:
    """d*, h and L, the epoch and the history's fingerprint — read off the bundle."""
    ss = st.session_state
    asof, h, lag = ss.get("asof"), ss.get("horizon"), ss.get("exec_lag")
    epoch, fp = ss.get("epoch"), str(ss.get("fingerprint") or "")
    flag, flag_sub = _history_flag(ss.get("priors_meta") or {}, target, fp)
    card = "system:run"
    render_kpi_strip([
        {"label": "As of d*", "value": f"{asof:%d %b %Y}" if asof is not None else "—",
         "subtext": "Mūla's newest published session: every verdict input is read here",
         "card": card},
        {"label": "Horizon", "value": f"{h}D" if h is not None else "—",
         "subtext": f"from the close {lag} session(s) after d* (EXECUTION_LAG)", "card": card},
        {"label": "Analysis epoch", "value": f"{epoch:%d %b %Y}" if epoch is not None else "—",
         "subtext": "the fixed first session of every history", "card": card},
        {"label": "History fingerprint", "value": fp or "—",
         "subtext": "content hash of the settled prepared rows", "card": card},
        {"label": "Against the priors", "value": flag, "subtext": flag_sub, "card": card,
         "color_class": "warning" if flag == "history rewritten" else "neutral"},
    ], max_cols=5, key="system-run")


def _render_priors() -> None:
    """The pooled-prior study's provenance, or that it has not run."""
    pm = st.session_state.get("priors_meta") or {}
    card = "system:priors"
    if pm.get("status") != "loaded":
        render_kpi_strip([
            {"label": "Pooled priors", "value": "study not yet run",
             "subtext": f"{pm.get('label', 'no pooled prior')} · POOLED_PRIOR "
                        f"{knobs.KNOBS['POOLED_PRIOR'].value or 'unset'}", "card": card},
        ], max_cols=1, key="system-priors")
        return
    render_kpi_strip([
        {"label": "Study", "value": str(pm.get("study") or "—"),
         "subtext": f"run {pm.get('run_date') or '—'} · {pm.get('label') or ''}", "card": card},
        {"label": "Script SHA-256", "value": str(pm.get("sha") or "—")[:12],
         "subtext": f"git {str(pm.get('git_sha') or '—')[:10]}", "card": card},
        {"label": "Studied at", "value": f"h {pm.get('h', '—')} · L {pm.get('exec_lag', '—')}",
         "subtext": "a file studied at another h or L is refused", "card": card},
    ], max_cols=3, key="system-priors")


def _render_coverage(engine) -> None:
    """The newest session's cross-section, and why any session was not valued."""
    ts = getattr(engine, "ts_data", None)
    if ts is None or ts.empty:
        return
    last = ts.iloc[-1]
    printed, sched, admitted = (int(v) if pd.notna(v) else 0 for v in
                                (pd.to_numeric(last[c], errors="coerce")
                                 for c in ("NAvailable", "NScheduled", "NAdmitted")))
    absent = list(getattr(engine, "panel_absent", []) or [])
    reasons = ts["WithheldReason"].astype(str)
    counts = reasons[reasons != ""].value_counts()
    card = "system:health"
    render_kpi_strip([
        {"label": "Newest session · printed", "value": f"{printed} of {sched}",
         "subtext": f"instruments whose markets were open, of {admitted} admitted", "card": card},
        {"label": "Open and not printed", "value": str(len(absent)),
         "subtext": (", ".join(absent[:6]) + (f" +{len(absent) - 6} more" if len(absent) > 6 else "")
                     if absent else "none"),
         "color_class": "warning" if absent else "neutral", "card": card},
        {"label": "Sessions not valued", "value": str(int(counts.sum())),
         "subtext": " · ".join(f"{k} {int(v)}" for k, v in counts.items()) or "none", "card": card},
    ], max_cols=3, key="system-coverage")


def _render_caches_and_circuits() -> None:
    """Two-tier cache statistics and circuit-breaker state for each external service."""
    items = []
    for cache in all_caches():
        s = cache.stats()
        hit_pct = s["hit_rate"] * 100.0
        total = s["hits"] + s["misses"]
        # Color: green ≥70% hit rate, amber 30-70%, rose <30% (or no data)
        if total == 0:
            color_cls = "neutral"
        elif hit_pct >= 70:
            color_cls = "success"
        elif hit_pct >= 30:
            color_cls = "warning"
        else:
            color_cls = "danger"
        last_fetch = s["last_fetch_time"]
        if last_fetch:
            mins = (pd.Timestamp.now().timestamp() - last_fetch) / 60.0
            sub = f"{s['disk_entries']} disk · last fetch {mins:.0f}m ago"
        else:
            sub = f"{s['disk_entries']} disk · no fetch this run"
        items.append({
            "label": f"CACHE · {s['namespace'].upper()}",
            "value": f"{hit_pct:.0f}%" if total else "—", "subtext": sub,
            "color_class": color_cls, "card": "system:health",
            "tooltip": (f"{s['hits']} hits / {s['misses']} misses · "
                        f"{s['stale_hits']} stale-fallback · "
                        f"{s['writes']} writes · TTL {s['ttl_seconds']}s · version {s['version']}"),
        })
    render_kpi_strip(items, max_cols=max(1, len(items)), key="system-caches")

    items = []
    for cb in all_circuits():
        st_dict = cb.get_state()
        state = st_dict["state"]
        if state == CircuitState.CLOSED.value:
            color_cls = "success"
            label_val = "CLOSED"
        elif state == CircuitState.HALF_OPEN.value:
            color_cls = "warning"
            label_val = "HALF-OPEN"
        else:
            color_cls = "danger"
            label_val = "OPEN"
        last_fail = st_dict["last_failure"]
        if last_fail:
            mins = (pd.Timestamp.now().timestamp() - last_fail) / 60.0
            sub = f"{st_dict['failure_count']} fails · last {mins:.0f}m ago"
        else:
            sub = f"{st_dict['success_count']} successful requests"
        items.append({
            "label": f"CIRCUIT · {st_dict['name'].upper()}", "value": label_val, "subtext": sub,
            "color_class": color_cls, "card": "system:health",
            "tooltip": (f"Threshold: {st_dict['failure_threshold']} failures · "
                        f"Recovery: {st_dict['recovery_timeout']:.0f}s · "
                        f"OPEN blocks requests; HALF-OPEN allows 1 test request after the "
                        f"recovery timeout."),
        })
    render_kpi_strip(items, max_cols=max(1, len(items)), key="system-circuits")


def _knobs_table() -> pd.DataFrame:
    """The KNOBS registry, one row per knob, as core.knobs declares it."""
    return pd.DataFrame([{
        "Knob": k.name, "Value": str(k.value), "Kind": k.kind, "Consumer": k.consumer,
        "Horizon coupling": k.horizon_coupling,
        "Reason or study": f"{k.study}@{k.date}" if k.kind == "measured" else k.reason,
    } for k in knobs.KNOBS.values()]).set_index("Knob")


def _roles_table() -> pd.DataFrame:
    """The ROLES registry, one row per role, as core.roles declares it, with the chip
    its kind carries at the role's own target (the call where it serves no one
    role-holder) and how many registered cards (``CARDS``) show it."""
    shown = pd.Series([role for role, _target in CARDS.values()]).value_counts()
    return pd.DataFrame([{
        "Role": r.key, "Kind": r.kind, "Serves": r.target or "—", "Title": r.title,
        "Graded by": r.grade_key or "—",
        "Chip": CHIPS[r.kind].split(" · ")[0].format(target=r.target or "call"),
        "Cards": int(shown.get(r.key, 0)), "Consumer": r.consumer,
    } for r in ROLES.values()]).set_index("Role")


def render_diagnostics_tab(engine, target: str) -> None:
    """System — the run's identity, the priors, the data's health, the two registries."""
    render_section_header(
        "The Run",
        "Which session the verdict is about, on which horizon and lag, from which fixed start, "
        "and which history was valued.",
        icon="target", accent="cyan", card="system:run",
    )
    _render_run(target)

    render_section_header(
        "Pooled Priors",
        "The catalogue study whose class priors every posterior is shrunk toward, read as "
        "data; without it every grade is the instrument's own. Its results are on the Grades "
        "page.",
        icon="layers", accent="cyan", card="system:priors",
    )
    _render_priors()

    render_section_header(
        "Data Health",
        "The newest session's cross-section, the sessions the engine did not value and why, "
        "and the state of the cache and each external service.",
        icon="database", accent="cyan", card="system:health",
    )
    _render_coverage(engine)
    _render_caches_and_circuits()

    render_section_header(
        "The KNOBS Registry",
        "Every constant that can move what Tattva issues, declared once (core.knobs): its "
        "value, its kind, the role it moves, how it couples to the horizon, and why it is set "
        "where it is.",
        icon="cpu", accent="cyan", card="system:knobs",
    )
    render_table_panel(_knobs_table(), "system-knobs", units=f"{len(knobs.KNOBS)} knobs",
                       windowed=False, index_label="Knob", max_height=520)
    render_note("structural: fixed by an argument, never tuned · structural-inherited: an older "
                "value kept without a study behind it, not evidence · measured: set by a "
                "committed study, named with its run date.")

    render_section_header(
        "The Roles",
        "What every part of the decision is for (core.roles): the role it plays, the "
        "role-holder it serves or audits, the grade that grades it, the chip every card of it "
        "carries, how many cards show it, and what reads it.",
        icon="grid", accent="cyan", card="system:roles",
    )
    render_table_panel(_roles_table(), "system-roles",
                       units=f"{len(ROLES)} roles · {len(CARDS)} cards",
                       windowed=False, index_label="Role", max_height=520)

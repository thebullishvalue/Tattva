"""
Causal regime inference: online hidden Markov model with forward filtering.
=================================================================

Regime labels published by most research code are produced by Baum-Welch,
which runs a *backward* pass: the label assigned to 2018-02-05 is computed
using data from 2019.  That is smoothing, and it repaints.  A regime series
built that way looks far more decisive than anything that was knowable at
the time.

This module deliberately uses only the forward filter,

    alpha_t(j) = P(S_t = j | y_1..y_t) ,

and updates the emission and transition parameters by recursive (online) EM
with exponential forgetting -- the filtered approximation to the E-step of
Cappe (2011).  Nothing here ever revisits a past state probability.

The number of states is not chosen.  A bank of models with K = 2, 3, 4 is
run in parallel and weighted by accumulated one-step predictive likelihood,
which is a proper scoring rule; a continuous stress index is then formed as
a weight-averaged expectation, so the reported quantity does not jump when
the preferred K changes.

References
----------
Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle." *Econometrica* 57(2).
Cappe, O. (2011). "Online EM Algorithm for Hidden Markov Models."
    *J. Computational and Graphical Statistics* 20(3).
Ang, A. & Timmermann, A. (2012). "Regime Changes and Financial Markets."
    *Annual Review of Financial Economics* 4.
"""

from __future__ import annotations

import math

import numpy as np

from analytics.causal import EPS, halflife_to_lambda

#: Observations absorbed before the filter emits anything.  Set by
#: identifiability of K Gaussian components, not by preference.
BURN_IN = 60


class _HMM:
    """Single K-state Gaussian HMM, filtered and updated online."""

    def __init__(self, k: int, d: int, halflife: float = 504.0) -> None:
        self.k = k
        self.d = d
        self.lam = halflife_to_lambda(halflife)
        self.alpha = np.full(k, 1.0 / k)
        # weakly informative, symmetric start; the burn-in fixes the scale
        self.A = np.full((k, k), 0.1 / max(k - 1, 1))
        np.fill_diagonal(self.A, 0.9)
        self.mu = np.zeros((k, d))
        self.var = np.ones((k, d))
        self.n = np.full(k, 1.0)
        self.m1 = np.zeros((k, d))
        self.m2 = np.ones((k, d))
        self.N = self.A.copy()
        self.buf: list[np.ndarray] = []
        self.ready = False
        self.loglik = 0.0
        self.order = np.arange(k)

    # -- deterministic initialisation ---------------------------------------
    def _initialise(self) -> None:
        Y = np.asarray(self.buf)                      # (T, d)
        key = Y[:, -1]                                # last column = log-vol
        qs = np.quantile(key, np.linspace(0, 1, self.k + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        for j in range(self.k):
            sel = (key > qs[j]) & (key <= qs[j + 1])
            if sel.sum() < 3:
                sel = np.ones(len(Y), dtype=bool)
            self.mu[j] = Y[sel].mean(axis=0)
            v = Y[sel].var(axis=0)
            self.var[j] = np.maximum(v, 1e-4)
            self.n[j] = float(sel.sum())
            self.m1[j] = self.mu[j] * self.n[j]
            self.m2[j] = (self.var[j] + self.mu[j] ** 2) * self.n[j]
        self.ready = True

    def _emission(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Relative emission likelihoods and the log offset removed from them.

        The offset has to be returned, not discarded: the filtered likelihood
        is what weights this model against the ones with a different state
        count, and models with different K carry different offsets.  Dropping
        it would make the comparison meaningless in exactly the way that is
        hard to notice, because the weights would still look plausible.
        """
        v = np.maximum(self.var, 1e-4)
        ll = -0.5 * np.sum(np.log(2.0 * math.pi * v) + (y[None, :] - self.mu) ** 2 / v,
                           axis=1)
        mx = float(ll.max())
        return np.exp(np.clip(ll - mx, -60.0, 0.0)), mx

    def step(self, y: np.ndarray) -> dict | None:
        """Filter one observation and absorb it. Returns pre-update diagnostics."""
        if not self.ready:
            self.buf.append(np.asarray(y, dtype=float))
            if len(self.buf) >= BURN_IN:
                self._initialise()
            return None

        pred = self.alpha @ self.A                     # P(S_t | y_1..t-1)
        b, offset = self._emission(y)
        joint = pred * b
        tot = float(joint.sum())
        if tot <= EPS or not np.isfinite(tot):
            return {"alpha": self.alpha.copy(), "pred": pred, "loglik": 0.0}
        alpha_new = joint / tot
        self.loglik = math.log(max(tot, 1e-300)) + offset

        # filtered two-slice statistic: the causal stand-in for the smoothed
        # xi of Baum-Welch
        xi = (self.alpha[:, None] * self.A) * b[None, :]
        xi /= max(xi.sum(), EPS)

        lam = self.lam
        self.N = lam * self.N + xi
        self.A = self.N / np.maximum(self.N.sum(axis=1, keepdims=True), EPS)

        self.n = lam * self.n + alpha_new
        self.m1 = lam * self.m1 + alpha_new[:, None] * y[None, :]
        self.m2 = lam * self.m2 + alpha_new[:, None] * (y[None, :] ** 2)
        nn = np.maximum(self.n, 1e-3)[:, None]
        self.mu = self.m1 / nn
        self.var = np.maximum(self.m2 / nn - self.mu ** 2, 1e-4)

        out = {"alpha": alpha_new.copy(), "pred": pred, "loglik": self.loglik}
        self.alpha = alpha_new
        return out

    def vol_rank(self) -> np.ndarray:
        """States ranked 0..1 by their emission volatility level."""
        key = self.mu[:, -1]
        order = np.argsort(key)
        rank = np.empty(self.k)
        rank[order] = np.arange(self.k) / max(self.k - 1, 1)
        return rank

    def drift(self) -> np.ndarray:
        return self.mu[:, 0]


class RegimeFilter:
    """Bank of online HMMs, averaged by causal predictive likelihood."""

    K_GRID = (2, 3, 4)

    def __init__(self, d: int = 2) -> None:
        self.models = [_HMM(k, d) for k in self.K_GRID]
        self.logw = np.zeros(len(self.K_GRID))
        self.d = d
        self.t = 0

    def update(self, y: np.ndarray) -> dict:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        outs = [m.step(y) for m in self.models]
        self.t += 1

        lls = np.array([o["loglik"] if o else 0.0 for o in outs])
        if np.all(np.isfinite(lls)) and any(o is not None for o in outs):
            self.logw = 0.99 * self.logw + lls
            self.logw -= self.logw.max()
        w = np.exp(self.logw - self.logw.max())
        w = w / max(w.sum(), EPS)

        stress = 0.0
        drift = 0.0
        switch = 0.0
        active = 0.0
        ready = False
        for wi, m, o in zip(w, self.models, outs):
            if o is None:
                continue
            ready = True
            a = o["alpha"]
            r = m.vol_rank()
            stress += wi * float(a @ r)
            drift += wi * float(a @ m.drift())
            switch += wi * float(1.0 - np.sum(a * np.diag(m.A)))
            active += wi * float(np.sum(a > 0.05))

        if not ready:
            return {"ready": False, "stress": 0.5, "drift": 0.0,
                    "switch_prob": 0.0, "label": "initialising",
                    "k_best": 0, "state_probs": np.zeros(0), "entropy": 1.0,
                    "persistence": 0.0, "model_weights": w}

        best = int(np.argmax(w))
        mb = self.models[best]
        ab = mb.alpha
        ent = float(-np.sum(ab * np.log(np.maximum(ab, 1e-12))) / math.log(max(mb.k, 2)))
        persistence = float(ab @ np.diag(mb.A))

        return {
            "ready": True,
            "stress": float(np.clip(stress, 0.0, 1.0)),
            "drift": float(drift),
            "switch_prob": float(np.clip(switch, 0.0, 1.0)),
            "label": self._label(stress, drift),
            "k_best": mb.k,
            "state_probs": ab.copy(),
            "state_vol_rank": mb.vol_rank(),
            "state_drift": mb.drift().copy(),
            "entropy": ent,
            "persistence": persistence,
            "model_weights": w,
        }

    @staticmethod
    def _label(stress: float, drift: float) -> str:
        if stress < 0.34:
            tier = "Calm"
        elif stress < 0.67:
            tier = "Transitional"
        else:
            tier = "Turbulent"
        bias = "risk-on" if drift > 0.02 else ("risk-off" if drift < -0.02 else "neutral")
        return f"{tier} / {bias}"


REGIME_ORDER = ["Calm / risk-on", "Calm / neutral", "Calm / risk-off",
                "Transitional / risk-on", "Transitional / neutral",
                "Transitional / risk-off", "Turbulent / risk-on",
                "Turbulent / neutral", "Turbulent / risk-off", "initialising"]

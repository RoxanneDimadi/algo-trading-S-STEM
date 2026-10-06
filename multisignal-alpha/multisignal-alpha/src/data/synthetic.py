"""Synthetic cross-sectional panel with PLANTED signals.

Why this module exists
----------------------
The roadmap's rule: validate the evaluation harness and the backtester on data
where the truth is known, BEFORE touching real data. This simulator plants:

  * several signals with known positive monthly betas (momentum-, liquidity-,
    volatility-, value-, quality-like AR(1) processes),
  * one signal with beta EXACTLY zero (`sig_dead`) -- the placebo. If the
    harness reports it as significant, the harness is broken.
  * McLean-Pontiff-style decay: each signal's beta is multiplied down after
    its (synthetic) sample_end and again after its pub_date, so the decay
    module has real structure to recover.
  * one nonlinear interaction term (default: momentum x value). A linear
    model cannot capture it; gradient-boosted trees should. This is the
    Gu-Kelly-Xiu comparison in miniature, with ground truth.

Return generation (data-generating process)
-------------------------------------------
    r[t+1, i] = sum_k beta_k(t) * z_k[t, i]
              + beta_int(t) * z_a[t, i] * z_b[t, i]      (if enabled)
              + b_mkt[i] * mkt[t+1]
              + eps[t+1, i]

where z are cross-sectionally standardized AR(1) signals OBSERVED at t and
returns realize over (t, t+1]. The panel row at date t therefore carries
signals known at t, the contemporaneous return over month t, and fwd_ret =
the month-(t+1) return -- i.e., point-in-time alignment is correct by
construction, and any evaluation that accidentally uses contemporaneous
information will disagree with the planted truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _standardize_cross_section(x: np.ndarray) -> np.ndarray:
    """Z-score each row (date) across tickers."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def make_synthetic_panel(cfg: dict, seed: int = 42):
    # one return-generating equation, written out in full
    # pylint: disable=too-many-locals
    """Build the synthetic panel.

    Returns
    -------
    panel : DataFrame [date, ticker, <signals...>, ret, fwd_ret]
        Long format. Signals at date t; `ret` is the month-t return; `fwd_ret`
        is the month-(t+1) return (NaN on the final date).
    factors : DataFrame indexed by date [mkt_rf, smb, hml, rmw, cma, mom]
        A synthetic factor panel. mkt_rf is the true simulated market factor;
        the rest are pure noise, uncorrelated with the planted signals, so a
        correctly-implemented factor regression should attribute strategy
        returns to alpha, not to these factors.
    meta : DataFrame indexed by signal [true_beta, ar, sample_end, pub_date]
        Ground truth for tests and the decay module.
    """
    scfg = cfg["data"]["synthetic"]
    rng = np.random.default_rng(seed)

    from src.utils.stats import month_end_freq

    dates = pd.date_range(scfg["start"], scfg["end"], freq=month_end_freq())
    T, N = len(dates), int(scfg["n_tickers"])
    tickers = [f"T{i:04d}" for i in range(N)]

    sig_cfg = scfg["signals"]
    names = list(sig_cfg.keys())
    K = len(names)

    # --- AR(1) signal paths, standardized within each date ------------------
    Z = np.zeros((K, T, N))
    for k, name in enumerate(names):
        rho = float(sig_cfg[name]["ar"])
        z = rng.standard_normal(N)
        for t in range(T):
            z = rho * z + np.sqrt(max(1e-12, 1 - rho**2)) * \
                rng.standard_normal(N)
            Z[k, t] = z
        Z[k] = _standardize_cross_section(Z[k])

    # --- time-varying betas with post-sample / post-publication decay -------
    d_samp = float(scfg["decay_post_sample"])
    d_pub = float(scfg["decay_post_pub"])
    B = np.zeros((K, T))
    for k, name in enumerate(names):
        b = float(sig_cfg[name]["beta"])
        s_end = pd.Timestamp(sig_cfg[name]["sample_end"])
        p_date = pd.Timestamp(sig_cfg[name]["pub_date"])
        mult = np.ones(T)
        mult[dates > s_end] = d_samp
        mult[dates > p_date] = d_pub
        B[k] = b * mult

    # --- market factor and betas --------------------------------------------
    m = scfg["market"]
    mkt = rng.normal(m["mean_monthly"], m["vol_monthly"], size=T)
    b_mkt = rng.normal(1.0, m["beta_dispersion"], size=N)

    # --- returns: r[t] driven by signals observed at t-1 ---------------------
    idio = float(scfg["idio_vol_monthly"])
    R = np.zeros((T, N))
    icfg = scfg.get("interaction", {"enabled": False})
    if icfg.get("enabled", False):
        ia = names.index(icfg["cols"][0])
        ib = names.index(icfg["cols"][1])
        b_int = float(icfg["beta"])
    for t in range(1, T):
        drift = np.einsum("k,kn->n", B[:, t - 1], Z[:, t - 1, :])
        if icfg.get("enabled", False):
            inter = Z[ia, t - 1, :] * Z[ib, t - 1, :]
            drift = drift + b_int * (inter - inter.mean())
        R[t] = drift + b_mkt * mkt[t] + rng.normal(0.0, idio, size=N)

    # --- assemble long panel ---------------------------------------------
    frames = []
    for t, dt in enumerate(dates):
        f = pd.DataFrame({"date": dt, "ticker": tickers, "ret": R[t]})
        for k, name in enumerate(names):
            f[name] = Z[k, t]
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel["fwd_ret"] = panel.groupby("ticker")["ret"].shift(-1)
    # Drop the first date: its `ret` is undefined (the DGP loop starts at
    # t=1), and a constant all-zero return cross-section is degenerate for
    # rank statistics. Signals on that date contribute nothing predictive
    # that date t=1 doesn't already carry.
    panel = panel[panel["date"] > dates[0]].reset_index(drop=True)

    # --- synthetic factor panel (mkt real, rest noise) --------------------
    factors = pd.DataFrame(
        {
            "mkt_rf": mkt,
            "smb": rng.normal(0, 0.020, T),
            "hml": rng.normal(0, 0.020, T),
            "rmw": rng.normal(0, 0.015, T),
            "cma": rng.normal(0, 0.015, T),
            "mom": rng.normal(0, 0.030, T),
        },
        index=dates,
    )
    factors.index.name = "date"

    meta = pd.DataFrame(
        {
            "true_beta": [float(sig_cfg[n]["beta"]) for n in names],
            "ar": [float(sig_cfg[n]["ar"]) for n in names],
            "sample_end": [pd.Timestamp(sig_cfg[n]["sample_end"])
                           for n in names],
            "pub_date": [pd.Timestamp(sig_cfg[n]["pub_date"]) for n in names],
        },
        index=pd.Index(names, name="signal"),
    )
    return panel, factors, meta

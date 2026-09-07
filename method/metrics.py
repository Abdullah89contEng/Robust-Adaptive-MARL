"""Evaluation metrics for the experiments of thesis Chapter 11.

Pure functions over per-episode arrays. Thresholds match the thesis
"Metrics" section and are fixed here, not tuned on results:
  * detection window W = 10 steps after a true change
  * stationary-segment length for a true negative = 20 steps
  * recovery band = +/- 10% of the 20-step pre-change average return
"""

from __future__ import annotations

import numpy as np

W_DETECT = 10          # a flag within W steps of a true change is a true positive
SEG_TN = 20            # a stationary SEG_TN-step run with no flag is a true negative
REC_BAND = 0.10        # recovery band as a fraction of the pre-change level
REC_WINDOW = 20        # moving-average window for return, and pre-change reference window


# --------------------------------------------------------------------------
# change-point detection
# --------------------------------------------------------------------------
def detection_metrics(reset_steps, true_change, horizon, *, W=W_DETECT, seg_tn=SEG_TN):
    """One episode with a single true change at `true_change` (or None for a
    no-change episode). `reset_steps` = sorted list of steps at which the
    detector fired a reset for the agent under test.

    Returns dict with tp, fp, fn, tn, delay (nan if no detection).
    """
    reset_steps = sorted(int(s) for s in reset_steps)
    out = dict(tp=0, fp=0, fn=0, tn=0, delay=float("nan"))

    if true_change is None:
        # every non-overlapping seg_tn-step stationary window with no flag is a TN;
        # any flag is a FP.
        out["fp"] = len(reset_steps)
        n_windows = max(1, horizon // seg_tn)
        flagged_windows = {s // seg_tn for s in reset_steps}
        out["tn"] = sum(1 for w in range(n_windows) if w not in flagged_windows)
        return out

    tc = int(true_change)
    pre = [s for s in reset_steps if s < tc]
    post = [s for s in reset_steps if tc <= s <= tc + W]
    late = [s for s in reset_steps if s > tc + W]
    out["fp"] = len(pre) + len(late)
    if post:
        out["tp"] = 1
        out["delay"] = float(post[0] - tc)
    else:
        out["fn"] = 1
    # a TN credit if the pre-change stretch (length tc) had no flag
    if not pre and tc >= seg_tn:
        out["tn"] = 1
    return out


def aggregate_detection(rows):
    """rows = list of detection_metrics dicts. Returns TPR, TNR, FA-rate,
    F1, mean/median delay."""
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    delays = np.array([r["delay"] for r in rows if not np.isnan(r["delay"])], float)
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * tpr / (prec + tpr) if prec and tpr and not np.isnan(prec + tpr) else float("nan")
    return dict(
        tpr=tpr, tnr=tnr, false_alarm_rate=1 - tnr if not np.isnan(tnr) else float("nan"),
        missed_detection_rate=1 - tpr if not np.isnan(tpr) else float("nan"),
        precision=prec, f1=f1,
        delay_mean=float(delays.mean()) if delays.size else float("nan"),
        delay_median=float(np.median(delays)) if delays.size else float("nan"),
        n_episodes=len(rows), n_detected=int(delays.size),
    )


# --------------------------------------------------------------------------
# recovery time
# --------------------------------------------------------------------------
def recovery_time(step_rewards, true_change, horizon, *, band=REC_BAND, window=REC_WINDOW):
    """`step_rewards` = per-step team reward (sum over agents), length ~horizon.
    Returns (recovery_steps, censored): steps after `true_change` until the
    `window`-step moving average first re-enters +/- `band` of the pre-change
    `window`-step average; censored=True (and recovery_steps=horizon-true_change)
    if it never does within the episode.
    """
    r = np.asarray(step_rewards, float)
    tc = int(true_change)
    if tc < window or tc >= len(r):
        return float("nan"), True
    ref = r[tc - window:tc].mean()
    lo, hi = ref - abs(ref) * band - 1e-8, ref + abs(ref) * band + 1e-8
    ma = np.convolve(r, np.ones(window) / window, mode="valid")  # index k = mean of r[k:k+window]
    for k in range(tc, min(len(ma), horizon)):
        if lo <= ma[k] <= hi:
            return float(k - tc), False
    return float(horizon - tc), True


def aggregate_recovery(pairs):
    """pairs = list of (recovery_steps, censored). Reports the mean over the
    RECOVERED episodes and the censoring fraction separately."""
    rec = [s for s, c in pairs if not c and not np.isnan(s)]
    n = len(pairs)
    cens = sum(1 for _, c in pairs if c)
    return dict(
        recovery_mean=float(np.mean(rec)) if rec else float("nan"),
        recovery_median=float(np.median(rec)) if rec else float("nan"),
        censored_fraction=cens / n if n else float("nan"),
        n_recovered=len(rec), n_episodes=n,
    )


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------
def degradation(nominal_return, attacked_return):
    """Absolute and relative drop. `*_return` are scalars (e.g. IQMs)."""
    absolute = nominal_return - attacked_return
    relative = absolute / abs(nominal_return) if nominal_return else float("nan")
    return dict(degradation_abs=float(absolute), degradation_rel=float(relative))


# --------------------------------------------------------------------------
# aggregation with bootstrap CI (used across seeds)
# --------------------------------------------------------------------------
def iqm(x):
    x = np.sort(np.asarray(x, float))
    if x.size == 0:
        return float("nan")
    lo, hi = int(0.25 * x.size), int(np.ceil(0.75 * x.size))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def bootstrap_ci(x, *, stat=iqm, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.array([stat(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)])
    return (float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))


def summarize(values, name="value"):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    lo, hi = bootstrap_ci(v)
    return {f"{name}_iqm": iqm(v), f"{name}_mean": float(v.mean()) if v.size else float("nan"),
            f"{name}_std": float(v.std()) if v.size else float("nan"),
            f"{name}_ci_lo": lo, f"{name}_ci_hi": hi, f"{name}_n": int(v.size)}

"""Meta-test evaluation: change-point detector performance AND recovery time.

Runs scripted meta-test episodes on a frozen Phase-1 checkpoint + trained
detector and reports, in one pass:

  DETECTOR
    * threshold sweep: TPR (detection rate), FPR (false-alarm rate), F1,
      mean detection delay -- from find_first_change / calculate_errors
      (third-party/InDiD metrics), over change AND no-change episodes.
    * the AS-DEPLOYED operating point: p >= threshold_C held
      trigger_persistence steps -> reset (MetaTestRunner's own rule).

  RECOVERY  (team-return dip after the switch, with vs. without the reset)
    * baseline = trailing-mean team return over a window ending at the
      switch; recovery time = first post-switch step whose trailing mean is
      back within `band` of baseline and stays there for `window` steps
      (censored at the horizon if it never recovers).
    * transient cost = area of the dip; re-exploration overhead = post-switch
      steps the target agent spends in `explore` mode.
    * "with detector" = normal MetaTestConfig; "passive" = the same run with
      the reset disabled (trigger_persistence -> inf), i.e. the
      `- change-point detector` ablation.

Usage:
    python scripts/recovery_eval.py \
        --phase1-ckpt runs/full_.../phase1_checkpoint_9500.pt \
        --detector    runs/full_.../phase2_detector.pt \
        --repeats 20 --horizon 140 --change-step 40 --out-dir runs/receval_<ts>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.scenario import Scenario
from method.io import load_phase1, load_detector
from method.training.phase1 import Phase1Config
from method.training.meta_test import MetaTestRunner, MetaTestConfig, RegimeChangeEvent
from method.training.regime import Regime
from method.detector.indid import find_first_change, calculate_errors, f1_score


# --------------------------------------------------------------------------
def run_episode(runner: MetaTestRunner, horizon: int, events: list) -> dict:
    runner.reset_state()
    runner._regime_events.clear()
    runner._attack_events.clear()
    runner.schedule(events)
    p, reset, mode, team_r = [], [], [], []
    for _ in range(horizon):
        s = runner.step(render=False)
        p.append([x if x is not None else np.nan for x in s.detector_p])
        reset.append(list(s.reset_fired))
        mode.append(list(s.mode))
        team_r.append(float(np.sum(s.rewards)))
    return dict(
        p=np.asarray(p, float),                 # (T, n_agents)
        reset=np.asarray(reset, bool),          # (T, n_agents)
        mode=np.asarray(mode, object),          # (T, n_agents)
        team_r=np.asarray(team_r, float),       # (T,)
    )


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty(len(x))
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def recovery_metrics(team_r: np.ndarray, change: int, w: int, band: float) -> dict:
    sm = trailing_mean(team_r, w)
    pre = sm[max(0, change - w):change]
    baseline = float(pre.mean()) if len(pre) else np.nan
    target = baseline - band * (abs(baseline) + 1e-6)          # "no worse than band below pre-switch"
    T = len(sm)
    t_rec = None
    for t in range(change, T):
        seg = sm[t:min(T, t + w)]
        if sm[t] >= target and len(seg) >= min(w, T - change) and np.all(seg >= target):
            t_rec = t
            break
    recovered = t_rec is not None
    rt = (t_rec - change) if recovered else (T - change)       # censored
    transient = float(np.clip(baseline - sm[change:], 0.0, None).sum())
    return dict(baseline=baseline, recovered=recovered, recovery_time=rt, transient_cost=transient)


def deployed_point(change_traces, nochange_traces, target: int, change: int, seq_len: int) -> dict:
    """Classify each episode by the AS-DEPLOYED reset (persistence-filtered)."""
    real, pred = [], []
    for ep in change_traces:
        fr = np.where(ep["reset"][:, target])[0]
        real.append(change)
        pred.append(int(fr[0]) if len(fr) else -1)
    for ep in nochange_traces:
        fr = np.where(ep["reset"][:, target])[0]
        real.append(-1)
        pred.append(int(fr[0]) if len(fr) else -1)
    real, pred = torch.tensor(real), torch.tensor(pred)
    TN, FP, FN, TP, _fpd, dl = calculate_errors(real, pred, seq_len)
    tpd = dl[(real <= pred) & (real != -1)].float()
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                tpr=TP / (TP + FN) if TP + FN else np.nan,
                fpr=FP / (FP + TN) if FP + TN else np.nan,
                f1=f1_score(TN, FP, FN, TP),
                delay=float(tpd.mean()) if tpd.numel() else np.nan)


def threshold_sweep(p_change, real_change, p_nochange, seq_len, thrs) -> list[dict]:
    pc = torch.tensor(p_change)
    pn = torch.tensor(p_nochange) if len(p_nochange) else torch.zeros(0, seq_len)
    real = torch.tensor(list(real_change) + [-1] * len(pn))
    rows = []
    for thr in thrs:
        pred_c = find_first_change(pc > thr)
        pred_n = find_first_change(pn > thr) if len(pn) else torch.zeros(0, dtype=torch.long)
        pred = torch.cat([pred_c, pred_n])
        TN, FP, FN, TP, _fpd, dl = calculate_errors(real, pred, seq_len)
        tpd = dl[(real <= pred) & (real != -1)].float()
        rows.append(dict(thr=float(thr), TP=TP, FP=FP, FN=FN, TN=TN,
                         tpr=TP / (TP + FN) if TP + FN else np.nan,
                         fpr=FP / (FP + TN) if FP + TN else np.nan,
                         f1=f1_score(TN, FP, FN, TP),
                         delay=float(tpd.mean()) if tpd.numel() else np.nan))
    return rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-ckpt", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--config-file", default=None, help="default: read from the ckpt's args.json")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=140)
    ap.add_argument("--change-step", type=int, default=40)
    ap.add_argument("--target-agent", type=int, default=0)
    ap.add_argument("--smooth-window", type=int, default=12)
    ap.add_argument("--recovery-band", type=float, default=0.10)
    ap.add_argument("--nominal", type=float, nargs=2, default=(1.10, 0.16), metavar=("MASS", "FRICTION"))
    ap.add_argument("--shifted", type=float, nargs=2, default=(1.52, 0.36), metavar=("MASS", "FRICTION"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ck = Path(args.phase1_ckpt).resolve()
    cfg_file = args.config_file
    if cfg_file is None:
        aj = ck.parent / "args.json"
        cfg_file = json.loads(aj.read_text()).get("config_file", "world_config.yaml") if aj.is_file() else "world_config.yaml"
    out = Path(args.out_dir) if args.out_dir else (ck.parents[1] / f"receval_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"phase1 ckpt : {ck}\ndetector    : {Path(args.detector).resolve()}\nconfig      : {cfg_file}\nout dir     : {out}")

    trainer = load_phase1(ck, scenario_factory=lambda: Scenario(config_file=cfg_file), config=None)
    detector = load_detector(args.detector, phase1_trainer=trainer, config=None).eval()
    NA, H, C, TA = trainer.n_agents, args.horizon, args.change_step, args.target_agent
    nominal, shifted = Regime(*args.nominal), Regime(*args.shifted)
    print(f"n_agents={NA}  detector_raw={detector.input_proj.in_features}  window={detector.window}")

    factory = lambda: Scenario(config_file=cfg_file)
    run_with = MetaTestRunner(trainer, detector, factory, MetaTestConfig())
    run_pass = MetaTestRunner(trainer, detector, factory,
                              MetaTestConfig(threshold_C=1.01, trigger_persistence=10 ** 9))

    nominal_all = [RegimeChangeEvent(0, i, nominal) for i in range(NA)]
    change_ev = nominal_all + [RegimeChangeEvent(C, TA, shifted)]

    # ---- collect episodes -------------------------------------------------
    ch_with, ch_pass, nochange = [], [], []
    for r in range(args.repeats):
        torch.manual_seed(1000 + r); ch_with.append(run_episode(run_with, H, change_ev))
        torch.manual_seed(1000 + r); ch_pass.append(run_episode(run_pass, H, change_ev))
        torch.manual_seed(5000 + r); nochange.append(run_episode(run_with, H, nominal_all))
        if r % 5 == 0:
            print(f"  [{time.time()-t0:5.0f}s] episode set {r+1}/{args.repeats}")

    # ---- detector: threshold sweep + deployed point ---------------------
    p_change = np.stack([e["p"][:, TA] for e in ch_with])
    p_noch = np.stack([e["p"][:, TA] for e in nochange]) if nochange else np.zeros((0, H))
    thrs = np.round(np.arange(0.10, 0.96, 0.05), 2)
    sweep = threshold_sweep(p_change, [C] * len(ch_with), p_noch, H, thrs)
    best = max(sweep, key=lambda x: (x["f1"] if not np.isnan(x["f1"]) else -1))
    deployed = deployed_point(ch_with, nochange, TA, C, H)

    # ---- recovery ------------------------------------------------------
    def agg(eps):
        m = [recovery_metrics(e["team_r"], C, args.smooth_window, args.recovery_band) for e in eps]
        rec = [x for x in m if x["recovered"]]
        return dict(
            n=len(m),
            recovered=len(rec),
            nonrecovery_rate=1 - len(rec) / len(m),
            rt_mean=float(np.mean([x["recovery_time"] for x in m])),
            rt_median=float(np.median([x["recovery_time"] for x in m])),
            rt_mean_recovered=float(np.mean([x["recovery_time"] for x in rec])) if rec else np.nan,
            transient_mean=float(np.mean([x["transient_cost"] for x in m])),
            baseline_mean=float(np.mean([x["baseline"] for x in m])),
        )
    rec_with, rec_pass = agg(ch_with), agg(ch_pass)
    reexplore = float(np.mean([np.sum((e["mode"][C:, TA] == "explore")) for e in ch_with]))

    # ---- report -----------------------------------------------------
    lines = []
    def pr(s=""):
        print(s); lines.append(s)
    pr("\n" + "=" * 68)
    pr(f"DETECTOR  (target agent {TA}, true change @ {C}, {args.repeats} change + {len(nochange)} no-change eps)")
    pr("=" * 68)
    pr(f"{'thr':>5} {'TPR':>6} {'FPR':>6} {'F1':>6} {'delay':>7}")
    for row in sweep:
        pr(f"{row['thr']:5.2f} {row['tpr']:6.2f} {row['fpr']:6.2f} {row['f1']:6.2f} {row['delay']:7.1f}")
    pr(f"best-F1 @ thr {best['thr']:.2f}: F1={best['f1']:.3f} TPR={best['tpr']:.2f} FPR={best['fpr']:.2f} delay={best['delay']:.1f}")
    pr(f"as-deployed (threshold_C={run_with.cfg.threshold_C}, persistence={run_with.cfg.trigger_persistence}): "
       f"F1={deployed['f1']:.3f} TPR={deployed['tpr']:.2f} FPR={deployed['fpr']:.2f} delay={deployed['delay']:.1f} "
       f"(TP{deployed['TP']} FP{deployed['FP']} FN{deployed['FN']} TN{deployed['TN']})")
    pr("\n" + "=" * 68)
    pr("RECOVERY  (team-return dip after the switch)")
    pr("=" * 68)
    pr(f"{'':<22}{'with detector':>16}{'passive':>16}")
    for k, lab in [("rt_mean", "recovery time (mean, censored)"),
                   ("rt_median", "recovery time (median)"),
                   ("rt_mean_recovered", "recovery time | recovered"),
                   ("nonrecovery_rate", "non-recovery rate"),
                   ("transient_mean", "transient cost (dip area)"),
                   ("baseline_mean", "pre-switch baseline")]:
        pr(f"{lab:<22}{rec_with[k]:>16.2f}{rec_pass[k]:>16.2f}")
    pr(f"{'re-exploration steps':<22}{reexplore:>16.2f}{'--':>16}")
    pr(f"\nrecovery-time reduction (passive - with): {rec_pass['rt_mean'] - rec_with['rt_mean']:+.2f} steps "
       f"({100*(rec_pass['rt_mean']-rec_with['rt_mean'])/max(rec_pass['rt_mean'],1e-6):+.0f}%)")
    (out / "summary.txt").write_text("\n".join(lines))

    # ---- figures --------------------------------------------------
    thr_x = [r["thr"] for r in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(thr_x, [r["tpr"] for r in sweep], "s-", label="TPR (detection)")
    ax[0].plot(thr_x, [r["fpr"] for r in sweep], "^-", label="FPR (false alarm)")
    ax[0].plot(thr_x, [r["f1"] for r in sweep], "o-", label="F1")
    ax[0].axvline(best["thr"], color="k", ls="--", lw=1); ax[0].set_xlabel("threshold"); ax[0].legend()
    ax[0].set_title("Detector vs threshold")
    ax[1].plot(thr_x, [r["delay"] for r in sweep], "d-", color="C3")
    ax[1].axvline(best["thr"], color="k", ls="--", lw=1)
    ax[1].set_xlabel("threshold"); ax[1].set_ylabel("mean detection delay (steps)")
    ax[1].set_title("Detection delay vs threshold")
    fig.tight_layout(); fig.savefig(out / "fig1_detector_vs_threshold.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for ax, e in zip(axes, ch_with[:3]):
        ax.plot(e["p"][:, TA], color="C0", lw=1.5, label="$p_t$ (target)")
        ax.axhline(run_with.cfg.threshold_C, ls="--", color="k", lw=.7)
        ax.axvline(C, color="C3", lw=2, label=f"true change t={C}")
        fr = np.where(e["reset"][:, TA])[0]
        if len(fr):
            ax.axvline(int(fr[0]), color="C2", ls=":", lw=2, label=f"reset t={int(fr[0])}")
        ax.set_ylim(-.05, 1.05); ax.set_ylabel("$p_t$")
    axes[0].legend(fontsize=8, loc="upper left"); axes[-1].set_xlabel("step")
    fig.suptitle("Change point vs detection (as deployed)")
    fig.tight_layout(); fig.savefig(out / "fig2_sawtooth.png", dpi=120); plt.close(fig)

    sm_with = np.stack([trailing_mean(e["team_r"], args.smooth_window) for e in ch_with])
    sm_pass = np.stack([trailing_mean(e["team_r"], args.smooth_window) for e in ch_pass])
    x = np.arange(H)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for sm, c, lab in [(sm_with, "C0", "with detector"), (sm_pass, "C1", "passive")]:
        mu, sd = sm.mean(0), sm.std(0)
        ax.plot(x, mu, color=c, lw=2, label=lab)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=.15)
    ax.axvline(C, color="C3", lw=2, label=f"switch t={C}")
    ax.axhline(rec_with["baseline_mean"], color="k", ls="--", lw=.8, label="pre-switch baseline")
    ax.set_xlabel("step"); ax.set_ylabel(f"team return (trailing mean, w={args.smooth_window})")
    ax.set_title("Recovery after the switch: reset vs. passive re-adaptation"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig3_recovery.png", dpi=120); plt.close(fig)

    print(f"\n[{time.time()-t0:.0f}s] wrote {out}/summary.txt + fig1..3")


if __name__ == "__main__":
    main()

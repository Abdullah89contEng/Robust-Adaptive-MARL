"""Run the Chapter-11 evaluation over {variant x condition x eps_a x seed x episode}
and write tidy CSV. No plotting, no interpretation -- just numbers.

Usage:
    python scripts/run_experiments.py \
        --phase1-ckpt runs/20260906_162246/phase1_checkpoint_2999.pt \
        --detector    runs/20260906_182413_phase2only/phase2_detector.pt \
        --seeds 5 --episodes 10 --horizon 200 --change-step 60 \
        --eps-list 0.0,0.05,0.1,0.2,0.3 \
        --variants full,detector_off,no_reexplore \
        --conditions nominal,attack,change,simultaneous
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from method.io import load_model
from method.training.meta_test import (
    MetaTestRunner, MetaTestConfig, RegimeChangeEvent, AdversarialAttackEvent,
)
from method.training.regime import Regime
from method import metrics as M
from method.viz import _victim_state
from utils.scenario import Scenario

NOMINAL = Regime(mass=0.9, linear_friction=0.03)
SHIFTED = Regime(mass=1.2, linear_friction=0.15)

# variants that only need config toggles on the one trained checkpoint
CONFIG_VARIANTS = {
    "full":         dict(),
    "detector_off": dict(threshold_C=10.0),          # never resets -> tests "no reset"
    "no_reexplore": dict(re_exploration_budget_K=0),  # reset but execute immediately
}
# variants that need a *separately trained* checkpoint (not run here)
NEEDS_CKPT = {"plain_fmasac", "context_only", "no_adversary", "maal_random",
              "mean_pooling", "gru_belief", "no_covariance", "single_phase",
              "fixed_epsilon", "sample_belief", "q_zero"}


def make_config(variant: str) -> MetaTestConfig:
    if variant in CONFIG_VARIANTS:
        return MetaTestConfig(**CONFIG_VARIANTS[variant])
    raise SystemExit(f"variant '{variant}' needs a checkpoint trained with that component "
                     f"changed; not available. Config-only variants: {list(CONFIG_VARIANTS)}")


def build_events(condition: str, change_step: int, eps_a: float, n_agents: int):
    base = [RegimeChangeEvent(0, i, NOMINAL) for i in range(n_agents)]
    if condition == "nominal":
        return base, None
    if condition == "attack":
        # PGD radii are set on MetaTestConfig; the *event* just switches the attack on.
        return base + [AdversarialAttackEvent(change_step, 0, True, True)], None
    if condition == "change":
        return base + [RegimeChangeEvent(change_step, 0, SHIFTED)], change_step
    if condition == "simultaneous":
        return (base + [RegimeChangeEvent(change_step, 0, SHIFTED),
                        AdversarialAttackEvent(change_step, 0, True, True)], change_step)
    raise SystemExit(f"unknown condition '{condition}'")


def run_episode(trainer, detector, cfg, events, horizon, seed):
    torch.manual_seed(seed)
    runner = MetaTestRunner(trainer, detector,
                            lambda: Scenario(config_file="world_config.yaml"), cfg)
    runner.schedule(events)
    runner.reset_state()
    step_reward, a0_resets = [], []
    for t in range(horizon):
        s = runner.step(render=False)
        step_reward.append(float(np.sum(s.rewards)))
        if s.reset_fired[0]:
            a0_resets.append(t)
        _, _, done = _victim_state(runner.env)
        if done:
            break
    return dict(step_reward=step_reward, a0_resets=a0_resets, steps=len(step_reward),
                total_return=float(np.sum(step_reward)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-ckpt", default=None)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--change-step", type=int, default=60)
    ap.add_argument("--eps-list", default="0.0,0.1,0.2,0.3")
    ap.add_argument("--variants", default="full")
    ap.add_argument("--conditions", default="nominal,attack,change,simultaneous")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    eps_list = [float(x) for x in args.eps_list.split(",")]
    variants = args.variants.split(",")
    conditions = args.conditions.split(",")
    out = Path(args.out_dir) if args.out_dir else Path("runs") / ("experiments_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    trainer, detector = load_model(args.phase1_ckpt, args.detector)
    n_agents = trainer.n_agents
    print(f"n_agents={n_agents} | out={out}")

    rows = []          # per-episode
    det_bucket = {}    # (variant, condition, eps) -> list of detection dicts
    rec_bucket = {}    # (variant, condition, eps) -> list of (recovery, censored)
    ret_bucket = {}    # (variant, condition, eps) -> list of per-episode total_return

    t0 = time.time()
    for variant in variants:
        for condition in conditions:
            eps_here = eps_list if condition in ("attack", "simultaneous") else [0.0]
            for eps_a in eps_here:
                cfg = make_config(variant)
                if condition in ("attack", "simultaneous"):
                    cfg.eps_action = eps_a
                    cfg.eps_pos = eps_a
                key = (variant, condition, eps_a)
                det_bucket[key], rec_bucket[key], ret_bucket[key] = [], [], []
                for seed in range(args.seeds):
                    for ep in range(args.episodes):
                        s = 1000 * seed + ep
                        events, true_change = build_events(condition, args.change_step, eps_a, n_agents)
                        r = run_episode(trainer, detector, cfg, events, args.horizon, s)
                        dm = M.detection_metrics(r["a0_resets"], true_change, r["steps"])
                        rt, cens = (M.recovery_time(r["step_reward"], true_change, r["steps"])
                                    if true_change is not None else (float("nan"), False))
                        det_bucket[key].append(dm)
                        if true_change is not None:
                            rec_bucket[key].append((rt, cens))
                        ret_bucket[key].append(r["total_return"])
                        rows.append(dict(variant=variant, condition=condition, eps_a=eps_a,
                                         seed=seed, episode=ep, steps=r["steps"],
                                         total_return=r["total_return"],
                                         n_resets_a0=len(r["a0_resets"]),
                                         tp=dm["tp"], fp=dm["fp"], fn=dm["fn"], tn=dm["tn"],
                                         delay=dm["delay"],
                                         recovery=rt, recovery_censored=int(cens)))
                print(f"  [{time.time()-t0:6.0f}s] {variant:14} {condition:13} eps={eps_a:.2f}  "
                      f"ret_iqm={M.iqm(ret_bucket[key]):+.3f}")

    # per-episode CSV
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # aggregated summary CSV
    summ = []
    for key in det_bucket:
        variant, condition, eps_a = key
        d = M.aggregate_detection(det_bucket[key])
        row = dict(variant=variant, condition=condition, eps_a=eps_a, **d)
        row.update(M.summarize(ret_bucket[key], "return"))
        if rec_bucket[key]:
            row.update(M.aggregate_recovery(rec_bucket[key]))
        summ.append(row)
    # degradation vs the eps=0 return of the same variant+condition family
    base_ret = {(v, "nominal"): M.iqm(ret_bucket.get((v, "nominal", 0.0), [np.nan]))
                for v in variants}
    for row in summ:
        b = base_ret.get((row["variant"], "nominal"), np.nan)
        row.update(M.degradation(b, row["return_iqm"]))
    keys = sorted({k for r in summ for k in r})
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in summ])

    print(f"\nwrote {out/'results.csv'} ({len(rows)} episodes) and {out/'summary.csv'} "
          f"({len(summ)} groups) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

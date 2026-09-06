"""Meta-test evaluation of a trained model (Algorithm 10.3): loads a frozen
Phase 1 checkpoint + a trained Phase 2 detector, then runs scripted test
episodes and reports detection delay / false-alarm behaviour and return.

Usage:
    python scripts/meta_test_eval.py \
        --phase1-ckpt runs/20260906_162246/phase1_checkpoint_2999.pt \
        --detector    runs/20260906_182413_phase2only/phase2_detector.pt \
        --repeats 5 --horizon 32 --change-step 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.scenario import Scenario
from method.training.phase1 import Phase1Trainer, Phase1Config
from method.detector.indid import CausalLocalWindowTransformer
from method.training.meta_test import (
    MetaTestRunner,
    MetaTestConfig,
    RegimeChangeEvent,
)
from method.training.regime import Regime

CHECKPOINT_KEYS = [
    "flow", "flow_momentum", "budget_encoder", "budget_decoder",
    "exploration_policies", "exploration_critics", "exploration_critics_target",
    "execution_policies", "execution_critics", "execution_critics_target",
    "mixer", "mixer_target",
]


def load_phase1(trainer: Phase1Trainer, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location=trainer.device)
    for key in CHECKPOINT_KEYS:
        if key in state:
            getattr(trainer, key).load_state_dict(state[key])
        elif key.endswith("_target"):
            getattr(trainer, key).load_state_dict(getattr(trainer, key[:-7]).state_dict())
    for name in ("log_alpha_exp", "log_alpha_exe"):
        if name in state:
            with torch.no_grad():
                getattr(trainer, name).copy_(state[name])


def run_episode(runner: MetaTestRunner, horizon: int):
    """Returns per-step arrays: mode, detector_p, reset_fired, reward-sum."""
    runner.reset_state()
    modes, ps, resets, rews = [], [], [], []
    for _ in range(horizon):
        r = runner.step(render=False)
        modes.append(list(r.mode))
        ps.append([(x if x is not None else float("nan")) for x in r.detector_p])
        resets.append(list(r.reset_fired))
        rews.append(float(np.sum(r.rewards)))
    return (np.array(modes), np.array(ps, dtype=float),
            np.array(resets, dtype=bool), np.array(rews, dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-ckpt", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--change-step", type=int, default=16)
    ap.add_argument("--config-file", default="world_config.yaml")
    args = ap.parse_args()

    ckpt = Path(args.phase1_ckpt).resolve()
    prev = json.loads((ckpt.parent / "args.json").read_text())
    print(f"Phase1 ckpt: {ckpt}\nDetector:    {Path(args.detector).resolve()}")

    trainer = Phase1Trainer(
        scenario_factory=lambda: Scenario(config_file=args.config_file),
        config=Phase1Config(n_envs=prev["n_envs"], horizon=prev["horizon"]),
    )
    load_phase1(trainer, ckpt)
    n_agents = trainer.n_agents
    print(f"n_agents={n_agents} code_dim={trainer.code_dim}")

    detector = CausalLocalWindowTransformer(
        code_dim=trainer.code_dim, d_model=64, n_heads=4, n_layers=2,
        window=16, max_len=trainer.cfg.horizon + 1,
    ).to(trainer.device)
    detector.load_state_dict(torch.load(args.detector, map_location=trainer.device))
    detector.eval()

    nominal = Regime(mass=0.9, linear_friction=0.03)
    shifted = Regime(mass=1.2, linear_friction=0.15)
    mtcfg = MetaTestConfig()
    print(f"MetaTestConfig: threshold_C={mtcfg.threshold_C} K={mtcfg.re_exploration_budget_K} "
          f"sigma2_min={mtcfg.sigma2_min}\n")

    def fresh_runner():
        return MetaTestRunner(
            phase1_trainer=trainer, detector=detector,
            scenario_factory=lambda: Scenario(config_file=args.config_file),
            config=mtcfg,
        )

    # ---- Scenario A: no regime change (false-alarm control) ----
    print("=" * 72)
    print("SCENARIO A -- no regime change (agent-0 & agent-1 held at nominal)")
    print("=" * 72)
    fa_counts, returns_a = [], []
    for rep in range(args.repeats):
        torch.manual_seed(1000 + rep)
        runner = fresh_runner()
        runner.schedule([RegimeChangeEvent(0, i, nominal) for i in range(n_agents)])
        _m, ps, resets, rews = run_episode(runner, args.horizon)
        fa = int(resets.any(axis=1).sum())
        fa_counts.append(fa)
        returns_a.append(rews.sum())
        fa_steps = np.where(resets.any(axis=1))[0].tolist()
        print(f"  rep {rep}: return={rews.sum():+8.2f} | false-alarm resets={fa} "
              f"at steps {fa_steps} | mean p (a0,a1)=({np.nanmean(ps[:,0]):.3f},{np.nanmean(ps[:,1]):.3f})")
    print(f"  --> false-alarm resets / episode: mean={np.mean(fa_counts):.2f}  "
          f"(rate {np.mean(fa_counts)/args.horizon:.3f}/step)   return mean={np.mean(returns_a):+.2f}\n")

    # ---- Scenario B: agent-0 regime change at --change-step ----
    print("=" * 72)
    print(f"SCENARIO B -- agent-0 regime change at step {args.change_step} "
          f"(mass 0.9->1.2, friction 0.03->0.15); agent-1 held nominal")
    print("=" * 72)
    delays, misses, pre_fa, returns_b = [], 0, [], []
    for rep in range(args.repeats):
        torch.manual_seed(2000 + rep)
        runner = fresh_runner()
        runner.schedule(
            [RegimeChangeEvent(0, i, nominal) for i in range(n_agents)]
            + [RegimeChangeEvent(args.change_step, 0, shifted)]
        )
        _m, ps, resets, rews = run_episode(runner, args.horizon)
        a0_resets = np.where(resets[:, 0])[0]
        post = a0_resets[a0_resets >= args.change_step]
        pre = a0_resets[a0_resets < args.change_step]
        pre_fa.append(len(pre))
        returns_b.append(rews.sum())
        if len(post):
            d = int(post[0] - args.change_step)
            delays.append(d)
            tag = f"detected at step {int(post[0])} (delay {d})"
        else:
            misses += 1
            tag = "MISSED (no reset after change)"
        p_around = ps[max(0, args.change_step - 2):args.change_step + 8, 0]
        print(f"  rep {rep}: return={rews.sum():+8.2f} | {tag} | "
              f"pre-change false resets(a0)={len(pre)} | "
              f"p[a0] around change={np.array2string(p_around, precision=2, floatmode='fixed')}")
    if delays:
        print(f"  --> detection delay: mean={np.mean(delays):.1f}  median={np.median(delays):.1f}  "
              f"min={min(delays)}  max={max(delays)}  (n={len(delays)})")
    print(f"  --> misses={misses}/{args.repeats}   pre-change false resets/ep mean={np.mean(pre_fa):.2f}   "
          f"return mean={np.mean(returns_b):+.2f}")


if __name__ == "__main__":
    main()

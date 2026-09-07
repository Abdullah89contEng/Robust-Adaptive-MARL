"""Perturbation-inference (identifiability) experiment -- thesis Chapter 11.

Runs deployment episodes with a KNOWN attack budget, reads the budget
encoder q_eta each step, and checks whether eps_hat = p_eta(rho) tracks the
true budget: MAE / RMSE at end of episode, posterior-std, decay of the
posterior variance, and behaviour out of distribution. A constant predictor
at E[p_eps] is the floor.

Usage:
    python scripts/eval_rho_inference.py \
        --phase1-ckpt runs/20260906_162246/phase1_checkpoint_2999.pt \
        --detector    runs/20260906_182413_phase2only/phase2_detector.pt \
        --eps-grid 0.0,0.05,0.1,0.2,0.3,0.45,0.6 --episodes 20 --horizon 120
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
from method.training.meta_test import MetaTestRunner, MetaTestConfig, AdversarialAttackEvent
from method.training.regime import Regime
from method import metrics as M
from utils.scenario import Scenario

NOMINAL = Regime(mass=0.9, linear_friction=0.03)
E_PRIOR = 0.15   # mean of the Phase-1 p_eps = U[0, 0.3]; the constant-predictor floor
TRAIN_MAX = 0.30  # top of the Phase-1 eps_a range; anything above is OOD


@torch.no_grad()
def read_budget(trainer, obs, n_rho_samples=32):
    """obs = list of per-agent obs tensors (each (1, obs_dim)). Returns
    (eps_hat_mean, eps_hat_std) over rho samples from q_eta(x_exe)."""
    joint = torch.stack(obs, dim=1)                      # (1, n_agents, obs_dim)
    zeros = torch.zeros(joint.shape[0], joint.shape[1], trainer.action_dim, device=joint.device)
    x_exe = torch.cat([joint, zeros], dim=-1)            # (1, n_agents, cond_dim) -- the pooled set
    mu, sigma2 = trainer.budget_encoder(x_exe)           # (1, rho_dim)
    eps_mu = float(trainer.budget_decoder(mu))
    rhos = mu + torch.sqrt(sigma2) * torch.randn(n_rho_samples, *mu.shape[1:], device=mu.device)
    eps_samples = trainer.budget_decoder(rhos).cpu().numpy().reshape(-1)
    return eps_mu, float(eps_samples.std()), float(sigma2.mean())


def run_episode(trainer, detector, eps_true, horizon, seed):
    torch.manual_seed(seed)
    cfg = MetaTestConfig(eps_action=eps_true, eps_pos=eps_true)
    runner = MetaTestRunner(trainer, detector,
                            lambda: Scenario(config_file="world_config.yaml"), cfg)
    runner.schedule([AdversarialAttackEvent(0, 0, True, True)])   # attack on from step 0
    runner.reset_state()
    trace = []   # per step: (eps_hat_mu, eps_hat_std, sigma2_rho_mean)
    for t in range(horizon):
        runner.step(render=False)
        trace.append(read_budget(trainer, list(runner.obs)))
    return np.array(trace)   # (T, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-ckpt", default=None)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--eps-grid", default="0.0,0.05,0.1,0.2,0.3,0.45,0.6")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    eps_grid = [float(x) for x in args.eps_grid.split(",")]
    out = Path(args.out_dir) if args.out_dir else Path("runs") / ("rho_inference_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    trainer, detector = load_model(args.phase1_ckpt, args.detector)
    print(f"n_agents={trainer.n_agents} rho_dim={trainer.rho_dim} | out={out}")

    per_ep, per_grid = [], []
    t0 = time.time()
    for eps_true in eps_grid:
        finals, half_sig, quart_sig = [], [], []
        for ep in range(args.episodes):
            tr = run_episode(trainer, detector, eps_true, args.horizon, 7000 + ep)
            eps_final = tr[-1, 0]
            finals.append(eps_final)
            quart_sig.append(tr[len(tr) // 4, 2])
            half_sig.append(tr[len(tr) // 2, 2])
            per_ep.append(dict(eps_true=eps_true, episode=ep,
                               eps_hat_final=float(eps_final),
                               eps_hat_std_final=float(tr[-1, 1]),
                               sigma2_rho_q1=float(tr[len(tr)//4, 2]),
                               sigma2_rho_mid=float(tr[len(tr)//2, 2]),
                               sigma2_rho_final=float(tr[-1, 2])))
        finals = np.array(finals)
        mae = float(np.mean(np.abs(finals - eps_true)))
        rmse = float(np.sqrt(np.mean((finals - eps_true) ** 2)))
        mae_const = float(np.mean(np.abs(E_PRIOR - eps_true)))   # constant-predictor floor
        per_grid.append(dict(
            eps_true=eps_true, ood=int(eps_true > TRAIN_MAX),
            eps_hat_iqm=M.iqm(finals), eps_hat_mean=float(finals.mean()),
            mae=mae, rmse=rmse, mae_constant_predictor=mae_const,
            beats_constant=int(mae < mae_const),
            sigma2_rho_contracts=int(np.mean(half_sig) < np.mean(quart_sig)),
            n=len(finals)))
        print(f"  [{time.time()-t0:5.0f}s] eps={eps_true:.2f}{' (OOD)' if eps_true>TRAIN_MAX else '':6} "
              f"eps_hat_iqm={M.iqm(finals):+.3f}  MAE={mae:.3f}  (const {mae_const:.3f})")

    with open(out / "per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ep[0].keys())); w.writeheader(); w.writerows(per_ep)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_grid[0].keys())); w.writeheader(); w.writerows(per_grid)
    print(f"\nwrote {out/'summary.csv'} and {out/'per_episode.csv'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

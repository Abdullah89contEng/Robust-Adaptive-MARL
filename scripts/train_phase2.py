"""Retrain ONLY the Phase 2 detector on top of an already-trained, frozen
Phase 1 checkpoint. Does not touch Phase 1.

Usage:
    python scripts/train_phase2.py \
        --phase1-ckpt runs/20260906_162246/phase1_checkpoint_2999.pt \
        --phase2-iters 300

Writes a fresh runs/<timestamp>_phase2only/ with phase2_log.csv, the trained
phase2_detector.pt, and a loss plot. The Phase 1 config (n_envs, horizon,
config_file) is read from the checkpoint's sibling args.json so the detector
sees the same episode distribution Phase 1 was trained on.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.scenario import Scenario
from method.training.phase1 import Phase1Trainer, Phase1Config
from method.training.phase2 import Phase2Trainer, Phase2Config

# Same set train.py uses, minus the log_alpha tensors (handled separately).
CHECKPOINT_KEYS = [
    "flow",
    "flow_momentum",
    "budget_encoder",
    "budget_decoder",
    "exploration_policies",
    "exploration_critics",
    "exploration_critics_target",
    "execution_policies",
    "execution_critics",
    "execution_critics_target",
    "mixer",
    "mixer_target",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-ckpt", type=str, required=True)
    p.add_argument("--phase2-iters", type=int, default=300)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--detector-loss", type=str, default="paper", choices=["paper", "indid"],
                   help="Phase-2 CPD objective: 'paper' (arXiv:2510.24988v1 BCE) or 'indid' (InDiD CPDLoss)")
    p.add_argument("--randomize-map", action="store_true",
                   help="re-randomize obstacle/victim/agent positions each episode (boundary fixed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def load_phase1(trainer: Phase1Trainer, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location=trainer.device)
    for key in CHECKPOINT_KEYS:
        if key in state:
            getattr(trainer, key).load_state_dict(state[key])
        elif key.endswith("_target"):
            online_key = key[: -len("_target")]
            getattr(trainer, key).load_state_dict(getattr(trainer, online_key).state_dict())
        else:
            print(f"  WARNING: checkpoint missing '{key}'")
    for name in ("log_alpha_exp", "log_alpha_exe"):
        if name in state:
            with torch.no_grad():
                getattr(trainer, name).copy_(state[name])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ckpt_path = Path(args.phase1_ckpt).resolve()
    prev_args = json.loads((ckpt_path.parent / "args.json").read_text())
    n_envs = prev_args["n_envs"]
    horizon = prev_args["horizon"]
    config_file = prev_args["config_file"]
    print(f"Phase 1 checkpoint: {ckpt_path}")
    print(f"  n_envs={n_envs} horizon={horizon} config_file={config_file}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parents[1] / "runs" / (time.strftime("%Y%m%d_%H%M%S") + "_phase2only")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps({**vars(args), "phase1_ckpt": str(ckpt_path)}, indent=2))
    print(f"Writing outputs to {out_dir}")

    trainer1 = Phase1Trainer(
        scenario_factory=lambda: Scenario(config_file=config_file, randomize_map=args.randomize_map),
        config=Phase1Config(n_envs=n_envs, horizon=horizon),
    )
    load_phase1(trainer1, ckpt_path)
    print(f"n_agents={trainer1.n_agents} obs_dim={trainer1.obs_dim} code_dim={trainer1.code_dim}")

    p2_config = Phase2Config(detector_loss=args.detector_loss)
    print(f"Phase2Config: {p2_config}")
    trainer2 = Phase2Trainer(trainer1, p2_config)

    log_path = out_dir / "phase2_log.csv"
    fields = None
    start = time.time()
    n_nan = 0
    for it in range(args.phase2_iters):
        stats = trainer2.train_iteration()
        if stats["l_cpd"] != stats["l_cpd"]:  # NaN
            n_nan += 1
        row = {"iter": it, "elapsed_s": round(time.time() - start, 1), **stats}
        if fields is None:
            fields = list(row.keys())
            with open(log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        if it % args.log_every == 0 or it == args.phase2_iters - 1:
            print(f"[phase2] iter {it:>5}/{args.phase2_iters} | l_cpd={stats['l_cpd']:.4f} "
                  f"grad_norm={stats['grad_norm']:.3f} | nan_so_far={n_nan}")

    torch.save(trainer2.detector.state_dict(), out_dir / "phase2_detector.pt")
    print(f"Phase 2 done in {time.time() - start:.1f}s | total NaN iters: {n_nan}/{args.phase2_iters}")

    df = pd.read_csv(log_path)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["iter"], df["l_cpd"])
    ax[0].set_title("phase2: l_cpd")
    ax[0].set_xlabel("iter")
    ax[1].plot(df["iter"], df["grad_norm"])
    ax[1].set_title("phase2: grad_norm (pre-clip)")
    ax[1].set_xlabel("iter")
    fig.tight_layout()
    fig.savefig(out_dir / "phase2_summary.png", dpi=120)
    print(f"Saved {out_dir / 'phase2_summary.png'}")
    print("DONE")


if __name__ == "__main__":
    main()

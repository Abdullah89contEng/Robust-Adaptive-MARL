"""Real (non-smoke-test) training run: Phase 1, then Phase 2, on the
search-and-rescue Scenario. Writes CSV logs, periodic checkpoints, and a
final loss-curve plot under runs/<timestamp>/.

Usage:
    python scripts/train.py --phase1-iters 1000 --phase2-iters 300
    python scripts/train.py --resume runs/20260906_162246   # continue an interrupted run
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-iters", type=int, default=1000)
    p.add_argument("--phase2-iters", type=int, default=300)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--config-file", type=str, default="world_config.yaml")
    p.add_argument("--shaping-weight", type=float, default=0.0,
                   help="potential-based reward shaping toward nearest unrescued victim (0 = off)")
    p.add_argument("--detector-loss", type=str, default="paper", choices=["paper", "indid"],
                   help="Phase-2 CPD objective: 'paper' (arXiv:2510.24988v1 BCE) or 'indid' (InDiD CPDLoss)")
    p.add_argument("--randomize-map", action="store_true",
                   help="re-randomize obstacle/victim/agent positions each iteration (boundary fixed)")
    p.add_argument("--device", type=str, default="auto",
                   help="torch device for models AND the vectorized sim: auto (default: cuda if available, else cpu) | cpu | cuda | cuda:N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Run dir to resume from (finds its latest phase1 checkpoint)")
    return p.parse_args()


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


def save_checkpoint(trainer: Phase1Trainer, path: Path) -> None:
    state = {key: getattr(trainer, key).state_dict() for key in CHECKPOINT_KEYS}
    state["log_alpha_exp"] = trainer.log_alpha_exp.detach().clone()
    state["log_alpha_exe"] = trainer.log_alpha_exe.detach().clone()
    torch.save(state, path)


def load_checkpoint(trainer: Phase1Trainer, path: Path) -> None:
    state = torch.load(path, map_location=trainer.device)
    missing = [key for key in CHECKPOINT_KEYS if key not in state]
    for key in CHECKPOINT_KEYS:
        if key in state:
            getattr(trainer, key).load_state_dict(state[key])
    if missing:
        # Older checkpoints (saved before target networks / log_alpha were
        # added to CHECKPOINT_KEYS) won't have these. Target networks fall
        # back to their standard init -- a copy of the just-loaded online
        # network -- rather than the fresh random init they'd otherwise get;
        # log_alpha just keeps Phase1Config's init value.
        print(f"Checkpoint missing keys (using defaults): {missing}")
        for key in missing:
            if key.endswith("_target"):
                online_key = key[: -len("_target")]
                getattr(trainer, key).load_state_dict(getattr(trainer, online_key).state_dict())
    if "log_alpha_exp" in state:
        with torch.no_grad():
            trainer.log_alpha_exp.copy_(state["log_alpha_exp"])
    if "log_alpha_exe" in state:
        with torch.no_grad():
            trainer.log_alpha_exe.copy_(state["log_alpha_exe"])
    # NOTE: optimizer momentum and the replay buffers (D_exp/D_exe) are not
    # restored -- a real, accepted gap in this resume, not an oversight.
    # Optimizers just start with zero momentum again (a brief transient, not
    # incorrect); the buffers start empty, so `update()` is a no-op again
    # until enough rollouts refill them past `batch_size`.


def find_latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    checkpoints = sorted(run_dir.glob("phase1_checkpoint_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No phase1_checkpoint_*.pt found in {run_dir}")
    latest = checkpoints[-1]
    return latest, int(latest.stem.rsplit("_", 1)[-1])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device %s requested but torch.cuda.is_available() is False; "
                         "install a CUDA build of torch on a machine with an NVIDIA GPU" % args.device)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(args.device)
    print("device:", dev)

    start_iter = 0
    if args.resume:
        out_dir = Path(args.resume)
        prev_args = json.loads((out_dir / "args.json").read_text())
        args.n_envs = prev_args["n_envs"]
        args.horizon = prev_args["horizon"]
        args.config_file = prev_args["config_file"]
        args.detector_loss = prev_args.get("detector_loss", "paper")
        args.randomize_map = prev_args.get("randomize_map", False)
        latest_ckpt, last_done_iter = find_latest_checkpoint(out_dir)
        start_iter = last_done_iter + 1
        print(f"Resuming from {latest_ckpt} (continuing at iteration {start_iter})")
    else:
        out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[1] / "runs" / time.strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Writing outputs to {out_dir}")

    p1_config = Phase1Config(n_envs=args.n_envs, horizon=args.horizon)
    trainer1 = Phase1Trainer(
        scenario_factory=lambda: Scenario(config_file=args.config_file, shaping_weight=args.shaping_weight,
                                          randomize_map=args.randomize_map),
        config=p1_config,
        device=dev,
    )
    print(f"n_agents={trainer1.n_agents} obs_dim={trainer1.obs_dim} code_dim={trainer1.code_dim}")

    if args.resume:
        load_checkpoint(trainer1, latest_ckpt)

    phase1_log_path = out_dir / "phase1_log.csv"
    phase1_fields = None
    if args.resume and phase1_log_path.exists():
        with open(phase1_log_path) as f:
            phase1_fields = next(csv.reader(f))

    start = time.time()
    end_iter = start_iter + args.phase1_iters
    for it in range(start_iter, end_iter):
        rollout_stats = trainer1.rollout_iteration()
        update_stats = trainer1.update()
        row = {"iter": it, "elapsed_s": round(time.time() - start, 1), **rollout_stats, **update_stats}

        if phase1_fields is None and update_stats:
            phase1_fields = list(row.keys())
            with open(phase1_log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=phase1_fields).writeheader()

        if phase1_fields and update_stats:
            with open(phase1_log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase1_fields)
                writer.writerow({k: row.get(k, "") for k in phase1_fields})

        if it % args.log_every == 0 or it == end_iter - 1:
            print(f"[phase1] iter {it:>5}/{end_iter} | {row}")

        if it % args.checkpoint_every == 0 or it == end_iter - 1:
            save_checkpoint(trainer1, out_dir / f"phase1_checkpoint_{it}.pt")

    print(f"Phase 1 done in {time.time() - start:.1f}s")

    # --- Phase 2 ---
    p2_config = Phase2Config(detector_loss=args.detector_loss)
    trainer2 = Phase2Trainer(trainer1, p2_config)

    phase2_log_path = out_dir / "phase2_log.csv"
    phase2_write_header = not phase2_log_path.exists()
    phase2_iter_offset = 0
    if not phase2_write_header:
        phase2_iter_offset = len(pd.read_csv(phase2_log_path))

    start2 = time.time()
    for it in range(args.phase2_iters):
        stats = trainer2.train_iteration()
        row = {"iter": phase2_iter_offset + it, "elapsed_s": round(time.time() - start2, 1), **stats}
        if phase2_write_header:
            with open(phase2_log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=list(row.keys())).writeheader()
            phase2_write_header = False
        with open(phase2_log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
        if it % args.log_every == 0 or it == args.phase2_iters - 1:
            print(f"[phase2] iter {row['iter']:>5} | {row}")

    torch.save(trainer2.detector.state_dict(), out_dir / "phase2_detector.pt")
    print(f"Phase 2 done in {time.time() - start2:.1f}s")

    # --- summary plot (reads the full CSVs back from disk, so it covers the
    # whole run's history even after a resume, not just this invocation's
    # in-memory rows) ---
    phase1_df = pd.read_csv(phase1_log_path)
    phase2_df = pd.read_csv(phase2_log_path) if phase2_log_path.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(phase1_df["iter"], phase1_df["episode_return_agent0"])
    axes[0, 0].set_title("phase1: rollout return")
    updated = phase1_df.dropna(subset=["l_elbo"])
    axes[0, 1].plot(updated["iter"], updated["l_elbo"])
    axes[0, 1].set_title("phase1: l_elbo")
    axes[0, 2].plot(updated["iter"], updated["l_cpc"])
    axes[0, 2].set_title("phase1: l_cpc")
    axes[1, 0].plot(updated["iter"], updated["exp_critic_loss"], label="exploration")
    axes[1, 0].plot(updated["iter"], updated["exe_critic_loss"], label="execution")
    axes[1, 0].set_title("phase1: critic losses")
    axes[1, 0].legend()
    if phase2_df is not None:
        axes[1, 1].plot(phase2_df["iter"], phase2_df["l_cpd"])
        axes[1, 1].set_title("phase2: l_cpd")
    else:
        axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=120)
    print(f"Saved summary plot to {out_dir / 'summary.png'}")

    print("DONE")


if __name__ == "__main__":
    main()

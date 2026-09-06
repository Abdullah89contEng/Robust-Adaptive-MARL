"""Record a meta-test (Algorithm 10.3) episode as an annotated video.

Runs one scripted test episode with the frozen trained model and writes an
mp4 (or gif fallback) where each frame is the VMAS render plus an overlay
showing step, per-agent mode / detector p / reset, and any scripted event.

Usage:
    python scripts/record_metatest_video.py \
        --phase1-ckpt runs/20260906_162246/phase1_checkpoint_2999.pt \
        --detector    runs/20260906_182413_phase2only/phase2_detector.pt \
        --scenario change --change-step 16 --out runs/metatest_change.mp4 --fps 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    AdversarialAttackEvent,
)
from method.training.regime import Regime

_CKPT_KEYS = [
    "flow", "flow_momentum", "budget_encoder", "budget_decoder",
    "exploration_policies", "exploration_critics", "exploration_critics_target",
    "execution_policies", "execution_critics", "execution_critics_target",
    "mixer", "mixer_target",
]


def build_model(phase1_ckpt: str, detector_path: str):
    ckpt = Path(phase1_ckpt).resolve()
    prev = json.loads((ckpt.parent / "args.json").read_text())
    trainer = Phase1Trainer(
        scenario_factory=lambda: Scenario(config_file="world_config.yaml"),
        config=Phase1Config(n_envs=prev["n_envs"], horizon=prev["horizon"]),
    )
    st = torch.load(ckpt, map_location=trainer.device)
    for k in _CKPT_KEYS:
        if k in st:
            getattr(trainer, k).load_state_dict(st[k])
        elif k.endswith("_target"):
            getattr(trainer, k).load_state_dict(getattr(trainer, k[:-7]).state_dict())
    for n in ("log_alpha_exp", "log_alpha_exe"):
        if n in st:
            with torch.no_grad():
                getattr(trainer, n).copy_(st[n])

    detector = CausalLocalWindowTransformer(
        code_dim=trainer.code_dim, d_model=64, n_heads=4, n_layers=2,
        window=16, max_len=trainer.cfg.horizon + 1,
    ).to(trainer.device)
    detector.load_state_dict(torch.load(detector_path, map_location=trainer.device))
    detector.eval()
    return trainer, detector


def record(trainer, detector, events, seed, out_path, fps, horizon, title):
    torch.manual_seed(seed)
    runner = MetaTestRunner(
        trainer, detector,
        scenario_factory=lambda: Scenario(config_file="world_config.yaml"),
        config=MetaTestConfig(),
    )
    runner.schedule(events)
    runner.reset_state()

    event_by_step: dict[int, list[str]] = {}
    for e in events:
        if isinstance(e, RegimeChangeEvent):
            event_by_step.setdefault(e.step, []).append(f"regime change: agent {e.agent}")
        else:
            event_by_step.setdefault(e.step, []).append(f"attack ON: agent {e.agent}")

    frames, metas = [], []
    for t in range(horizon):
        res = runner.step(render=True)
        if res.frame is None:
            raise RuntimeError("env.render returned None -- is a display / rgb backend available?")
        frames.append(np.asarray(res.frame))
        metas.append(res)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write(frames, metas, event_by_step, out_path, fps, title)
    print(f"wrote {out_path}  ({len(frames)} frames @ {fps} fps)")

    # per-step console log
    for t, m in enumerate(metas):
        ev = "  <<< " + "; ".join(event_by_step[t]) + " >>>" if t in event_by_step else ""
        pcol = " ".join(f"a{i}:{m.mode[i][:3]}/p={m.detector_p[i]:.2f}"
                        + ("/RESET" if m.reset_fired[i] else "") for i in range(len(m.mode)))
        print(f"  step {t:>2} | {pcol} | r={np.sum(m.rewards):+.3f}{ev}")


def _write(frames, metas, event_by_step, out_path, fps, title):
    h, w = frames[0].shape[:2]
    txt_w = 4.6
    fig = plt.figure(figsize=(w / 100 + txt_w, max(h / 100, 3.2)))
    gs = fig.add_gridspec(1, 2, width_ratios=[w / 100, txt_w], wspace=0.02)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_img.axis("off")
    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    def draw(i):
        ax_img.clear(); ax_img.axis("off")
        ax_txt.clear(); ax_txt.axis("off")
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
        ax_img.imshow(frames[i])
        m = metas[i]
        lines = [title, "", f"step {i:>2} / {len(frames) - 1}"]
        if i in event_by_step:
            lines += [f">>> {s} <<<" for s in event_by_step[i]]
        lines.append("")
        for a in range(len(m.mode)):
            tag = "  <-- RESET" if m.reset_fired[a] else ""
            atk = "  [ATTACKED]" if m.active_attacks[a] else ""
            lines.append(f"agent {a}: {m.mode[a].upper():7s} p={m.detector_p[a]:.3f}{tag}{atk}")
        lines.append("")
        lines.append(f"step reward: {np.sum(m.rewards):+.3f}")
        ax_txt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=10, transform=ax_txt.transAxes)

    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        with writer.saving(fig, str(out_path), dpi=100):
            for i in range(len(frames)):
                draw(i)
                writer.grab_frame()
        return
    except Exception as e:
        print(f"  FFMpegWriter failed ({e!r}); falling back to GIF via Pillow")

    from PIL import Image
    gif_path = out_path.with_suffix(".gif")
    pil_frames = []
    for i in range(len(frames)):
        draw(i)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        pil_frames.append(Image.fromarray(buf))
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=int(1000 / fps), loop=0)
    print(f"  wrote {gif_path} instead")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-ckpt", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--scenario", choices=["control", "change", "attack"], default="change")
    p.add_argument("--change-step", type=int, default=16)
    p.add_argument("--attack-step", type=int, default=16)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out", default="runs/metatest.mp4")
    return p.parse_args()


def main():
    a = parse_args()
    trainer, detector = build_model(a.phase1_ckpt, a.detector)
    n = trainer.n_agents
    nominal = Regime(mass=0.9, linear_friction=0.03)
    shifted = Regime(mass=1.2, linear_friction=0.15)

    base = [RegimeChangeEvent(0, i, nominal) for i in range(n)]
    if a.scenario == "control":
        events, title = base, "meta-test: no change"
    elif a.scenario == "change":
        events = base + [RegimeChangeEvent(a.change_step, 0, shifted)]
        title = f"meta-test: agent-0 regime change @ {a.change_step}"
    else:
        events = base + [AdversarialAttackEvent(a.attack_step, 0, attack_position=True, attack_action=True)]
        title = f"meta-test: adversarial attack on agent 0 @ {a.attack_step}"

    record(trainer, detector, events, a.seed, a.out, a.fps, a.horizon, title)


if __name__ == "__main__":
    main()

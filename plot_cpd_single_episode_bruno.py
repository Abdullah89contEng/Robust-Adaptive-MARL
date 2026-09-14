import sys
sys.path.insert(0, "/home/abodeh/thesis")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
from method.io import load_model
from method.training.meta_test import MetaTestRunner, MetaTestConfig, RegimeChangeEvent
from method.training.regime import Regime
from utils.scenario import Scenario

NOMINAL = Regime(mass=0.9, linear_friction=0.03)
SHIFTED = Regime(mass=1.2, linear_friction=0.15)

trainer, detector = load_model(
    "runs/uni_bruno_phase1/phase1_checkpoint_19500.pt",
    "runs/uni_bruno_phase2/phase2_detector.pt",
)
n = trainer.n_agents
change_step = 60
horizon = 200
cfg = MetaTestConfig()
threshold_C = cfg.threshold_C
seed = 0

torch.manual_seed(1000 + seed)
runner = MetaTestRunner(trainer, detector, lambda: Scenario(config_file="map.yaml"), cfg)
events = [RegimeChangeEvent(0, i, NOMINAL) for i in range(n)] + [RegimeChangeEvent(change_step, 0, SHIFTED)]
runner.schedule(events)
runner.reset_state()

ps, resets, modes = [], [], []
for t in range(horizon):
    r = runner.step(render=False)
    ps.append(r.detector_p[0])
    resets.append(r.reset_fired[0])
    modes.append(r.mode[0])

ps = np.array(ps)
resets = np.array(resets)
t_axis = np.arange(horizon)
detect_steps = np.where(resets)[0]

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(t_axis, ps, lw=1.4, color="#1f5aa6", label=r"detector probability $p_t$ (agent 0)")
ax.axhline(threshold_C, color="#b02418", lw=1.3, ls="--", label=f"firing threshold ($C={threshold_C}$)")
ax.axvline(change_step, color="#555555", lw=1.3, ls=":", label=f"true regime switch ($t={change_step}$)")

if len(detect_steps) > 0:
    for k, ts in enumerate(detect_steps):
        ax.axvline(ts, color="#2a9d5c", lw=1.6, ls="-",
                    label="detected switch (reset fired)" if k == 0 else None)
        ax.plot(ts, ps[ts], marker="*", color="#2a9d5c", markersize=14, zorder=5)
    title_note = f"{len(detect_steps)} reset(s) fired"
else:
    ax.annotate(
        "no reset ever fires:\n$p_t$ never reaches $C$",
        xy=(change_step, ps[change_step]),
        xytext=(change_step + 55, threshold_C - 0.15),
        fontsize=9, color="#b02418",
        arrowprops=dict(arrowstyle="->", color="#b02418", lw=1.0),
    )
    title_note = "no detection in this episode"

ax.set_xlabel("environment step $t$")
ax.set_ylabel(r"detector probability $p_t$")
ax.set_ylim(0, max(1.0, threshold_C + 0.2))
ax.set_xlim(0, horizon - 1)
ax.set_title(f"Change-point detection on a single episode ({title_note})")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
out = "/home/abodeh/thesis/bruno-cpd-single-episode.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

print()
print("seed =", seed, " change_step =", change_step, " threshold_C =", threshold_C,
      " trigger_persistence =", cfg.trigger_persistence)
print("n reset_fired steps:", len(detect_steps), " at:", detect_steps.tolist())
print("p_t min/mean/max:", ps.min(), ps.mean(), ps.max())
print("mode sequence changes:", [(i, m) for i, m in enumerate(modes) if i == 0 or modes[i] != modes[i-1]])

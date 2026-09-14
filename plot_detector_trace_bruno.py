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

traces = []
n_seeds = 8
for seed in range(n_seeds):
    torch.manual_seed(1000 + seed)
    runner = MetaTestRunner(trainer, detector, lambda: Scenario(config_file="map.yaml"), cfg)
    events = [RegimeChangeEvent(0, i, NOMINAL) for i in range(n)] + [RegimeChangeEvent(change_step, 0, SHIFTED)]
    runner.schedule(events)
    runner.reset_state()
    ps = []
    for t in range(horizon):
        r = runner.step(render=False)
        ps.append(r.detector_p[0])
    traces.append(np.array(ps))

traces = np.array(traces)  # (n_seeds, horizon)
np.save("/home/abodeh/thesis/bruno_detector_traces.npy", traces)

t_axis = np.arange(horizon)
mean_trace = traces.mean(axis=0)

fig, ax = plt.subplots(figsize=(7, 4))
for i in range(n_seeds):
    ax.plot(t_axis, traces[i], lw=0.6, color="#1f5aa6", alpha=0.35)
ax.plot(t_axis, mean_trace, lw=1.8, color="#1f5aa6", label=f"mean over {n_seeds} episodes")
ax.axhline(threshold_C, color="#b02418", lw=1.3, ls="--", label=f"firing threshold ($C={threshold_C}$)")
ax.axvline(change_step, color="#555555", lw=1.1, ls=":", label=f"regime switch ($t={change_step}$)")
ax.set_xlabel("environment step $t$")
ax.set_ylabel(r"detector probability $p_t$")
ax.set_ylim(0, max(1.0, threshold_C + 0.15))
ax.set_title("Raw change-point score across a full episode")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
out = "/home/abodeh/thesis/bruno-detector-trace.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

print()
print("threshold_C =", threshold_C, " trigger_persistence =", cfg.trigger_persistence)
print("change_step =", change_step, " n_seeds =", n_seeds)
print("overall min/mean/max across all seeds & steps:", traces.min(), traces.mean(), traces.max())
print("max p pre-change  [0,60):   ", traces[:, :60].max())
print("max p post-change [60,200):", traces[:, 60:200].max())
print("mean p pre-change  [0,60):   ", traces[:, :60].mean())
print("mean p post-change [60,200):", traces[:, 60:200].mean())

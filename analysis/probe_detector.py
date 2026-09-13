import sys
sys.path.insert(0, "/home/abodeh/thesis")
import torch
import numpy as np
from method.io import load_model
from method.training.meta_test import MetaTestRunner, MetaTestConfig, RegimeChangeEvent
from method.training.regime import Regime
from utils.scenario import Scenario

NOMINAL = Regime(mass=0.9, linear_friction=0.03)
SHIFTED = Regime(mass=1.2, linear_friction=0.15)

trainer, detector = load_model(
    "runs/uni_20260912_170324/phase1_checkpoint_5500.pt",
    "runs/20260913_175126_phase2only/phase2_detector.pt",
)
n = trainer.n_agents
change_step = 60
horizon = 200

traces = []
for seed in range(5):
    torch.manual_seed(1000 + seed)
    runner = MetaTestRunner(trainer, detector, lambda: Scenario(config_file="map.yaml"), MetaTestConfig())
    events = [RegimeChangeEvent(0, i, NOMINAL) for i in range(n)] + [RegimeChangeEvent(change_step, 0, SHIFTED)]
    runner.schedule(events)
    runner.reset_state()
    ps = []
    for t in range(horizon):
        r = runner.step(render=False)
        ps.append(r.detector_p[0])
    traces.append(np.array(ps))

traces = np.array(traces)  # (5, horizon)
print("threshold_C =", MetaTestConfig().threshold_C, " trigger_persistence =", MetaTestConfig().trigger_persistence)
print("change_step =", change_step)
print()
print("per-seed max p over WHOLE episode:", traces.max(axis=1))
print("per-seed max p in [0,60) (pre-change):", traces[:, :60].max(axis=1))
print("per-seed max p in [60,110) (first 50 post-change):", traces[:, 60:110].max(axis=1))
print("per-seed max p in [110,200) (rest of episode):", traces[:, 110:200].max(axis=1))
print()
print("overall min/mean/max across all seeds & steps:", traces.min(), traces.mean(), traces.max())
print()
print("seed 0 trace, every 10 steps:")
for t in range(0, horizon, 10):
    marker = " <-- CHANGE" if t == change_step else ""
    print(f"  t={t:3d}  p={traces[0, t]:.4f}{marker}")

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from utils.scenario import Scenario
from method.training.phase1 import Phase1Trainer, Phase1Config

t = Phase1Trainer(lambda: Scenario(config_file="world_config_5v.yaml", shaping_weight=0.3),
                  Phase1Config(n_envs=8, horizon=48))
t0 = time.time(); L = []
for it in range(260):
    r = t.rollout_iteration(); u = t.update()
    if u:
        L.append(u)
    if it % 20 == 0 or it == 259:
        w = L[-1] if L else {}
        print("[%4.0fs] it %3d  l_elbo %8.1f  l_cpc %7.3f  ret %+6.2f  alpha_exp %.3f"
              % (time.time()-t0, it, w.get("l_elbo", float("nan")), w.get("l_cpc", float("nan")),
                 r["episode_return_agent0"], w.get("alpha_exp_mean", 0.0)), flush=True)
print("random-chance floor log(K+1) = %.3f" % np.log(17))

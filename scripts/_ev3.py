"""v3 eval: (A) code-shift probe AUC, (B) CPD delay + accuracy. Robust to
early episode termination (policy now finishes episodes)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, numpy as np, torch
from method.io import load_model
from method.training.meta_test import MetaTestRunner, MetaTestConfig, RegimeChangeEvent
from method.training.regime import Regime, apply_regime
from utils.scenario import Scenario

ICY, NORMAL, HEAVY = Regime(0.80, 0.02), Regime(1.10, 0.16), Regime(1.52, 0.36)
ap = argparse.ArgumentParser()
ap.add_argument("--phase1-ckpt", required=True)
ap.add_argument("--detector", required=True)
ap.add_argument("--config-file", default="world_config_5v.yaml")
ap.add_argument("--horizon", type=int, default=200)
ap.add_argument("--change-step", type=int, default=60)
ap.add_argument("--episodes", type=int, default=24)
args = ap.parse_args()
trainer, detector = load_model(args.phase1_ckpt, args.detector)
flow = trainer.flow.eval(); env = trainer.env; n = trainer.n_agents
H, tc, W = args.horizon, args.change_step, 10
factory = lambda: Scenario(config_file=args.config_file)


def probe_auc(pre, post):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = np.concatenate([pre, post]); y = np.r_[np.zeros(len(pre)), np.ones(len(post))]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=800), X, y, cv=5, scoring="roc_auc")))


@torch.no_grad()
def codeshift(m1, m2, seed, T):
    torch.manual_seed(seed)
    for ag in env.agents:
        apply_regime(ag, m1)
    obs = env.reset()
    C = []
    for t in range(1, T + 1):
        if t == tc:
            for ag in env.agents:
                apply_regime(ag, m2)
        acts = [trainer.exploration_policies[i].act(obs[i])[0] for i in range(n)]
        nobs, rew, dones, _ = env.step(acts)
        cond = torch.cat([torch.stack(obs, 1), torch.stack(acts, 1)], -1)
        y = torch.cat([torch.stack(rew, 1).unsqueeze(-1), torch.stack(nobs, 1)], -1)
        c, _ = flow(y, cond); C.append(c[:, 0])   # agent 0: (n_envs, D)
        obs = nobs
    return torch.stack(C, 1)   # (n_envs, T, D)


print("=== (A) code-shift pre vs post (agent 0, probe AUC; 0.5 = no signal) ===")
T = 2 * tc
mar = 6
for name, (m1, m2) in {"normal->heavy": (NORMAL, HEAVY), "icy->heavy": (ICY, HEAVY), "normal->icy": (NORMAL, ICY)}.items():
    C = torch.cat([codeshift(m1, m2, 200 + e, T) for e in range(args.episodes)], 0).numpy()
    pre = C[:, mar:tc - mar].reshape(-1, C.shape[-1])
    post = C[:, tc + mar:T - mar].reshape(-1, C.shape[-1])
    print(f"  {name:14s}  AUC = {probe_auc(pre, post):.3f}")


def ptrace(seed, change):
    torch.manual_seed(seed)
    ev = [RegimeChangeEvent(0, i, NORMAL) for i in range(n)]
    if change:
        ev.append(RegimeChangeEvent(tc, 0, HEAVY))
    r = MetaTestRunner(trainer, detector, factory, MetaTestConfig(threshold_C=1.01))
    r.schedule(ev); r.reset_state()
    return np.array([float(r.step(render=False).detector_p[0]) for _ in range(H)])


chg = [ptrace(1000 + i, True) for i in range(args.episodes)]
noc = [ptrace(5000 + i, False) for i in range(args.episodes)]
ct, nt = np.array(chg), np.array(noc)
print(f"\n=== (B) CPD delay + accuracy (NORMAL->HEAVY @ {tc}) ===")
print(f"detector_p: change pre {ct[:,:tc].mean():.3f} post {ct[:,tc:].mean():.3f} | no-change {nt.mean():.3f} (max {nt.max():.3f})")
print(f"{'thr':>5}{'persist':>8}{'accuracy':>9}{'TPR':>6}{'TNR':>6}{'delay_med':>10}{'delay_mean':>11}{'FA/ep':>7}")
for thr in [0.3, 0.5, 0.7]:
    for persist in [1, 3]:
        TP = FN = 0; delays = []; fp_pre = 0
        for tr in chg:
            trig = [k for k in range(len(tr)) if k + 1 >= persist and np.all(tr[k + 1 - persist:k + 1] > thr)]
            post = [f for f in trig if tc <= f <= tc + W]
            fp_pre += sum(1 for f in trig if f < tc)
            if post:
                TP += 1; delays.append(post[0] - tc)
            else:
                FN += 1
        tn = sum(1 for tr in noc if not [k for k in range(len(tr)) if k + 1 >= persist and np.all(tr[k + 1 - persist:k + 1] > thr)])
        fp_ep = len(noc) - tn
        acc = (TP + tn) / (TP + FN + tn + fp_ep)
        dmed = f"{np.median(delays):.1f}" if delays else "-"
        dmean = f"{np.mean(delays):.2f}" if delays else "-"
        print(f"{thr:>5.2f}{persist:>8d}{acc:>9.2f}{TP/(TP+FN):>6.2f}{tn/len(noc):>6.2f}{dmed:>10}{dmean:>11}{fp_pre/len(chg):>7.2f}")

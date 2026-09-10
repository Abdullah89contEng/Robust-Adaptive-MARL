"""Fast full run: Phase 1 (short) + Phase 2 (dataset + minibatch),
target wall-clock < 30 min. Writes checkpoints and result figures to
runs/fast_<ts>/.  Phase 2 detector = causal local-window Transformer;
objective defaults to the paper CPD loss (arXiv:2510.24988v1 Eq. 7:
near-boundary-weighted, label-smoothed BCE); CPDLOSS=indid switches to
InDiD CPDLoss.  Metrics: calculate_errors (third-party/InDiD)."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_E = lambda k, d: type(d)(os.environ.get(k, d))

from utils.scenario import Scenario
from method.io import load_phase1, resolve_device
from method.training.phase1 import Phase1Trainer, Phase1Config
from method.training.phase2 import Phase2Trainer, Phase2Config
from method.detector.indid import (detection_loss, labels_from_switch_time,
                                   paper_cpd_loss, boundary_labels_from_switch_time,
                                   find_first_change, calculate_errors, f1_score)
from method.training.meta_test import MetaTestRunner, MetaTestConfig, RegimeChangeEvent
from method.training.regime import Regime, apply_regime
from method.viz import _victim_state

CFG          = os.environ.get("CFG", "world_config_5v.yaml")  # map config (yaml) to train on
RANDMAP      = _E("RANDMAP", 1)   # 1 = new obstacle/victim/agent layout each iteration (boundary fixed)
HORIZON      = _E("FH", 48)
P1_ITERS     = _E("P1", 850)
P1_CKPT      = os.environ.get("P1CKPT", "")   # if set, skip Phase-1 training
SHAPING      = 0.3
COLLISION    = _E("COLLISION", 0.0)   # per-agent per-step overlap penalty (0 = off)
DS_EPISODES  = _E("DS", 90)    # Phase-2 labeled-episode dataset
P2_EPOCHS    = _E("EP", 90)
P2_BATCH     = 64
LEN_SEGMENT  = _E("T", 32)
CPD_LOSS     = os.environ.get("CPDLOSS", "paper")  # "paper" (arXiv:2510.24988v1) or "indid"
BCE_W        = _E("BCEW", 0.0)  # indid path only: >0 = InDiD "combined" (BCE + InDiD)
EVAL_ROLLOUTS = _E("EV", 4)    # x8 envs = 32 change + 32 no-change traces
EVAL_H       = 140
CHANGE_STEP  = 40
NORMAL, HEAVY = Regime(1.10, 0.16), Regime(1.52, 0.36)
DEVICE       = os.environ.get("DEVICE", "auto")  # auto | cpu | cuda | cuda:N
_dev = resolve_device(DEVICE)

CKPT_KEYS = ["flow", "flow_momentum", "budget_encoder", "budget_decoder",
             "exploration_policies", "exploration_critics", "exploration_critics_target",
             "execution_policies", "execution_critics", "execution_critics_target",
             "mixer", "mixer_target"]

out = Path(__file__).resolve().parents[1] / "runs" / ("fast_" + time.strftime("%Y%m%d_%H%M%S"))
out.mkdir(parents=True, exist_ok=True)
print("output dir:", out)
t0 = time.time()

# =====================================================================
# PHASE 1
# =====================================================================
if P1_CKPT:
    print(f"phase1: SKIPPED, loading {P1_CKPT}")
    t1 = load_phase1(P1_CKPT, scenario_factory=lambda: Scenario(config_file=CFG, shaping_weight=SHAPING,
                     randomize_map=bool(RANDMAP), collision_penalty=COLLISION),
                     config=Phase1Config(n_envs=8, horizon=HORIZON), device=_dev)
    log = []
else:
  p1cfg = Phase1Config(n_envs=8, horizon=HORIZON)
  t1 = Phase1Trainer(lambda: Scenario(config_file=CFG, shaping_weight=SHAPING,
                                      randomize_map=bool(RANDMAP), collision_penalty=COLLISION),
                     p1cfg, device=_dev)
  print(f"phase1: n_agents={t1.n_agents} obs_dim={t1.obs_dim}  {P1_ITERS} iters, horizon {HORIZON}")
  def _save_p1(path):
      st = {k: getattr(t1, k).state_dict() for k in CKPT_KEYS}
      st["log_alpha_exp"] = t1.log_alpha_exp.detach().clone()
      st["log_alpha_exe"] = t1.log_alpha_exe.detach().clone()
      torch.save(st, path)

  CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 2000))
  log = []
  for it in range(P1_ITERS):
    r = t1.rollout_iteration()
    u = t1.update()
    if u:
        log.append({"it": it, **r, **u})
    if it % 100 == 0 or it == P1_ITERS - 1:
        last = log[-1] if log else {}
        print(f"  [{time.time()-t0:5.0f}s] it {it:4d}  ret {r['episode_return_agent0']:+.2f}  "
              f"l_elbo {last.get('l_elbo', float('nan')):.1f}  l_cpc {last.get('l_cpc', float('nan')):.2f}", flush=True)
    if it > 0 and it % CKPT_EVERY == 0:
        _save_p1(out / f"phase1_checkpoint_{it}.pt")
  _save_p1(out / "phase1_checkpoint.pt")

def sm(a, k=31):
    a = np.asarray(a, float)
    k = max(1, min(k, len(a) // 3) | 1)
    if k <= 1 or len(a) < 3:
        return a
    return np.convolve(a, np.ones(k) / k, "same")

if log:
    L = {k: np.array([d.get(k, np.nan) for d in log]) for k in log[0]}
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0,0].plot(L["it"], L["episode_return_agent0"], lw=.4, alpha=.4); ax[0,0].plot(L["it"], sm(L["episode_return_agent0"]), lw=2, label="agent0")
    if "episode_return_team" in L: ax[0,0].plot(L["it"], sm(L["episode_return_team"]), lw=2, color="k", label="team")
    ax[0,0].set_title("Phase 1: rollout return"); ax[0,0].axhline(0, color="k", lw=.5); ax[0,0].legend(fontsize=7)
    ax[0,1].plot(L["it"], sm(L["l_elbo"])); ax[0,1].set_title("Phase 1: L_elbo"); ax[0,1].set_yscale("log")
    ax[1,0].plot(L["it"], sm(L["l_cpc"])); ax[1,0].set_title("Phase 1: L_cpc"); ax[1,0].set_yscale("log")
    ax[1,1].plot(L["it"], sm(L["exp_critic_loss"]), label="exploration"); ax[1,1].plot(L["it"], sm(L["exe_critic_loss"]), label="execution")
    ax[1,1].set_title("Phase 1: critic losses"); ax[1,1].set_yscale("log"); ax[1,1].legend()
    for a in ax.flat: a.set_xlabel("iteration")
    fig.suptitle(f"Phase 1 training  ({P1_ITERS} iters, horizon {HORIZON})"); fig.tight_layout()
    fig.savefig(out / "fig1_phase1_curves.png", dpi=120); plt.close(fig)
print(f"[{time.time()-t0:.0f}s] phase 1 ready ({'loaded' if P1_CKPT else 'trained'})")

# =====================================================================
# PHASE 2  (InDiD: dataset + shuffled-minibatch CPDLoss training)
# =====================================================================
p2cfg = Phase2Config(len_segment=LEN_SEGMENT, window=min(Phase2Config.window, HORIZON),
                     detector_loss=CPD_LOSS)
t2 = Phase2Trainer(t1, p2cfg)
print(f"phase2: detector={type(t2.detector).__name__}  input dim {t2.detector.input_proj.in_features}  "
      f"window {t2.detector.window}  building dataset ({DS_EPISODES} episodes)...")
Xs, SWs = [], []
for e in range(DS_EPISODES):
    fs, st = t2.collect_labeled_episode()
    ne, na, T, D = fs.shape
    Xs.append(fs.reshape(ne*na, T, D)); SWs.append(st.unsqueeze(0).expand(ne, na).reshape(ne*na))
X = torch.cat(Xs).to(t2.device); SW = torch.cat(SWs).to(t2.device)
t2.detector.update_norm(X)
n_change = int((SW < X.shape[1]).sum())
print(f"  dataset: {len(X)} sequences ({n_change} with a switch), T={X.shape[1]}")

opt = t2.optimizer
ep_loss = []
for ep in range(P2_EPOCHS):
    perm = torch.randperm(len(X))
    losses = []
    for b in range(0, len(X), P2_BATCH):
        idx = perm[b:b+P2_BATCH]
        xb = X[idx]
        if CPD_LOSS == "paper":
            if np.random.rand() < p2cfg.input_noise_prob:          # Sec. 5.2 token noise
                xb = xb + p2cfg.input_noise_std * torch.randn_like(xb)
            p = t2.detector(xb)
            loss = paper_cpd_loss(p, SW[idx], half_width=p2cfg.label_half_width,
                                  smooth_eps=p2cfg.label_smooth_eps,
                                  near_alpha=p2cfg.near_boundary_alpha)
            gclip = p2cfg.grad_clip_norm
        else:
            p = t2.detector(xb)
            loss = detection_loss(p, SW[idx], LEN_SEGMENT)
            if BCE_W > 0:   # InDiD "combined": BCE + InDiD
                lbl = labels_from_switch_time(SW[idx].long(), p.shape[1]).float()
                loss = loss + BCE_W * torch.nn.functional.binary_cross_entropy(p.clamp(1e-4, 1 - 1e-4), lbl)
            gclip = float("inf")
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(t2.detector.parameters(), gclip)
        if torch.isfinite(gn) and torch.isfinite(loss):
            opt.step(); losses.append(loss.item())
    ep_loss.append(np.mean(losses) if losses else np.nan)
    if ep % 10 == 0 or ep == P2_EPOCHS - 1:
        with torch.no_grad():
            pp = t2.detector(X[:256])
            m = (SW[:256] < X.shape[1])
            pre = float(pp[m][:, :5].mean()) if m.any() else float("nan")
            if m.any():
                yb = boundary_labels_from_switch_time(SW[:256][m].long(), X.shape[1], p2cfg.label_half_width)
                at_b = float((pp[m] * yb).sum() / yb.sum().clamp(min=1))
            else:
                at_b = float("nan")
        print(f"  [{time.time()-t0:5.0f}s] epoch {ep:3d}  loss {ep_loss[-1]:8.3f}  "
              f"p@start {pre:.3f}  p@boundary {at_b:.3f}")
torch.save(t2.detector.state_dict(), out / "phase2_detector.pt")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ep_loss)
ax.set_title(f"Phase 2: {'paper CPD (arXiv:2510.24988v1)' if CPD_LOSS == 'paper' else 'InDiD CPDLoss'} per epoch")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
fig.tight_layout(); fig.savefig(out / "fig2_phase2_loss.png", dpi=120); plt.close(fig)
print(f"[{time.time()-t0:.0f}s] phase 2 done, detector + fig2 saved")

# =====================================================================
# EVAL  (InDiD calculate_errors: any flag >= true change = TP)
# =====================================================================
detector = t2.detector.eval()
factory = lambda: Scenario(config_file=CFG, randomize_map=bool(RANDMAP), collision_penalty=COLLISION)
NA = t1.n_agents
env = t1.env
W = detector.window
FEAT = detector.input_proj.in_features

@torch.no_grad()
def batched_ptraces(seed, change):
    """One rollout of t1.env (8 envs) with the frozen exploration policy;
    returns (8, EVAL_H) detector-p traces for agent 0."""
    torch.manual_seed(seed)
    for ag in env.agents:
        apply_regime(ag, NORMAL)
    obs = env.reset()
    hist = []
    traces = []
    for step in range(1, EVAL_H + 1):
        if change and step == CHANGE_STEP:
            apply_regime(env.agents[0], HEAVY)
        acts = [t1.exploration_policies[i].act(obs[i])[0] for i in range(NA)]
        nobs, rew, dn, _ = env.step(acts)
        os_, as_ = torch.stack(obs, 1), torch.stack(acts, 1)
        nos_, rs_ = torch.stack(nobs, 1), torch.stack(rew, 1).unsqueeze(-1)
        feat0 = torch.cat([os_, as_, rs_, nos_], -1)[:, 0]        # (8, FEAT)
        hist.append(feat0)
        if len(hist) > W:
            hist = hist[-W:]
        seq = torch.stack(hist, dim=1)                            # (8, <=W, FEAT)
        traces.append(detector(seq)[:, -1])                       # (8,)
        obs = nobs
    return torch.stack(traces, dim=1).cpu().numpy()               # (8, EVAL_H)

chg = np.concatenate([batched_ptraces(1000 + i, True) for i in range(EVAL_ROLLOUTS)], 0)
noc = np.concatenate([batched_ptraces(5000 + i, False) for i in range(EVAL_ROLLOUTS)], 0)
real = torch.tensor([CHANGE_STEP]*len(chg) + [-1]*len(noc))
P = torch.tensor(np.concatenate([chg, noc], 0)); seq_len = P.shape[1]
print(f"[{time.time()-t0:.0f}s] eval: p change pre {chg[:,:CHANGE_STEP].mean():.3f} post {chg[:,CHANGE_STEP:].mean():.3f}"
      f"  no-change {noc.mean():.3f}")

ths = np.round(np.arange(0.1, 0.96, 0.05), 2)
rows = []
for thr in ths:
    pred = find_first_change(P > thr)
    TN, FP, FN, TP, fpd, dl = calculate_errors(real, pred, seq_len)
    tpd = dl[(real <= pred) & (real != -1)].float()
    rows.append(dict(thr=thr, f1=f1_score(TN, FP, FN, TP),
                     tpr=TP/(TP+FN) if TP+FN else np.nan,
                     fpr=FP/(FP+TN) if FP+TN else np.nan,
                     delay=float(tpd.mean()) if tpd.numel() else np.nan))
best = max(rows, key=lambda r: (r["f1"] if not np.isnan(r["f1"]) else -1))
print(f"  BEST-F1 thr {best['thr']:.2f}: F1={best['f1']:.3f} TPR={best['tpr']:.2f} FPR={best['fpr']:.2f} "
      f"mean detection delay {best['delay']:.1f} steps")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot([r["thr"] for r in rows], [r["f1"] for r in rows], "o-", label="F1")
ax[0].plot([r["thr"] for r in rows], [r["tpr"] for r in rows], "s-", label="TPR (detection rate)")
ax[0].plot([r["thr"] for r in rows], [r["fpr"] for r in rows], "^-", label="FPR (false alarm)")
ax[0].axvline(best["thr"], color="k", ls="--", lw=1); ax[0].set_xlabel("threshold"); ax[0].set_title("Detection vs threshold"); ax[0].legend()
ax[1].plot([r["thr"] for r in rows], [r["delay"] for r in rows], "d-", color="C3")
ax[1].axvline(best["thr"], color="k", ls="--", lw=1); ax[1].set_xlabel("threshold"); ax[1].set_ylabel("mean detection delay (steps)")
ax[1].set_title("Detection delay vs threshold (delayed detection still counts)")
fig.suptitle(f"Detection — NORMAL->HEAVY @ t={CHANGE_STEP}"); fig.tight_layout()
fig.savefig(out / "fig3_detection_metrics.png", dpi=120); plt.close(fig)

thr = best["thr"]
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
for ax, p in zip(axes, chg[:3]):
    pi = int(find_first_change((torch.tensor(p) > thr).unsqueeze(0))[0])
    ax.plot(p, color="C0", lw=1.5, label="detector $p_t$")
    ax.axhline(thr, ls="--", color="k", lw=.7)
    ax.axvline(CHANGE_STEP, color="C3", lw=2.2, label=f"true change t={CHANGE_STEP}")
    if pi >= CHANGE_STEP:
        ax.axvline(pi, color="C2", ls=":", lw=2.2, label=f"detected t={pi}  (delay {pi-CHANGE_STEP})")
        ax.annotate("", xy=(pi, .12), xytext=(CHANGE_STEP, .12), arrowprops=dict(arrowstyle="<->", color="C2"))
    elif pi >= 0:
        ax.axvline(pi, color="C1", ls=":", lw=2, label=f"false alarm t={pi}")
    else:
        ax.text(CHANGE_STEP+10, .5, "missed", color="C3")
    ax.set_ylim(-.05, 1.05); ax.set_ylabel("$p_t$")
axes[0].legend(fontsize=8, loc="upper left"); axes[-1].set_xlabel("step")
fig.suptitle(f"Change point vs detection time  (thr={thr}; any flag after the change = true detection)")
fig.tight_layout(); fig.savefig(out / "fig4_changepoint_detection.png", dpi=120); plt.close(fig)

# one rescue episode
torch.manual_seed(2024)
r = MetaTestRunner(t1, detector, factory, MetaTestConfig(threshold_C=best["thr"], trigger_persistence=3))
r.schedule([RegimeChangeEvent(0,i,NORMAL) for i in range(NA)] + [RegimeChangeEvent(CHANGE_STEP,0,HEAVY)])
r.reset_state()
HP, RW, MODE, PP = [], [], [], []
for _ in range(EVAL_H):
    s = r.step(render=False)
    h,_,done = _victim_state(r.env)
    HP.append(h); RW.append(float(np.sum(s.rewards))); MODE.append(s.mode[0]); PP.append(float(s.detector_p[0]))
    if done: break
scen = r.env.scenario; resc = int(sum(bool(sv.rescued[0]) for sv in scen._survivals))
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
ax[0].plot(HP, color="C3"); ax[0].axvline(CHANGE_STEP, color="C3", lw=2); ax[0].set_ylabel("min victim health")
ax[0].set_title(f"Rescue episode: {resc}/5 rescued in {len(HP)} steps")
ax[1].plot(PP, color="C0"); ax[1].axhline(best["thr"], ls="--", color="k", lw=.7); ax[1].axvline(CHANGE_STEP, color="C3", lw=2)
ax[1].set_ylabel("detector $p_t$"); ax[1].set_ylim(-.05,1.05)
ax[2].plot(np.cumsum(RW), color="C2"); ax[2].axvline(CHANGE_STEP, color="C3", lw=2)
ax[2].set_ylabel("cumulative return"); ax[2].set_xlabel("step")
fig.tight_layout(); fig.savefig(out / "fig5_rescue_episode.png", dpi=120); plt.close(fig)

summary = dict(minutes=round((time.time()-t0)/60, 1), config_file=CFG, p1_iters=P1_ITERS, horizon=HORIZON,
               p1_return_end=(float(np.nanmean(L["episode_return_agent0"][-100:])) if log else "loaded"),
               device=DEVICE, cpd_loss=CPD_LOSS, randomize_map=bool(RANDMAP), collision_penalty=COLLISION,
               bce_w=BCE_W, len_segment=LEN_SEGMENT,
               p2_epochs=P2_EPOCHS, best_thr=best["thr"], f1=round(best["f1"],3),
               tpr=round(best["tpr"],3), fpr=round(best["fpr"],3), mean_delay=round(best["delay"],1),
               rescued=f"{resc}/5")
(out / "summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in summary.items()))
print(f"\n=== DONE in {summary['minutes']} min ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
print("figures:", ", ".join(p.name for p in sorted(out.glob("fig*.png"))))

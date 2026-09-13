import csv
import math
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p1_path, p2_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

p1 = list(csv.DictReader(open(p1_path)))
p2 = list(csv.DictReader(open(p2_path)))

def col(rows, name):
    return np.array([float(r[name]) for r in rows])

def smooth(x, w=51):
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]

def plot_raw_and_smooth(ax, x, y, color, w=51, label=None):
    ax.plot(x, y, lw=0.3, color=color, alpha=0.25)
    ax.plot(x, smooth(y, w), lw=1.2, color=color, label=label)

it1 = col(p1, "iter")
chance = math.log(17)

# ---------------------------------------------------------------
# Figure A: representation + detector losses (5 panels)
# ---------------------------------------------------------------
elbo = col(p1, "l_elbo")
cpc = col(p1, "l_cpc")
rho = col(p1, "l_rho")
cov = col(p1, "l_cov")
it2 = col(p2, "iter")
cpd = col(p2, "l_cpd")

fig, axes = plt.subplots(1, 5, figsize=(19, 3.2))
ax = axes[0]; plot_raw_and_smooth(ax, it1, elbo, "#1f5aa6"); ax.set_yscale("log")
ax.set_title(r"(a) $\mathcal{L}_{\mathrm{ELBO}}$"); ax.set_xlabel("Phase 1 iter")

ax = axes[1]; plot_raw_and_smooth(ax, it1, cpc, "#1f5aa6")
ax.axhline(chance, color="#b02418", lw=1.0, ls="--", label=r"chance")
ax.set_title(r"(b) $\mathcal{L}_{\mathrm{CPC}}$"); ax.set_xlabel("Phase 1 iter"); ax.legend(fontsize=7)

ax = axes[2]; plot_raw_and_smooth(ax, it1, rho, "#1f5aa6")
ax.set_title(r"(c) $\mathcal{L}_{\rho}$"); ax.set_xlabel("Phase 1 iter")

ax = axes[3]; plot_raw_and_smooth(ax, it1, cov, "#1f5aa6"); ax.set_yscale("log")
ax.set_title(r"(d) $\mathcal{L}_{\mathrm{cov}}$"); ax.set_xlabel("Phase 1 iter")

ax = axes[4]; ax.plot(it2, cpd, lw=1.0, color="#1f5aa6", marker="o", markersize=2)
ax.set_title(r"(e) $\mathcal{L}_{\mathrm{CPD}}$ (Phase 2)"); ax.set_xlabel("Phase 2 iter")

for ax in axes:
    ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
fig.savefig(f"{out_dir}/vae-training-curves.pdf", bbox_inches="tight")
print("wrote", f"{out_dir}/vae-training-curves.pdf")

# ---------------------------------------------------------------
# Figure B: task performance (3 panels)
# ---------------------------------------------------------------
ret_team = col(p1, "episode_return_team")
ret_a0 = col(p1, "episode_return_agent0")
ret_a1 = col(p1, "episode_return_agent1")
rescued_frac = col(p1, "rescued_frac")
rescued_count = col(p1, "rescued_count")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
ax = axes[0]
plot_raw_and_smooth(ax, it1, ret_team, "#1f5aa6", label="team")
ax.set_title("(a) Team episode return"); ax.set_xlabel("Phase 1 iter"); ax.set_ylabel("return")

ax = axes[1]
ax.plot(it1, smooth(ret_a0), lw=1.1, color="#1f5aa6", label="agent 0")
ax.plot(it1, smooth(ret_a1), lw=1.1, color="#b02418", label="agent 1")
ax.set_title("(b) Per-agent return (smoothed)"); ax.set_xlabel("Phase 1 iter"); ax.legend(fontsize=8)

ax = axes[2]
ax.plot(it1, smooth(rescued_frac, 101), lw=1.2, color="#1f5aa6")
ax.set_title("(c) Rescued fraction (smoothed)"); ax.set_xlabel("Phase 1 iter"); ax.set_ylabel("fraction of victims")

for ax in axes:
    ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
fig.savefig(f"{out_dir}/vae-task-performance.pdf", bbox_inches="tight")
print("wrote", f"{out_dir}/vae-task-performance.pdf")

# ---------------------------------------------------------------
# Figure C: RL training dynamics -- exploration (top) vs execution (bottom)
# ---------------------------------------------------------------
exp_cl = col(p1, "exp_critic_loss")
exp_pl = col(p1, "exp_policy_loss")
alpha_exp = col(p1, "alpha_exp_mean")
exe_cl = col(p1, "exe_critic_loss")
exe_pl = col(p1, "exe_policy_loss")
alpha_exe = col(p1, "alpha_exe")

fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
ax = axes[0, 0]; plot_raw_and_smooth(ax, it1, exp_cl, "#1f5aa6"); ax.set_yscale("log")
ax.set_title(r"(a) Exploration critic loss"); ax.set_ylabel("exploration")

ax = axes[0, 1]; plot_raw_and_smooth(ax, it1, exp_pl, "#1f5aa6")
ax.set_title(r"(b) Exploration policy loss")

ax = axes[0, 2]; ax.plot(it1, alpha_exp, lw=1.0, color="#1f5aa6")
ax.set_title(r"(c) $\alpha_{\mathrm{exp}}$ (entropy temperature)")

ax = axes[1, 0]; plot_raw_and_smooth(ax, it1, exe_cl, "#1f5aa6"); ax.set_yscale("log")
ax.set_title(r"(d) Execution critic loss"); ax.set_xlabel("Phase 1 iter"); ax.set_ylabel("execution")

ax = axes[1, 1]; plot_raw_and_smooth(ax, it1, exe_pl, "#1f5aa6")
ax.set_title(r"(e) Execution policy loss"); ax.set_xlabel("Phase 1 iter")

ax = axes[1, 2]; ax.plot(it1, alpha_exe, lw=1.0, color="#1f5aa6")
ax.set_title(r"(f) $\alpha_{\mathrm{exe}}$ (entropy temperature)"); ax.set_xlabel("Phase 1 iter")

for row in axes:
    for ax in row:
        ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
fig.savefig(f"{out_dir}/vae-rl-dynamics.pdf", bbox_inches="tight")
print("wrote", f"{out_dir}/vae-rl-dynamics.pdf")

# ---------------------------------------------------------------
# printed summary stats for the writeup
# ---------------------------------------------------------------
def last_mean(x, n=200):
    return float(np.mean(x[-n:]))

print()
print("=== summary stats (mean of last 200 logged Phase-1 iters unless noted) ===")
print(f"l_elbo:        {last_mean(elbo):.2f}")
print(f"l_cpc:         {last_mean(cpc):.4f}  (chance = {chance:.4f})")
print(f"l_rho:         {last_mean(rho):.4f}")
print(f"l_cov:         {last_mean(cov):.6f}")
print(f"l_cpd (last):  {cpd[-1]:.4f}  (first: {cpd[0]:.4f})")
print(f"episode_return_team:   {last_mean(ret_team):.2f}  (first 50 mean: {np.mean(ret_team[:50]):.2f})")
print(f"rescued_frac:          {last_mean(rescued_frac):.4f}  (max over run: {rescued_frac.max():.4f})")
print(f"exp_critic_loss:       {last_mean(exp_cl):.4f}  (max: {exp_cl.max():.2f})")
print(f"exe_critic_loss:       {last_mean(exe_cl):.4f}  (max: {exe_cl.max():.2f})")
print(f"alpha_exp_mean:        {last_mean(alpha_exp):.4f}  (first: {alpha_exp[0]:.4f})")
print(f"alpha_exe:             {last_mean(alpha_exe):.4f}  (first: {alpha_exe[0]:.4f})")
print(f"n phase1 iters logged: {len(p1)} (iter {int(it1[0])}-{int(it1[-1])})")
print(f"n phase2 iters logged: {len(p2)} (iter {int(it2[0])}-{int(it2[-1])})")

import sys
sys.path.insert(0, "/home/abodeh/thesis")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
from method.io import load_model
from method.training.meta_test import MetaTestRunner, MetaTestConfig, RegimeChangeEvent, AdversarialAttackEvent
from method.training.regime import Regime
from utils.scenario import Scenario

NOMINAL = Regime(mass=0.9, linear_friction=0.03)
change_step = 60
horizon = 200
n_seeds = 5
eps_list = [0.0, 0.15, 0.3]

trainer, detector = load_model(
    "runs/uni_20260912_170324/phase1_checkpoint_5500.pt",
    "runs/20260913_175126_phase2only/phase2_detector.pt",
)
n = trainer.n_agents


def probe_episode(eps_a, seed):
    torch.manual_seed(1000 + seed)
    cfg = MetaTestConfig()
    cfg.eps_pos = eps_a
    cfg.eps_action = eps_a
    runner = MetaTestRunner(trainer, detector, lambda: Scenario(config_file="map.yaml"), cfg)
    events = [RegimeChangeEvent(0, i, NOMINAL) for i in range(n)] + [AdversarialAttackEvent(change_step, 0, True, True)]
    runner.schedule(events)
    runner.reset_state()
    eps_hats, sigma2s = [], []
    for t in range(horizon):
        runner.step(render=False)
        with torch.no_grad():
            x_exe = torch.cat(
                [runner.obs[0], torch.zeros(1, runner.p1.action_dim, device=runner.device)], dim=-1
            ).unsqueeze(1)
            mu_rho, sigma2_rho = runner.p1.budget_encoder(x_exe)
            eps_hat = runner.p1.budget_decoder(mu_rho)
        eps_hats.append(eps_hat.item())
        sigma2s.append(sigma2_rho.mean().item())
    return np.array(eps_hats), np.array(sigma2s)


results_eps, results_sigma2 = {}, {}
for eps_a in eps_list:
    eh, s2 = [], []
    for seed in range(n_seeds):
        e, s = probe_episode(eps_a, seed)
        eh.append(e)
        s2.append(s)
    results_eps[eps_a] = np.array(eh)     # (n_seeds, horizon)
    results_sigma2[eps_a] = np.array(s2)  # (n_seeds, horizon)

colors = {0.0: "#2a9d5c", 0.15: "#e08e2c", 0.3: "#b02418"}
t_axis = np.arange(horizon)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
for eps_a in eps_list:
    arr = results_eps[eps_a]
    mean_trace = arr.mean(axis=0)
    for i in range(arr.shape[0]):
        ax.plot(t_axis, arr[i], lw=0.4, color=colors[eps_a], alpha=0.25)
    ax.plot(t_axis, mean_trace, lw=1.8, color=colors[eps_a], label=rf"true $\epsilon_a={eps_a}$")
    ax.axhline(eps_a, color=colors[eps_a], lw=1.0, ls="--", alpha=0.8)
ax.axvline(change_step, color="#555555", lw=1.1, ls=":", label=f"attack on ($t={change_step}$)")
ax.set_xlabel("environment step $t$")
ax.set_ylabel(r"inferred budget $\hat\epsilon = p_\eta(\mu_\rho)$")
ax.set_title(r"(a) Inferred vs.\ true perturbation budget")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.25, linewidth=0.5)

ax = axes[1]
for eps_a in eps_list:
    arr = results_sigma2[eps_a]
    ax.plot(t_axis, arr.mean(axis=0), lw=1.6, color=colors[eps_a], label=rf"$\epsilon_a={eps_a}$")
ax.axvline(change_step, color="#555555", lw=1.1, ls=":")
ax.set_xlabel("environment step $t$")
ax.set_ylabel(r"posterior spread $\overline{\sigma^2_\rho}$")
ax.set_title(r"(b) Posterior uncertainty over the episode")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, linewidth=0.5)

fig.tight_layout()
out = "/home/abodeh/thesis/vae-perturbation-inference.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

print()
print(f"{'eps_a':>6s} {'eps_hat_mean':>14s} {'eps_hat_std':>12s} {'pre_mean':>10s} {'post_mean':>10s} {'sigma2_mean':>12s}")
for eps_a in eps_list:
    arr = results_eps[eps_a]
    s2 = results_sigma2[eps_a]
    print(f"{eps_a:6.2f} {arr.mean():14.4f} {arr.std():12.4f} "
          f"{arr[:, :change_step].mean():10.4f} {arr[:, change_step:].mean():10.4f} {s2.mean():12.5f}")

mae = np.mean([abs(results_eps[e].mean() - e) for e in eps_list])
print()
print("overall MAE of mean eps_hat vs true eps_a across the 3 budgets:", mae)

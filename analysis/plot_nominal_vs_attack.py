import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = list(csv.DictReader(open("/home/abodeh/thesis/runs/eval_vae_20260913/summary.csv")))

def f(r, k):
    v = r[k]
    return float(v) if v not in ("", "nan") else float("nan")

def find(cond, eps):
    for r in rows:
        if r["condition"] == cond and abs(f(r, "eps_a") - eps) < 1e-9:
            return r
    raise KeyError((cond, eps))

nominal = find("nominal", 0.0)
attack0 = find("attack", 0.0)
attack15 = find("attack", 0.15)
attack30 = find("attack", 0.3)
change = find("change", 0.0)
sim0 = find("simultaneous", 0.0)
sim15 = find("simultaneous", 0.15)
sim30 = find("simultaneous", 0.3)

entries = [
    ("Nominal", nominal, "#7a7a7a"),
    ("Attack\n$\\epsilon_a{=}0$", attack0, "#a9c6e8"),
    ("Attack\n$\\epsilon_a{=}0.15$", attack15, "#5b8fd6"),
    ("Attack\n$\\epsilon_a{=}0.3$", attack30, "#1f5aa6"),
    ("Switch\nonly", change, "#e08e2c"),
    ("Both\n$\\epsilon_a{=}0$", sim0, "#e6a3a3"),
    ("Both\n$\\epsilon_a{=}0.15$", sim15, "#c85c5c"),
    ("Both\n$\\epsilon_a{=}0.3$", sim30, "#b02418"),
]

labels = [e[0] for e in entries]
iqm = [f(e[1], "return_iqm") for e in entries]
lo = [f(e[1], "return_ci_lo") for e in entries]
hi = [f(e[1], "return_ci_hi") for e in entries]
colors = [e[2] for e in entries]

nominal_iqm = f(nominal, "return_iqm")

x = np.arange(len(labels))
iqm_arr, lo_arr, hi_arr = np.array(iqm), np.array(lo), np.array(hi)

y_min = min(lo_arr.min(), iqm_arr.min()) - 1.5
y_max = max(hi_arr.max(), iqm_arr.max()) + 1.5

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhline(nominal_iqm, color="black", lw=1.0, ls="--", alpha=0.6, zorder=1,
           label=f"nominal IQM ({nominal_iqm:.1f})")
for xi, m, l, h, c in zip(x, iqm, lo, hi, colors):
    ax.errorbar(xi, m, yerr=[[m - l], [h - m]], fmt="o", color=c, ecolor=c,
                elinewidth=2, capsize=5, markersize=9, markeredgecolor="black",
                markeredgewidth=0.6, zorder=3)
    ax.annotate(f"{m:.1f}", (xi, m), xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=8)
    if h - l > 1e-6:
        ax.annotate(f"CI hi {h:.1f}", (xi, h), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=c)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_xlim(-0.6, len(labels) - 0.4)
ax.set_ylim(y_min, y_max)
ax.set_ylabel("team return (IQM, 95% bootstrap CI)")
ax.set_title("Nominal performance vs. performance under attack and regime switch")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, linewidth=0.5, axis="y")
fig.tight_layout()
out = "/home/abodeh/thesis/vae-nominal-vs-attack.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

print()
print(f"{'condition':20s} {'iqm':>10s} {'mean':>10s} {'std':>8s} {'ci_lo':>10s} {'ci_hi':>10s}")
for name, r, _ in entries:
    print(f"{name.replace(chr(10),' '):20s} {f(r,'return_iqm'):10.3f} {f(r,'return_mean'):10.3f} "
          f"{f(r,'return_std'):8.3f} {f(r,'return_ci_lo'):10.3f} {f(r,'return_ci_hi'):10.3f}")

print()
print("max |iqm - nominal_iqm| across all conditions:",
      max(abs(v - nominal_iqm) for v in iqm))
print("max |mean - nominal_mean| across all conditions:",
      max(abs(f(e[1], "return_mean") - f(nominal, "return_mean")) for e in entries))

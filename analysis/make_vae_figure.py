import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p1_path, p2_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

p1 = list(csv.DictReader(open(p1_path)))
p2 = list(csv.DictReader(open(p2_path)))

def col(rows, name):
    return [float(r[name]) for r in rows]

it1 = col(p1, "iter")
elbo = col(p1, "l_elbo")
cpc = col(p1, "l_cpc")
it2 = col(p2, "iter")
cpd = col(p2, "l_cpd")

import math
chance = math.log(17)  # log(K+1), K=16 negatives

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

ax = axes[0]
ax.plot(it1, elbo, lw=0.6, color="#1f5aa6")
ax.set_yscale("log")
ax.set_xlabel("Phase 1 iteration")
ax.set_ylabel(r"$\mathcal{L}_{\mathrm{ELBO}}$ (log scale)")
ax.set_title("(a) Reconstruction loss")
ax.grid(alpha=0.25, linewidth=0.5)

ax = axes[1]
ax.plot(it1, cpc, lw=0.6, color="#1f5aa6")
ax.axhline(chance, color="#b02418", lw=1.0, ls="--", label=r"chance ($\log 17$)")
ax.set_xlabel("Phase 1 iteration")
ax.set_ylabel(r"$\mathcal{L}_{\mathrm{CPC}}$")
ax.set_title("(b) Contrastive loss")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25, linewidth=0.5)

ax = axes[2]
ax.plot(it2, cpd, lw=1.0, color="#1f5aa6", marker="o", markersize=2)
ax.set_xlabel("Phase 2 iteration")
ax.set_ylabel(r"$\mathcal{L}_{\mathrm{CPD}}$")
ax.set_title("(c) Detector loss")
ax.grid(alpha=0.25, linewidth=0.5)

fig.tight_layout()
fig.savefig(out_path, bbox_inches="tight")
print("wrote", out_path)
print(f"phase1 rows={len(p1)} (iters {int(it1[0])}-{int(it1[-1])})")
print(f"phase2 rows={len(p2)} (iters {int(it2[0])}-{int(it2[-1])})")
print(f"final l_elbo(mean last 200)={sum(elbo[-200:])/200:.2f}  final l_cpc(mean last 200)={sum(cpc[-200:])/200:.4f}  chance={chance:.4f}")
print(f"final l_cpd(last)={cpd[-1]:.4f}")

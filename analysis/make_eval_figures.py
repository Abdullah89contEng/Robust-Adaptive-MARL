import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

summary_path, out_dir = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(summary_path)))

def f(r, k):
    v = r[k]
    return float(v) if v not in ("", "nan") else float("nan")

# ---------------------------------------------------------------
# Figure E1: detection outcomes, by condition (only ones with a true change)
# ---------------------------------------------------------------
det_conditions = ["change", "simultaneous"]
det_rows = {}
for r in rows:
    if r["condition"] in det_conditions and (r["condition"] == "change" or f(r, "eps_a") == 0.3):
        det_rows[r["condition"]] = r
# also grab simultaneous at eps=0.3 specifically as the "hardest" combined case;
# change has only one eps value (0.0) since eps only varies for attack/simultaneous
labels = ["change\n(switch only)", "simultaneous\n(switch + attack, eps=0.3)"]
tpr = [f(det_rows["change"], "tpr"), f(det_rows["simultaneous"], "tpr")]
tnr = [f(det_rows["change"], "tnr"), f(det_rows["simultaneous"], "tnr")]
fa = [f(det_rows["change"], "false_alarm_rate"), f(det_rows["simultaneous"], "false_alarm_rate")]
miss = [f(det_rows["change"], "missed_detection_rate"), f(det_rows["simultaneous"], "missed_detection_rate")]

x = np.arange(len(labels))
w = 0.2
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(x - 1.5*w, tpr, w, label="TPR", color="#1f5aa6")
ax.bar(x - 0.5*w, tnr, w, label="TNR", color="#2a9d5c")
ax.bar(x + 0.5*w, fa, w, label="false-alarm rate", color="#e08e2c")
ax.bar(x + 1.5*w, miss, w, label="missed-detection rate", color="#b02418")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel("rate")
ax.legend(fontsize=8, ncol=2, loc="upper center")
ax.set_title("Detection outcomes (15 episodes/condition, 3 seeds x 5 episodes)")
ax.grid(alpha=0.25, linewidth=0.5, axis="y")
fig.tight_layout()
fig.savefig(f"{out_dir}/vae-eval-detection.pdf", bbox_inches="tight")
print("wrote vae-eval-detection.pdf")
print("change:", det_rows["change"])
print("simultaneous@0.3:", det_rows["simultaneous"])

# ---------------------------------------------------------------
# Figure E2: return vs attack budget (attack-only and simultaneous), with nominal reference
# ---------------------------------------------------------------
nominal = [r for r in rows if r["condition"] == "nominal"][0]
nominal_iqm = f(nominal, "return_iqm")

fig, ax = plt.subplots(figsize=(6.5, 4))
for cond, color, marker in [("attack", "#1f5aa6", "o"), ("simultaneous", "#b02418", "s")]:
    crows = sorted([r for r in rows if r["condition"] == cond], key=lambda r: f(r, "eps_a"))
    eps = [f(r, "eps_a") for r in crows]
    iqm = [f(r, "return_iqm") for r in crows]
    lo = [f(r, "return_ci_lo") for r in crows]
    hi = [f(r, "return_ci_hi") for r in crows]
    yerr = [np.array(iqm) - np.array(lo), np.array(hi) - np.array(iqm)]
    ax.errorbar(eps, iqm, yerr=yerr, marker=marker, color=color, capsize=3, label=cond)

ax.axhline(nominal_iqm, color="gray", ls="--", lw=1.0, label="nominal (eps=0, no switch)")
ax.set_xlabel(r"attack budget $\epsilon_a$")
ax.set_ylabel("team return (IQM, 95% bootstrap CI)")
ax.set_title("Return vs. attack budget")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
fig.savefig(f"{out_dir}/vae-eval-return.pdf", bbox_inches="tight")
print("wrote vae-eval-return.pdf")
for r in rows:
    print(r["condition"], r["eps_a"], "iqm=", r["return_iqm"], "ci=[", r["return_ci_lo"], ",", r["return_ci_hi"], "]")

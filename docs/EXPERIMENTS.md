# Running the Chapter-11 experiments

## What is here

| Piece | File | Status |
|---|---|---|
| Metrics (TP/TN/delay, recovery time, degradation, IQM+CI) | `method/metrics.py` | done, self-tested |
| Main runner: {variant × condition × eps_a × seed × episode} → tidy CSV | `scripts/run_experiments.py` | done, smoke-tested |
| Perturbation-inference (ε_a identifiability) experiment | `scripts/eval_rho_inference.py` | done, smoke-tested |
| Model loader | `method/io.py` | done |
| Interactive playback | `method/viz.py` | done |

Thresholds in `method/metrics.py` (`W_DETECT=10`, `SEG_TN=20`, `REC_BAND=0.10`,
`REC_WINDOW=20`) match the thesis "Metrics" section and are **fixed a priori** — do not
tune them on results.

## Prerequisites

1. A trained Phase 1 checkpoint and a trained Phase 2 detector. The checkpoints behind the
   thesis's results chapter: `runs/uni_20260912_170324/phase1_checkpoint_5500.pt` and
   `runs/20260913_175126_phase2only/phase2_detector.pt`. If you don't pass `--phase1-ckpt`
   / `--detector`, `method.io.load_model` auto-picks whatever is newest under `runs/`.
2. **That checkpoint is still undertrained** (Phase 1 stopped at 5,747 of a planned 20,000
   iterations, Phase 2 at 121 of a planned 300 — see the thesis's results chapter for why).
   Agents rarely reach a victim, so return sits on a passive floor. Detection and ε_a metrics
   still run fine on it, but the *return* / *degradation* / *recovery* columns won't mean
   anything until Phase 1 is trained further. Resume with `scripts/train.py --resume <run
   dir> --phase1-iters <N>` before trusting those columns.

## Full run

```bash
conda activate marl
MPLBACKEND=Agg python scripts/run_experiments.py \
    --seeds 5 --episodes 20 --horizon 200 --change-step 60 \
    --eps-list 0.0,0.05,0.1,0.2,0.3 \
    --variants full,detector_off,no_reexplore \
    --conditions nominal,attack,change,simultaneous
```

Writes `runs/experiments_<ts>/results.csv` (per episode) and `summary.csv` (per
variant×condition×eps_a: TPR, TNR, false-alarm rate, delay, recovery, return IQM + 95% CI,
degradation vs. the variant's own nominal).

```bash
MPLBACKEND=Agg python scripts/eval_rho_inference.py \
    --eps-grid 0.0,0.05,0.1,0.2,0.3,0.45,0.6 --episodes 20 --horizon 120
```

Writes `runs/rho_inference_<ts>/summary.csv`: `eps_hat_iqm`, MAE, RMSE, the constant-predictor
floor (`mae_constant_predictor`), `beats_constant`, `sigma2_rho_contracts`, and an `ood` flag
for `eps_true > 0.30`.

## Variants

`run_experiments.py` covers the variants that need **only a config toggle** on the one
trained checkpoint:

| `--variants` name | Change | Tests |
|---|---|---|
| `full` | none | reference |
| `detector_off` | `threshold_C = 10` (never fires) | value of the explicit reset |
| `no_reexplore` | `re_exploration_budget_K = 0` | value of the re-exploration window vs. the reset alone |

A few more variants are wired into `run_experiments.py` (`plain_fmasac`, `context_only`,
`no_adversary`, `maal_random`, `mean_pooling`, `gru_belief`, `no_covariance`, `single_phase`,
`fixed_epsilon`, `sample_belief`, `q_zero`), each needing a **separately trained checkpoint**
with that component removed at training time; the thesis itself doesn't report a
component-wise study, so these are optional if you want to run one yourself.
`run_experiments.py` lists them and exits with a message if asked for one you haven't
trained.

## Conditions

| `--conditions` name | Events | Isolates |
|---|---|---|
| `nominal` | regime held at nominal, no attack | RQ3 premise: does the method match FMASAC at ε=0 |
| `attack` | PGD attack on agent 0 from `--change-step`, at each ε in `--eps-list` | RQ2/RQ3: degradation vs. ε |
| `change` | abrupt regime switch for agent 0 at `--change-step` | RQ4: detection delay, recovery |
| `simultaneous` | both at once | RQ4: interference |

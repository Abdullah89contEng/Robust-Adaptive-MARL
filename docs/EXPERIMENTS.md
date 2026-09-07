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

1. A trained Phase 1 checkpoint and a trained Phase 2 detector. Current defaults
   (auto-picked by `method.io.load_model`): `runs/20260906_162246/phase1_checkpoint_2999.pt`
   and `runs/20260906_182413_phase2only/phase2_detector.pt`.
2. **Phase 1 is currently undertrained** (~3000 iters, agents do not reach the victim, return
   is a constant floor). Detection and ε_a metrics work regardless, but the *return* /
   *degradation* / *recovery* numbers are only meaningful once Phase 1 is trained to a policy
   that actually rescues. Retrain with `scripts/train.py --phase1-iters <N>` on a larger
   config before trusting those columns.

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

The remaining ablations from the thesis table (`plain_fmasac`, `context_only`,
`no_adversary`, `maal_random`, `mean_pooling`, `gru_belief`, `no_covariance`, `single_phase`,
`fixed_epsilon`, `sample_belief`, `q_zero`) each need a **separately trained checkpoint** with
that component removed at training time. `run_experiments.py` lists them and exits with a
message if asked for one — training those checkpoints is the remaining work.

## Conditions

| `--conditions` name | Events | Isolates |
|---|---|---|
| `nominal` | regime held at nominal, no attack | RQ3 premise: does the method match FMASAC at ε=0 |
| `attack` | PGD attack on agent 0 from `--change-step`, at each ε in `--eps-list` | RQ2/RQ3: degradation vs. ε |
| `change` | abrupt regime switch for agent 0 at `--change-step` | RQ4: detection delay, recovery |
| `simultaneous` | both at once | RQ4: interference |

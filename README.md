# Two-Phase Exchangeable-Context CCM-FMASAC

Code for the MARL method in the thesis: a cooperative team that infers its own task context
with an exchangeable-process posterior, explores with a closed-form information-gain reward,
stays robust to a worst-case teammate-action perturbation, and resets itself when a
representation-learned detector catches a regime change.

See `method-spec.md` for the full method spec every module's docstring cites by section
number, `architecture.md` for a class-level diagram, and `docs/EXPERIMENTS.md` for the
evaluation harness in more detail. The declared source of truth is `ch-proposed.tex` (thesis
Ch. 10).

## Setup

```bash
conda create -n marl python=3.11
conda activate marl
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Everything here was trained and evaluated on CPU (see the thesis for why). `requirements.txt`
has exact pinned versions.

## Layout

- `method/` -- the model itself: encoders (flow, BRUNO/VAE posterior, budget encoder),
  policies, critics, the change-point detector, the disentangling loss, replay buffers.
- `utils/` -- the environment: `Scenario`/`Map` (the search-and-rescue task), victims,
  ears/lidar sensing, the physics task sampler for domain randomization.
- `scripts/` -- entry points for training, evaluation, and video recording. Details below.
- `configs/` -- scenario YAML files beyond the ones at the repo root (`map.yaml`,
  `world_config*.yaml`).
- `notebooks/` -- exploratory and reporting notebooks. Details below.
- `analysis/` -- one-off scripts that produced the figures in the thesis's results chapter.
  Each one hardcodes the checkpoint paths it was run against, so treat these as a record of
  how a specific figure was made, not a general-purpose tool.
- `third-party/` -- vendored baseline implementations referenced in the thesis (CCM, InDiD,
  M3DDPG, MASAC).
- `runs/` -- training output (checkpoints, logs, eval CSVs). Not tracked in git; large and
  disposable except for whatever run you actually care about keeping.

## Training

Phase 1 trains the encoder, exploration policy, and execution policy jointly. Phase 2 freezes
the encoder and trains the change-point detector on top of it. `scripts/run.sh` runs both,
back to back, and reads every setting from an environment variable so nothing needs editing:

```bash
bash scripts/run.sh                                  # defaults: map.yaml, BRUNO posterior
CONTEXT_MODE=vae bash scripts/run.sh                 # learned GRU posterior instead of BRUNO
CONFIG=world_config_random.yaml bash scripts/run.sh  # different map
DEVICE=cuda N_ENVS=1024 bash scripts/run.sh          # on a GPU node
RESUME=runs/uni_20260912_170324 bash scripts/run.sh  # continue a run that stopped early
```

It checkpoints periodically (`CKPT_EVERY`, default 500 iterations), so a run that gets killed
partway through doesn't lose its progress. Output lands in `runs/uni_<timestamp>/`.

`scripts/train.py` is what `run.sh` calls; run it directly if you want flags instead of
env vars:

```bash
python scripts/train.py --context-mode bruno --phase1-iters 20000 --phase2-iters 300 \
    --n-envs 10 --horizon 350 --config-file map.yaml --checkpoint-every 500
```

`--context-mode {bruno,vae}` picks between the exact exchangeable-process posterior (the
adopted design) and an amortized GRU posterior + decoder (the alternative discussed in the
thesis). `--randomize-map` turns on a fresh obstacle/victim/agent layout every episode.

To train Phase 2 on an existing Phase 1 checkpoint without redoing Phase 1:

```bash
python scripts/train_phase2.py --phase1-ckpt runs/uni_.../phase1_checkpoint_N.pt \
    --phase2-iters 300 --checkpoint-every 10
```

`--checkpoint-every` defaults to 0 (off) here -- set it to something small if the run might
get cut off before it finishes.

For a quick sanity check rather than a full run, `scripts/train_fast.py` does a short Phase 1
+ Phase 2 pass end to end in under half an hour and dumps its own figures to `runs/fast_<ts>/`.
It's configured through env vars the same way `run.sh` is (see the top of the file).

## Evaluation

```bash
python scripts/run_experiments.py \
    --phase1-ckpt runs/uni_.../phase1_checkpoint_N.pt \
    --detector runs/.../phase2_detector.pt \
    --seeds 5 --episodes 10 --horizon 200 --eps-list 0.0,0.15,0.3 \
    --conditions nominal,attack,change,simultaneous
```

Writes `results.csv` (one row per episode) and `summary.csv` (TPR/TNR, false-alarm rate,
detection delay, recovery time, return IQM with a 95% bootstrap CI, per variant x condition x
epsilon). `docs/EXPERIMENTS.md` explains what each condition and variant actually does.

`scripts/meta_test_eval.py` is a smaller, single-scenario version of the same evaluation.
`scripts/record_metatest_video.py` renders one episode to an `.mp4` so you can watch what the
policy is actually doing under a given scenario instead of reading numbers:

```bash
python scripts/record_metatest_video.py \
    --phase1-ckpt runs/uni_.../phase1_checkpoint_N.pt \
    --detector runs/.../phase2_detector.pt \
    --scenario change --change-step 60 --horizon 200 --out runs/demo.mp4
```

## Notebooks

- `notebooks/deployment_test.ipynb` -- walks through the frozen meta-test loop
  (Algorithm 10.3) step by step: context reset, re-exploration, the detector firing.
- `notebooks/test_environment_report.ipynb` -- detector threshold sweep and
  return-under-attack report for one checkpoint.
- `notebooks/training_report.ipynb` -- a training-curve report for an earlier run
  (`runs/retrain_v3_phys`), from before the run described in the thesis. Kept for reference,
  not the checkpoint the thesis reports on.

## Analysis scripts

Everything under `analysis/` was run once, from the repo root, against the checkpoints in
`runs/uni_20260912_170324/` and `runs/20260913_175126_phase2only/`, to produce a specific
figure or number in the thesis's results chapter. They're not a reusable library -- for a new
checkpoint, copy the pattern rather than expecting a flag for it.

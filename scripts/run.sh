#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Training launcher: Two-Phase Exchangeable-Context CCM-FMASAC (Phase 1 + 2).
#
#   bash scripts/run.sh
#   THREADS=4 bash scripts/run.sh                     # sweep 2/4/8 on CPU
#   DEVICE=cuda N_ENVS=1024 RANDMAP=0 bash scripts/run.sh   # real GPU node
#   CONFIG=world_config_random.yaml bash scripts/run.sh
#   RESUME=runs/uni_20260909_120000 bash scripts/run.sh
#   sbatch scripts/run.sh                             # also a SLURM job
#
# Every setting is an env-var override -- no need to edit this file.
# ---------------------------------------------------------------------------
#SBATCH --job-name=ccm-fmasac
#SBATCH --gres=gpu:1                # drop this line for a CPU-only partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"

# ---- config ----------------------------------------------------------
CONDA_ENV=${CONDA_ENV:-marl}
CONFIG=${CONFIG:-world_config_random.yaml}
DEVICE=${DEVICE:-auto}                 # auto | cpu | cuda | cuda:N
N_ENVS=${N_ENVS:-10}                   # the only real parallelism knob on CPU;
                                       # 256-1024 on a real GPU
HORIZON=${HORIZON:-350}
P1_ITERS=${P1_ITERS:-20000}
P2_ITERS=${P2_ITERS:-1000}
SHAPING=${SHAPING:-0.3}
COLLISION=${COLLISION:-0.5}            # per-agent per-step overlap penalty (0 = off)
DETECTOR_LOSS=${DETECTOR_LOSS:-paper}  # paper (arXiv:2510.24988v1) | indid
RANDMAP=${RANDMAP:-1}                  # 1 = --randomize-map; 0 = off
                                       # (the per-env reshuffle is O(N_ENVS)
                                       #  pure-Python -- set 0 at high N_ENVS)
SEED=${SEED:-0}
CKPT_EVERY=${CKPT_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-20}
THREADS=${THREADS:-}                   # blank -> auto (min(8, available CPUs))
# --------------------------------------------------------------------

# ---- conda ---------------------------------------------------------
set +u
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

# ---- CPU threads -------------------------------------------------
# How many CPUs may this process actually use?
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  AVAIL_CPUS=$SLURM_CPUS_PER_TASK
elif [[ -r /sys/fs/cgroup/cpu.max ]] && read -r q _ < /sys/fs/cgroup/cpu.max && [[ "$q" != max ]]; then
  AVAIL_CPUS=$(( (q + 99999) / 100000 ))
else
  AVAIL_CPUS=$(nproc 2>/dev/null || echo 4)
fi
# The models here are tiny; oversubscribed BLAS on small matmuls pegs one
# core while the rest stall. Cap threads low unless told otherwise.
if [[ -z "$THREADS" ]]; then
  THREADS=$(( AVAIL_CPUS < 8 ? AVAIL_CPUS : 8 ))
  (( THREADS < 1 )) && THREADS=1
fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
       OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close OMP_PLACES=cores
export MALLOC_ARENA_MAX=4              # tame glibc malloc under many threads
export PYTHONUNBUFFERED=1              # live console.log

[[ "$AVAIL_CPUS" -le 1 ]] && echo "WARNING: only $AVAIL_CPUS CPU visible -- ask for more (--cpus-per-task=8)."

# ---- args --------------------------------------------------------
ARGS=(
  --config-file "$CONFIG"
  --shaping-weight "$SHAPING"
  --collision-penalty "$COLLISION"
  --device "$DEVICE"
  --n-envs "$N_ENVS"
  --horizon "$HORIZON"
  --phase1-iters "$P1_ITERS"
  --phase2-iters "$P2_ITERS"
  --detector-loss "$DETECTOR_LOSS"
  --checkpoint-every "$CKPT_EVERY"
  --log-every "$LOG_EVERY"
  --seed "$SEED"
)
[[ "$RANDMAP" == "1" ]] && ARGS+=(--randomize-map)

if [[ -n "${RESUME:-}" ]]; then
  RUN_DIR=$RESUME;                         ARGS+=(--resume "$RESUME")
else
  RUN_DIR=runs/uni_$(date +%Y%m%d_%H%M%S); ARGS+=(--out-dir "$RUN_DIR")
fi
mkdir -p "$RUN_DIR" runs

echo "env      : $CONDA_ENV"
echo "cpus     : avail=$AVAIL_CPUS  threads=$THREADS"
echo "run dir  : $RUN_DIR"
echo "command  : python scripts/train.py ${ARGS[*]}"
echo "----------------------------------------------------------------------"
python scripts/train.py "${ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/console.log"

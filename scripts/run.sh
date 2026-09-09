#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Training launcher: Two-Phase Exchangeable-Context CCM-FMASAC (Phase 1 + 2).
#
#   bash scripts/run.sh                         # defaults (below)
#   N_ENVS=512 DEVICE=cuda bash scripts/run.sh  # on a real GPU node (sm_75+)
#   CONFIG=world_config_random.yaml bash scripts/run.sh
#   RESUME=runs/uni_20260909_120000 bash scripts/run.sh   # continue a run
#   sbatch scripts/run.sh                       # also valid as a SLURM job
#
# Every knob below is overridable from the environment.
# ---------------------------------------------------------------------------
#SBATCH --job-name=ccm-fmasac
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.log

set -euo pipefail

# repo root, whether launched with `bash scripts/run.sh` or `sbatch`
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"

# ---- config -------------------------------------------------------------
CONDA_ENV=${CONDA_ENV:-r-marl}          # conda env name
CONFIG=${CONFIG:-world_config_random.yaml}  # map yaml
DEVICE=${DEVICE:-auto}                  # auto | cpu | cuda | cuda:N
                                        #   auto -> cuda if usable, else cpu
N_ENVS=${N_ENVS:-700}                     # raise to 256-512 on a real GPU
HORIZON=${HORIZON:-500}
P1_ITERS=${P1_ITERS:-20000}
P2_ITERS=${P2_ITERS:-1000}
SHAPING=${SHAPING:-0.3}
DETECTOR_LOSS=${DETECTOR_LOSS:-paper}   # paper (arXiv:2510.24988v1) | indid
SEED=${SEED:-0}
CKPT_EVERY=${CKPT_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-20}
# ----------------------------------------------------------------------

# activate conda (tolerate `set -u` inside older conda init)
set +u
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p runs

ARGS=(
  --config-file "$CONFIG"
  --shaping-weight "$SHAPING"
  --randomize-map
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

if [[ -n "${RESUME:-}" ]]; then
  RUN_DIR=$RESUME
  ARGS+=(--resume "$RESUME")
else
  RUN_DIR=runs/uni_$(date +%Y%m%d_%H%M%S)
  ARGS+=(--out-dir "$RUN_DIR")
fi
mkdir -p "$RUN_DIR"

echo "env      : $CONDA_ENV   threads=$OMP_NUM_THREADS"
echo "run dir  : $RUN_DIR"
echo "command  : python scripts/train.py ${ARGS[*]}"
echo "-----------------------------------------------------------------------"
python scripts/train.py "${ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/console.log"

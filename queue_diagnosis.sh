#!/bin/bash
# CIFAR Random: one checkpoint, mask stability and matched recovery controls.
# Submit from repository root: sbatch queue_diagnosis.sh
# Defaults reuse job 59215804's split, gold predictions and 40-shadow LiRA bank.
# Overrides: PROJECT_DIR, SOURCE_SUITE, REPETITION, TRAIN_SEED, SCORE_SEED,
# OUTPUT_ROOT (new directory), VENV_PATH, CINECA_AI_MODULE, POWER_ITERS,
# SCORE_MASS, RECOVERY_EPOCHS. To reuse a bootstrap, set BOOTSTRAP_DIR to its path.

#SBATCH --account=CASD_prod
#SBATCH --job-name=cifar_diagnosis
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
if [[ ! -f "${PROJECT_DIR}/experiments/cifar_diagnosis.py" ]]; then
    echo "PROJECT_DIR does not contain experiments/cifar_diagnosis.py: ${PROJECT_DIR}" >&2
    exit 2
fi
cd "${PROJECT_DIR}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This job runs matched controls sequentially; do not submit as an array." >&2
    exit 2
fi
SOURCE_SUITE="${SOURCE_SUITE:-stat_tests/CIFAR/random_job_59215804/CIFAR_random_full}"
REPETITION="${REPETITION:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-stat_tests/CIFAR/diagnosis_job_${SLURM_JOB_ID:?Run with sbatch}}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "OUTPUT_ROOT already exists: ${OUTPUT_ROOT}. Choose a new directory." >&2
    exit 2
fi
if [[ -z "${BOOTSTRAP_DIR:-}" && ! -f "${SOURCE_SUITE}/lira_shadow_bank.npz" ]]; then
    echo "Saved LiRA bank not found in SOURCE_SUITE: ${SOURCE_SUITE}" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

module purge
module load profile/deeplrn
module load "${CINECA_AI_MODULE:-cineca-ai}"

# Preserve cineca-ai's nonstandard torchvision source directory.
CINECA_PYTHON="$(command -v python)"
CINECA_TORCHVISION_ROOT="$("${CINECA_PYTHON}" - <<'PY'
from pathlib import Path
import torchvision
print(Path(torchvision.__file__).resolve().parent.parent)
PY
)"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/venv}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    echo "Python environment not found at ${VENV_PATH}." >&2
    exit 2
fi
source "${VENV_PATH}/bin/activate"
PYTHON="$(command -v python)"
export PYTHONPATH="${CINECA_TORCHVISION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" - <<'PY'
import torch
import torchvision
import backpack
import sklearn
assert torch.cuda.is_available(), 'PyTorch cannot access the allocated GPU'
print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0))
PY

echo "=== CIFAR fixed-checkpoint diagnosis ==="
echo "Job: ${SLURM_JOB_ID}; host: $(hostname); output: ${OUTPUT_ROOT}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Source revision: $(git rev-parse HEAD)"
    git status --short
fi
if [[ -z "${BOOTSTRAP_DIR:-}" ]]; then
    BOOTSTRAP_DIR="${OUTPUT_ROOT}/bootstrap"
    srun "${PYTHON}" -u -m experiments.cifar_bootstrap \
        --source-suite "${SOURCE_SUITE}" --repetition "${REPETITION}" \
        --output "${BOOTSTRAP_DIR}" --seed "${TRAIN_SEED:-3000}" --batch-size 128
fi

srun "${PYTHON}" -u -m experiments.cifar_diagnosis \
    --suite "${BOOTSTRAP_DIR}/suite" --repetition "${REPETITION}" \
    --checkpoint "${BOOTSTRAP_DIR}/original_model.pth" \
    --initial-checkpoint "${BOOTSTRAP_DIR}/initial_model.pth" \
    --output "${OUTPUT_ROOT}/controls" --seed "${SCORE_SEED:-0}" \
    --rank 20 --higher-rank 40 --baseline-power-iters 5 \
    --power-iters "${POWER_ITERS:-25}" --power-tolerance 0.001 \
    --max-samples 2048 --score-mass "${SCORE_MASS:-55.55555555555556}" \
    --epochs "${RECOVERY_EPOCHS:-5}" --batch-size 128

echo "=== Diagnosis finished: ${OUTPUT_ROOT}/controls/report.json and metrics.csv ==="
date

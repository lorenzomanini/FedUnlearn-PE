#!/bin/bash
# CIFAR-10 random-client spectral sweep on CINECA Leonardo Booster.
# Submit from the repository root: sbatch queue.sh
# Three training seeds share one 40-model online-LiRA bank; LiRA is evaluated
# at all ten score masses. Recovery ablations use queue_ablation.sh.
# Overrides: PROJECT_DIR=/path/to/repo OUTPUT_ROOT=/path/to/new/results
# VENV_PATH=/path/to/venv CINECA_AI_MODULE=cineca-ai/<version>

#SBATCH --account=CASD_prod
#SBATCH --job-name=cifar_random
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
if [[ ! -f "${PROJECT_DIR}/experiments/configs/cifar_full_validation.py" ]]; then
    echo "PROJECT_DIR does not contain the CIFAR random experiment: ${PROJECT_DIR}" >&2
    exit 2
fi
cd "${PROJECT_DIR}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This job builds one shared LiRA bank and runs all repetitions sequentially; do not use an array." >&2
    exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-stat_tests/CIFAR/random_job_${SLURM_JOB_ID:?Run with sbatch}}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "OUTPUT_ROOT already exists: ${OUTPUT_ROOT}. Choose a new directory." >&2
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

# cineca-ai supplies torchvision from a nonstandard source directory.
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
print('BackPACK and scikit-learn: OK')
PY

echo "=== CIFAR-10 random-client run ==="
echo "Job: ${SLURM_JOB_ID}; host: $(hostname); output: ${OUTPUT_ROOT}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Source revision: $(git rev-parse HEAD)"
    git status --short
fi

# Check the full-Hessian path before training the shadow bank.
srun "${PYTHON}" -u -m experiments.spectral_smoke \
    --device cuda --backend full --image-size 64 \
    --batch-size 128 --rank 20 --hmp-chunk-size 1

srun "${PYTHON}" -u -m experiments.configs.cifar_full_validation \
    --output "${OUTPUT_ROOT}"

echo "=== Validating and summarizing all repetitions and LiRA cases ==="
srun "${PYTHON}" -u validation/accuracy_audit/analyze_saved_cifar.py \
    --input "${OUTPUT_ROOT}/CIFAR_random_full" \
    --output "${OUTPUT_ROOT}/summary"

echo "=== CIFAR-10 random-client run finished ==="
date

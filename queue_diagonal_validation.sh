#!/bin/bash
# CIFAR-10 old diagonal GGN-MC sweeps on CINECA Leonardo Booster.
# Submit from the repository root: sbatch queue_diagonal_validation.sh
# A regular job runs correction off, then on, sequentially.
# Optional array: sbatch --array=0-1%1 queue_diagonal_validation.sh (0: off, 1: on).
# Each variant runs twenty training seeds and builds its own 40-model LiRA bank.
# Both use the same partition/shadow/repetition seeds and ten score masses.
# Run just one variant with --array=0 or --array=1.
# Overrides: PROJECT_DIR=/path/to/repo OUTPUT_ROOT=/path/to/new/results
# VENV_PATH=/path/to/venv CINECA_AI_MODULE=cineca-ai/<version>

#SBATCH --account=CASD_prod
#SBATCH --job-name=cifar_diagonal
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
if [[ ! -f "${PROJECT_DIR}/experiments/configs/cifar_diagonal_validation.py" ]]; then
    echo "PROJECT_DIR does not contain the CIFAR diagonal experiment: ${PROJECT_DIR}" >&2
    exit 2
fi
cd "${PROJECT_DIR}"
case "${SLURM_ARRAY_TASK_ID:-}" in
    "") CORRECTION_MODES=(off on) ;;
    0) CORRECTION_MODES=(off) ;;
    1) CORRECTION_MODES=(on) ;;
    *) echo "Array task must be 0 (off) or 1 (on)." >&2; exit 2 ;;
esac

# OUTPUT_ROOT is a parent override; each variant always gets its own directory.
OUTPUT_BASE="${OUTPUT_ROOT:-stat_tests/CIFAR}"
for STOCHASTIC_CORRECTION in "${CORRECTION_MODES[@]}"; do
    OUTPUT_ROOT="${OUTPUT_BASE}/diagonal_stochastic_${STOCHASTIC_CORRECTION}_job_${SLURM_JOB_ID:?Run with sbatch}"
    if [[ -e "${OUTPUT_ROOT}" ]]; then
        echo "OUTPUT_ROOT already exists: ${OUTPUT_ROOT}. Choose a new directory." >&2
        exit 2
    fi
done

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

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Source revision: $(git rev-parse HEAD)"
    git status --short
fi

for STOCHASTIC_CORRECTION in "${CORRECTION_MODES[@]}"; do
    OUTPUT_ROOT="${OUTPUT_BASE}/diagonal_stochastic_${STOCHASTIC_CORRECTION}_job_${SLURM_JOB_ID}"
    echo "=== CIFAR-10 old diagonal run; stochastic correction ${STOCHASTIC_CORRECTION} ==="
    echo "Job: ${SLURM_JOB_ID}; host: $(hostname); output: ${OUTPUT_ROOT}"

    # Check this variant's actual diagonal path before training the shadow bank.
    srun "${PYTHON}" -u -m experiments.configs.cifar_diagonal_validation \
        --smoke --device cuda --stochastic-correction "${STOCHASTIC_CORRECTION}"

    srun "${PYTHON}" -u -m experiments.configs.cifar_diagonal_validation \
        --stochastic-correction "${STOCHASTIC_CORRECTION}" --output "${OUTPUT_ROOT}"

    echo "=== Validating and summarizing all repetitions and LiRA cases ==="
    srun "${PYTHON}" -u validation/accuracy_audit/analyze_saved_cifar.py \
        --input "${OUTPUT_ROOT}/CIFAR_random_diag_ggn_mc_stochastic_${STOCHASTIC_CORRECTION}" \
        --output "${OUTPUT_ROOT}/summary"

    echo "=== CIFAR-10 old diagonal run finished; stochastic correction ${STOCHASTIC_CORRECTION} ==="
done
date

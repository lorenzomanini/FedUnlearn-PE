#!/bin/bash
# Follow-up CIFAR controls using the completed diagnosis checkpoint and masks.
# Submit from repository root: sbatch queue_recovery_controls.sh
# Overrides: DIAGNOSIS_DIR=/path/to/diagnosis_job OUTPUT_ROOT=/path/to/new/results
# PROJECT_DIR, VENV_PATH, CINECA_AI_MODULE, RECOVERY_EPOCHS (defaults to source).
# Four full-weight paths: no reset, actual mask, pseudo mask, matched random mask.

#SBATCH --account=CASD_prod
#SBATCH --job-name=cifar_recovery_controls
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
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
    echo "Run these matched controls sequentially; do not submit as an array." >&2
    exit 2
fi
if [[ -z "${DIAGNOSIS_DIR:-}" ]]; then
    if [[ -f "stat_tests/CIFAR/diagnosis_job_59395209/controls/report.json" ]]; then
        DIAGNOSIS_DIR="stat_tests/CIFAR/diagnosis_job_59395209"
    else
        DIAGNOSIS_DIR="stat_tests/CIFAR/diagnosis_job"
    fi
fi
for artifact in controls/report.json controls/split.pt \
    controls/actual_selected_mask.pt controls/pseudo_selected_mask.pt controls/random_selected_mask.pt \
    bootstrap/original_model.pth bootstrap/initial_model.pth bootstrap/suite/lira_shadow_bank.npz; do
    if [[ ! -f "${DIAGNOSIS_DIR}/${artifact}" ]]; then
        echo "Required source artifact not found: ${DIAGNOSIS_DIR}/${artifact}. Set DIAGNOSIS_DIR." >&2
        exit 2
    fi
done
OUTPUT_ROOT="${OUTPUT_ROOT:-stat_tests/CIFAR/recovery_controls_job_${SLURM_JOB_ID:?Run with sbatch}}"
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

echo "=== CIFAR full-weight recovery controls ==="
echo "Job: ${SLURM_JOB_ID}; host: $(hostname); source: ${DIAGNOSIS_DIR}; output: ${OUTPUT_ROOT}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Source revision: $(git rev-parse HEAD)"
    git status --short
fi
RUN_ARGS=(--reuse-diagnosis "${DIAGNOSIS_DIR}" --output "${OUTPUT_ROOT}/controls")
if [[ -n "${RECOVERY_EPOCHS:-}" ]]; then
    RUN_ARGS+=(--epochs "${RECOVERY_EPOCHS}")
fi
srun "${PYTHON}" -u -m experiments.cifar_diagnosis "${RUN_ARGS[@]}"

echo "=== Controls finished: ${OUTPUT_ROOT}/controls/report.json and metrics.csv ==="
date

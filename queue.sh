#!/bin/bash
# CINECA Leonardo Booster: bounded CIFAR10-random core-score run.
# Submit from the repository root (Slurm opens its logs before this script runs):
#   mkdir -p job_logs
#   sbatch queue.sh
# Optional overrides, for example a utility-only run:
#   NUM_TESTS=1 LIRA_SHADOW_MODELS=0 sbatch queue


# Keep the project account configured for this allocation.
#SBATCH --account=CASD_prod
#SBATCH --job-name=spectral_cifar10
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=job_logs/slurm-%x-%j.out
#SBATCH --error=job_logs/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"

# Array indices take precedence only when an array was explicitly submitted.
export NUM_POWER_ITERS="${SLURM_ARRAY_TASK_ID:-${NUM_POWER_ITERS:-3}}"
export NUM_TESTS="${NUM_TESTS:-3}"
export LIRA_SHADOW_MODELS="${LIRA_SHADOW_MODELS:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

module purge
module load profile/deeplrn
module load "${CINECA_AI_MODULE:-cineca-ai}"

# cineca-ai/4.1.1 exposes torchvision from a nonstandard source directory.
# Record its working import root before switching to the personal venv.
CINECA_PYTHON="$(command -v python)"
CINECA_TORCHVISION_ROOT="$("${CINECA_PYTHON}" - <<'PY'
from pathlib import Path
import torchvision

print(Path(torchvision.__file__).resolve().parent.parent)
PY
)"

# Override the module at submission time with CINECA_AI_MODULE=cineca-ai/<version>.
VENV_PATH="${VENV_PATH:-${SLURM_SUBMIT_DIR}/venv}"
PYTHON="${VENV_PATH}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Python environment not found at ${VENV_PATH}. Create it before submitting." >&2
    exit 1
fi

# Activation is required for venvs layered on CINECA's module-provided packages.
source "${VENV_PATH}/bin/activate"
PYTHON="$(command -v python)"
export PYTHONPATH="${CINECA_TORCHVISION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== Job information ==="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Array: ${SLURM_ARRAY_JOB_ID}, task: ${SLURM_ARRAY_TASK_ID}"
fi
echo "NUM_POWER_ITERS: ${NUM_POWER_ITERS}"
echo "NUM_TESTS: ${NUM_TESTS}"
echo "LIRA_SHADOW_MODELS: ${LIRA_SHADOW_MODELS}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
"${PYTHON}" -V
nvidia-smi

"${PYTHON}" - <<'PY'
import os
import sys

# Reject invalid budgets before the smoke check or any model training.
num_tests = int(os.environ["NUM_TESTS"])
power_iters = int(os.environ["NUM_POWER_ITERS"])
shadows = int(os.environ["LIRA_SHADOW_MODELS"])
if num_tests < 1 or power_iters < 0:
    raise SystemExit("NUM_TESTS must be positive and NUM_POWER_ITERS nonnegative.")
if shadows != 0 and (shadows < 4 or shadows % 2):
    raise SystemExit("LIRA_SHADOW_MODELS must be 0 or an even integer of at least 4.")
print("Full model trainings in this suite:", 2 * num_tests + shadows)

try:
    import torch
    import torchvision
    import backpack
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing Python package {error.name!r} in {sys.executable}. "
        f"Install it in the venv before submitting the job."
    ) from error

assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"
print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("BackPACK import: OK")
PY

echo "=== Running spectral GPU smoke test ==="
# With set -e, a failed smoke test stops the job before any training.
srun "${PYTHON}" -u -m experiments.spectral_smoke \
    --device cuda --backend full --image-size 64 \
    --batch-size 128 --rank 10 --hmp-chunk-size 1

echo "=== Smoke test passed; running CIFAR10-random Spectral experiment ==="
srun "${PYTHON}" -u -m experiments.configs.spectral_wip

echo "=== Finished ==="
date

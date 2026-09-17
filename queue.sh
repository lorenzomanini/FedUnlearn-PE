#!/bin/bash
# CINECA Leonardo Booster: CIFAR10-random Spectral sweep (power_iters 2, 3, 4)

# Replace CHANGE_ME with the account shown by: saldo -b
#SBATCH --account=CASD_PROD
#SBATCH --job-name=spectral_cifar10
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --array=2-4
#SBATCH --output=job_logs/slurm-%x-%A_%a.out
#SBATCH --error=job_logs/slurm-%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"

export NUM_POWER_ITERS="${SLURM_ARRAY_TASK_ID:?This job must run as an array}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

module purge
module load profile/deeplrn
module load "${CINECA_AI_MODULE:-cineca-ai}"

# Override the module at submission time with CINECA_AI_MODULE=cineca-ai/<version>.
VENV_PATH="${VENV_PATH:-${SLURM_SUBMIT_DIR}/venv}"
PYTHON="${VENV_PATH}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Python environment not found at ${VENV_PATH}. Create it before submitting." >&2
    exit 1
fi

echo "=== Job information ==="
echo "Host: $(hostname)"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "NUM_POWER_ITERS: ${NUM_POWER_ITERS}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
"${PYTHON}" -V
nvidia-smi

"${PYTHON}" -c 'import torch, backpack; assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"; print("PyTorch:", torch.__version__); print("GPU:", torch.cuda.get_device_name(0)); print("BackPACK import: OK")'

echo "=== Running spectral GPU smoke test ==="
# With set -e, a failed smoke test stops this array task before any training.
srun "${PYTHON}" -u -m experiments.spectral_smoke \
    --device cuda --batch-size 128 --rank 10 --hmp-chunk-size 1

echo "=== Smoke test passed; running CIFAR10-random Spectral experiment ==="
srun "${PYTHON}" -u -m experiments.configs.spectral_wip

echo "=== Finished ==="
date

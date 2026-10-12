#!/bin/bash
# TMLR experiments for the curvature core score on CINECA Leonardo Booster.
# Each array task runs one suite of experiments/configs/tmlr_paper.py; list them with
#   python -m experiments.configs.tmlr_paper --list
#    0-5   CIFAR-10 heterogeneity sweep (preferential strength 1 to 0) and Dirichlet(0.8)
#    6-8   FashionMNIST sweep (strength 1, 0.5, 0)
#    9-12  diagonal TIS on matched splits and seeds
#   13-19  estimator sensitivity: rank, solver iterations, curvature samples
#   20     recovery budget
#   21-23  core-optimal (generalized-eigenvector) subspace, strength 1, 0.5, 0
#   24-26  its damping (1e-6, 1e-2) and rank-40 ablations on the IID split
#   27-28  top-curvature subspace on the same Gauss-Newton curvature, strength 1 and 0
# Compute nodes are offline: once, on a login node, run
#   python -m experiments.configs.tmlr_paper --download
# Commit the code first so the recorded revision identifies it. Then, from the
# repository root: sbatch queue.sh (all suites) or e.g. sbatch --array=0-5 queue.sh.
# Each suite trains 20 original/gold pairs and its own 64-shadow LiRA bank, about
# 1.5 days on one A100; generalized suites add up to 100 LOBPCG iterations per model,
# about 2 days. --time is the boost_qos_lprod maximum (4 days); lower it with
# sbatch --time=... if your QoS differs. %4 caps concurrent suites.
# Overrides: PROJECT_DIR=/path/to/repo OUTPUT_ROOT=/parent/of/results
# VENV_PATH=/path/to/venv CINECA_AI_MODULE=cineca-ai/<version>

#SBATCH --account=CASD_prod
#SBATCH --job-name=tmlr_core
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-28%4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
if [[ ! -f "${PROJECT_DIR}/experiments/configs/tmlr_paper.py" ]]; then
    echo "PROJECT_DIR does not contain the TMLR experiment suites: ${PROJECT_DIR}" >&2
    exit 2
fi
cd "${PROJECT_DIR}"
SUITE_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as an array job; the header sets --array}"

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

SUITE="$("${PYTHON}" -m experiments.configs.tmlr_paper --name "${SUITE_INDEX}")"
OUTPUT_DIR="${OUTPUT_ROOT:-stat_tests/TMLR}/${SUITE}_job_${SLURM_ARRAY_JOB_ID}"
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output directory already exists: ${OUTPUT_DIR}. Choose a new OUTPUT_ROOT." >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}"

echo "=== TMLR suite ${SUITE_INDEX}: ${SUITE} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SUITE_INDEX}; host: $(hostname); output: ${OUTPUT_DIR}"
"${PYTHON}" -m experiments.configs.tmlr_paper --suite "${SUITE}" --describe > "${OUTPUT_DIR}/suite.json"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "${OUTPUT_DIR}/source_revision.txt"
    git status --short > "${OUTPUT_DIR}/source_status.txt"
    git diff > "${OUTPUT_DIR}/source.diff"
    echo "Source revision: $(cat "${OUTPUT_DIR}/source_revision.txt")"
    cat "${OUTPUT_DIR}/source_status.txt"
fi

# Check the dataset and this suite's scoring path on the GPU before the shadow bank.
srun "${PYTHON}" -u -m experiments.configs.tmlr_paper --suite "${SUITE}" --smoke --device cuda

srun "${PYTHON}" -u -m experiments.configs.tmlr_paper --suite "${SUITE}" --output "${OUTPUT_DIR}"

echo "=== Validating and summarizing all repetitions and LiRA cases ==="
# Deletion cost is reported, not enforced: the eigenbasis is target-independent and
# its time is recorded separately (subspace_seconds in cifar_score_diagnostics.csv).
srun "${PYTHON}" -u validation/accuracy_audit/analyze_saved_cifar.py \
    --input "${OUTPUT_DIR}/${SUITE}" --output "${OUTPUT_DIR}/summary" --allow-inefficient

echo "=== TMLR suite ${SUITE} finished ==="
date

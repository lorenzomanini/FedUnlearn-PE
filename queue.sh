#!/bin/bash
# CINECA Leonardo Booster: bounded CIFAR accuracy ablations.
# Submit from the repository root. With no saved CIFAR artifacts, this job
# trains one original and one gold model on a fresh, saved split, then reuses
# the original checkpoint for every ablation. LiRA is deferred in fresh mode.
#
#   sbatch queue.sh
# If a matching original checkpoint and its saved suite are available, skip setup:
#   SUITE=/path/to/CIFAR CHECKPOINT=/path/to/original_model.pth sbatch queue.sh
#
# Default cases isolate reset values, then recovery duration, at 44.44% score
# mass. Optional cases isolate BatchNorm mode, larger curvature samples, or
# more subspace iterations (one factor at a time):
#   ABLATION_CASES=initial_5_frozen_bn,initial_5_samples2048 sbatch queue.sh
#
# Other overrides: PROJECT_DIR=/path/to/repo SUITE=/path/to/CIFAR
# SOURCE_MODE=auto|saved|fresh
# REPETITION=0 SCORE_MASS=44.444444
# OUTPUT_ROOT=/path/to/new/results VENV_PATH=/path/to/venv BUDGET_FRACTION=0.25

#SBATCH --account=CASD_prod
#SBATCH --job-name=cifar_ablation
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}}"
if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Project directory does not exist: ${PROJECT_DIR}" >&2
    exit 2
fi
cd "${PROJECT_DIR}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This job runs all selected cases sequentially; do not submit it as an array." >&2
    exit 2
fi

SUITE="${SUITE:-stat_tests/CIFAR}"
SOURCE_MODE="${SOURCE_MODE:-auto}"
REPETITION="${REPETITION:-0}"
SCORE_MASS="${SCORE_MASS:-44.44444444444444}"
BUDGET_FRACTION="${BUDGET_FRACTION:-0.25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SUITE}/ablations_job_${SLURM_JOB_ID}}"
ABLATION_CASES="${ABLATION_CASES:-zero_1,initial_1,initial_5}"

if [[ -n "${CHECKPOINT:-}" && ! -f "${CHECKPOINT}" ]]; then
    echo "CHECKPOINT does not exist: ${CHECKPOINT}" >&2
    exit 2
fi
case "${SOURCE_MODE}" in
    auto)
        if [[ -f "${SUITE}/init_params.pkl" ]]; then
            SOURCE_MODE=saved
        else
            SOURCE_MODE=fresh
        fi
        ;;
    saved|fresh) ;;
    *) echo "SOURCE_MODE must be auto, saved or fresh." >&2; exit 2 ;;
esac
if [[ "${SOURCE_MODE}" == saved ]]; then
    for required in init_params.pkl labels.pkl "test_${REPETITION}/score_diagnostics.pkl" "test_${REPETITION}/stage_timings.pkl" "test_${REPETITION}/initial_eval_train_results.pkl" "test_${REPETITION}/initial_eval_test_results.pkl"; do
        if [[ ! -f "${SUITE}/${required}" ]]; then
            echo "Incomplete saved CIFAR suite: ${SUITE}/${required}" >&2
            echo "Use SOURCE_MODE=fresh to start a self-contained new experiment." >&2
            exit 2
        fi
    done
elif [[ -n "${CHECKPOINT:-}" ]]; then
    echo "A checkpoint requires its matching saved suite; set SUITE or use fresh mode without CHECKPOINT." >&2
    exit 2
elif [[ "${REPETITION}" != 0 ]]; then
    echo "Fresh mode creates repetition 0; set REPETITION=0." >&2
    exit 2
fi
if [[ ! "${REPETITION}" =~ ^[0-9]+$ ]]; then
    echo "REPETITION must be a nonnegative integer." >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "OUTPUT_ROOT already exists: ${OUTPUT_ROOT}. Choose a new directory." >&2
    exit 2
fi

IFS=',' read -r -a CASES <<< "${ABLATION_CASES}"
if (( ${#CASES[@]} == 0 )); then
    echo "ABLATION_CASES must contain at least one case." >&2
    exit 2
fi
seen_cases=,
for test_case in "${CASES[@]}"; do
    case "${test_case}" in
        zero_1|initial_1|initial_5|initial_5_frozen_bn|initial_5_samples2048|initial_5_iters5) ;;
        *) echo "Unknown ablation case: ${test_case}" >&2; exit 2 ;;
    esac
    if [[ "${seen_cases}" == *",${test_case},"* ]]; then
        echo "Duplicate ablation case: ${test_case}" >&2
        exit 2
    fi
    seen_cases+="${test_case},"
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

module purge
module load profile/deeplrn
module load "${CINECA_AI_MODULE:-cineca-ai}"

# cineca-ai exposes torchvision from a nonstandard source directory. Record
# that path before activating the project's environment, as in the prior job.
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
assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("BackPACK import: OK")
PY

echo "=== CIFAR recovery ablations ==="
echo "Job: ${SLURM_JOB_ID}; host: $(hostname); repetition: ${REPETITION}"
echo "Source mode: ${SOURCE_MODE}; source suite: ${SUITE}; output: ${OUTPUT_ROOT}"
echo "Cases: ${ABLATION_CASES}; score mass: ${SCORE_MASS}%"
echo "Per-case deletion budget: ${BUDGET_FRACTION} of saved gold retraining time"
mkdir -p "${OUTPUT_ROOT}"

if [[ -z "${CHECKPOINT:-}" ]]; then
    if [[ "${SOURCE_MODE}" == fresh ]]; then
        echo "=== Training one original and one gold model on a new CIFAR split ==="
        srun "${PYTHON}" -u -m experiments.cifar_bootstrap \
            --output "${OUTPUT_ROOT}/bootstrap" \
            --batch-size 128 --train-epochs 40
    else
        echo "=== Training one fresh original model on the saved split ==="
        srun "${PYTHON}" -u -m experiments.cifar_bootstrap \
            --source-suite "${SUITE}" \
            --repetition "${REPETITION}" \
            --output "${OUTPUT_ROOT}/bootstrap" \
            --batch-size 128
    fi
    CHECKPOINT="${OUTPUT_ROOT}/bootstrap/original_model.pth"
    SUITE="${OUTPUT_ROOT}/bootstrap/suite"
fi
echo "Ablation suite: ${SUITE}; checkpoint: ${CHECKPOINT}"

# Every case uses the same original checkpoint, saved partition, score seed and
# data order. The Python command checks checkpoint predictions before scoring.
for test_case in "${CASES[@]}"; do
    strategy=initial
    epochs=5
    batchnorm_mode=train
    samples=512
    power_iters=3
    case "${test_case}" in
        zero_1) strategy=zero; epochs=1 ;;
        initial_1) epochs=1 ;;
        initial_5) ;;
        initial_5_frozen_bn) batchnorm_mode=frozen ;;
        initial_5_samples2048) samples=2048 ;;
        initial_5_iters5) power_iters=5 ;;
    esac

    echo "=== Running ${test_case} ==="
    srun "${PYTHON}" -u -m experiments.cifar_recovery_ablation \
        --suite "${SUITE}" \
        --repetition "${REPETITION}" \
        --checkpoint "${CHECKPOINT}" \
        --output "${OUTPUT_ROOT}/${test_case}" \
        --reset-strategy "${strategy}" \
        --batchnorm-mode "${batchnorm_mode}" \
        --score-mass "${SCORE_MASS}" \
        --epochs "${epochs}" \
        --max-samples "${samples}" \
        --target-max-samples "${samples}" \
        --power-iters "${power_iters}" \
        --rank 10 --batch-size 128 \
        --budget-fraction "${BUDGET_FRACTION}"
done

echo "=== All requested ablations finished ==="
date

#!/bin/bash
# CINECA Leonardo Booster: CIFAR-100 within-superclass deletion (forget apple).
# Download once: python -m experiments.configs.cifar100_within_class --download
# Submit from the repository root: sbatch queue.sh
# KEEP_CHECKPOINTS=1 sbatch queue.sh keeps every *.pt (audit and compaction still run).
# KEEP_SCORES=1 also keeps full per-coordinate scores as scores.npz (~165 MB per seed).
#
# Array map (Slurm starts tasks in index order):
#   0-2    pilot seeds 3000-3002: controls on the existing job-38148587 checkpoints
#          (c5 c1 c2 diagonal c4 at ranks 20/40/80), then audit, compact, delete *.pt.
#          Completed in job 64858158; the pilot checkpoints are deleted, so the default
#          array below starts at 3.
#   3-14   seeds 3003-3014, standard recipe: train; core, orange-swap, cross-superclass
#          swap, leverage and random arms; controls c5 c1 c2 diagonal c4 (ranks 40/80);
#          audit, compact, delete *.pt.
#   15-17  seeds 3003-3005, regularized recipe (augmentation + weight decay 5e-4):
#          train; main arms; controls c5 c2; audit, compact, delete *.pt.
# Rough GPU time per task: pilot 2 h on A100; standard 6-9 h; regularized 3-4 h.
#
# PRE-SPECIFIED PRIMARY CONTRAST, fixed 2026-11-13; amended 2026-11-20 (seeds 3003-3009
# -> 3003-3014, nothing else) before any confirmatory seed was trained. Source of
# truth: PRIMARY_CONTRAST in experiments/configs/cifar100_within_class.py.
#   endpoint  apple accuracy closure (NFS_TargetAcc, paper Eq. 55)
#   contrast  core at 10% minus layer-matched random interpolated, within each seed,
#             to core's retained test accuracy (random at 8/10/15/20/30/40%;
#             no extrapolation: unbracketed seeds are excluded and reported)
#   test      two-sided one-sample t-test over seeds 3003-3014 (standard recipe),
#             alpha 0.05; exact sign-flip test as a robustness check.
#   Pilot seeds 3000-3002, the regularized runs and all other contrasts are descriptive.

#SBATCH --account=CASD_prod
#SBATCH --job-name=c100_within
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --array=3-17%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PILOT_SEEDS=(3000 3001 3002)
NEW_SEEDS=(3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3013 3014)
REGULARIZED_SEEDS=(3003 3004 3005)
PILOT_JOB="${PILOT_JOB:-38148587}"
OUTPUT_ROOT="${OUTPUT_ROOT:-stat_tests/CIFAR100_within_class}"
REGULARIZED_ROOT="${REGULARIZED_ROOT:-stat_tests/CIFAR100_within_class_regularized}"
MODULE="experiments.configs.cifar100_within_class"
TASK="${SLURM_ARRAY_TASK_ID:?Submit as an array job; the header sets --array}"
if [[ ! "${TASK}" =~ ^([0-9]|1[0-7])$ ]]; then
    echo "Array index ${TASK} must be in 0-17." >&2
    exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "${PROJECT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
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
PYTHON="${VENV_PATH}/bin/python"
export PYTHONPATH="${CINECA_TORCHVISION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" - <<'PY'
import torch
import torchvision
import backpack
import sklearn
from experiments.configs.cifar100_within_class import (
    CONFIRMATORY_SEEDS, PILOT_SEEDS, PRIMARY_CONTRAST, REGULARIZED_SEEDS)
assert torch.cuda.is_available(), 'PyTorch cannot access the allocated GPU'
assert PILOT_SEEDS == (3000, 3001, 3002) and CONFIRMATORY_SEEDS == tuple(range(3003, 3015))
assert REGULARIZED_SEEDS == (3003, 3004, 3005)
assert PRIMARY_CONTRAST['seeds'] == list(CONFIRMATORY_SEEDS), 'pre-specified seeds changed'
PY

snapshot_source() {  # Record exactly the code this task runs, untracked files included.
    local dir="$1"
    git rev-parse HEAD > "${dir}/source_revision.txt"
    git status --short > "${dir}/source_status.txt"
    local files
    files="$({ git ls-files --cached --others --exclude-standard -- experiments fisherunlearn analysis queue.sh \
               | grep -E '\.(py|sh)$' || true; } | while read -r file; do
                   if [[ -e "${file}" ]]; then echo "${file}"; fi  # skip tracked files deleted locally
               done)"
    {
        git diff HEAD
        { git ls-files --others --exclude-standard -- experiments fisherunlearn analysis \
            | grep -E '\.py$' || true; } | while read -r file; do
            git diff --no-index -- /dev/null "${file}" || true
        done
    } > "${dir}/source.diff"
    printf '%s\n' "${files}" | tar -czf "${dir}/source.tar.gz" -T -
}

finalize() {  # Audit and compact; delete *.pt only after every check passed.
    local flags=(--delete-checkpoints)
    if [[ "${KEEP_CHECKPOINTS:-0}" == 1 ]]; then
        flags=()
    fi
    if [[ "${KEEP_SCORES:-0}" == 1 ]]; then  # full float32 score vectors, ~165 MB per seed
        flags+=(--keep-scores)
    fi
    srun "${PYTHON}" -u -m experiments.within_class_finalize --experiment "$1" --controls "$2" \
        --device cuda ${flags[@]+"${flags[@]}"}
}

pilot_controls() {
    local seed="$1"
    local job_dir="${OUTPUT_ROOT}/seed_${seed}_job_${PILOT_JOB}"
    local experiment="${job_dir}/experiment"
    for file in original.pt initial.pt scores.pt core_10_mask.pt results.json; do
        if [[ ! -f "${experiment}/${file}" ]]; then
            echo "Pilot artifact missing: ${experiment}/${file}" >&2
            exit 2
        fi
    done
    local controls_job="${job_dir}/controls_job_${SLURM_ARRAY_JOB_ID}"
    if [[ -e "${controls_job}" ]]; then
        echo "Output directory already exists: ${controls_job}." >&2
        exit 2
    fi
    mkdir -p "${controls_job}"
    snapshot_source "${controls_job}"
    srun "${PYTHON}" -u -m "${MODULE}" --seed "${seed}" --preflight
    # Rank 20 is recomputed for the cross-superclass control at the source rank.
    srun "${PYTHON}" -u -m experiments.within_class_controls --source "${experiment}" \
        --output "${controls_job}/controls" --stages c5 c1 c2 diagonal c4 --ranks 20 40 80 --device cuda
    finalize "${experiment}" "${controls_job}/controls"
}

new_seed() {
    local variant="$1" seed="$2" root="$3"
    shift 3
    local job_dir="${root}/seed_${seed}_job_${SLURM_ARRAY_JOB_ID}"
    if [[ -e "${job_dir}" ]]; then
        echo "Output directory already exists: ${job_dir}. Choose a new output root." >&2
        exit 2
    fi
    mkdir -p "${job_dir}"
    snapshot_source "${job_dir}"
    "${PYTHON}" -m "${MODULE}" --seed "${seed}" --variant "${variant}" --save-models --describe \
        > "${job_dir}/protocol.json"
    # Compute nodes are offline; check all cached data before any long training.
    srun "${PYTHON}" -u -m "${MODULE}" --seed "${seed}" --preflight
    srun "${PYTHON}" -u -m "${MODULE}" --seed "${seed}" --smoke --device cuda
    srun "${PYTHON}" -u -m "${MODULE}" --seed "${seed}" --variant "${variant}" --save-models \
        --device cuda --output "${job_dir}/experiment"
    srun "${PYTHON}" -u -m experiments.within_class_controls --source "${job_dir}/experiment" \
        --output "${job_dir}/controls" "$@" --device cuda
    finalize "${job_dir}/experiment" "${job_dir}/controls"
}

if (( TASK <= 2 )); then
    pilot_controls "${PILOT_SEEDS[TASK]}"
elif (( TASK <= 14 )); then
    new_seed standard "${NEW_SEEDS[TASK - 3]}" "${OUTPUT_ROOT}" \
        --stages c5 c1 c2 diagonal c4 --ranks 40 80
else
    new_seed regularized "${REGULARIZED_SEEDS[TASK - 15]}" "${REGULARIZED_ROOT}" --stages c5 c2
fi

# FedUnlearn-PE   

This repository contains the official implementation of the methods proposed in the paper:  

> # Tackling Federated Unlearning as a Parameter Estimation Problem
> Antonio Balordi, Lorenzo Manini, Fabio Stella, Alessio Merlo
> ## Abstract
> Privacy regulations require the erasure of data from deep learning models. This is a significant challenge that is amplified in Federated Learning, where data remains on clients, making full retraining or coordinated updates often infeasible. This work introduces an efficient Federated Unlearning framework based on information theory, modeling leakage as a parameter estimation problem. Our method uses second-order Hessian information to identify and selectively reset only the parameters most sensitive to the data being forgotten, followed by minimal federated retraining. This model-agnostic approach supports categorical and client unlearning without requiring server access to raw client data after initial information aggregation. Evaluations on benchmark datasets demonstrate strong privacy (MIA success near random, categorical knowledge erased) and high performance (Normalized Accuracy against re-trained benchmarks of $\approx$ 0.9), while aiming for increased efficiency over complete retraining. Furthermore, in a targeted backdoor attack scenario, our framework effectively neutralizes the malicious trigger, restoring model integrity. This offers a practical solution for data forgetting in FL.

The preprint is available on [Arxiv](https://doi.org/10.48550/arXiv.2508.19065).

## Implementation layout

The repository intentionally retains three behaviorally distinct experiment generations:

- `tester.py`: legacy diagonal workflow;
- `new_tester.py`: revised diagonal workflow;
- `new_new_tester.py`: spectral work-in-progress workflow.

These root modules remain compatibility launchers. Their implementations now live under `experiments/`; reusable scientific code lives under `fisherunlearn/`; and artifact readers, metrics, and plots live under `analysis/`. The spectral implementation remains explicitly WIP and is not the diagonal default.

The spectral score now follows the noncommuting core allocation in `core_score_framework_didactic.pdf`, equations (55)–(58). Runtime changes and local validation evidence are recorded in [RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md).

## Online LiRA privacy evaluation

The revised diagonal and spectral runners implement the online likelihood-ratio
attack from Carlini et al. A reusable bank of shadow models is trained once per
experiment suite. For every audit record, the saved membership mask identifies
the shadows trained with that record (shadow-IN) and without it (shadow-OUT).
The rest of each shadow's training set is sampled independently from the common
data pool, and every shadow uses the same number of records as the attacked
model. The attack fits per-record Gaussian means to the stable logit confidence
and computes `log p(score | IN) - log p(score | OUT)` for audited trained,
unlearned, and gold-retrained models.

LiRA runs only when a case explicitly includes `"LiRA"` in `tests`.
Set `num_shadow_models` (default `8`, use `0` to disable), `lira_seed`, and
`lira_global_variance` in the initial experiment configuration. The global
variance option pools within-record residuals separately for IN and OUT; the
per-record means are never pooled. The suite-level `lira_shadow_bank.npz`
contains the shadow scores and explicit membership mask, while each `test_<n>`
directory contains candidate scores only for its audited cases.
`lira_case_indices.pkl` maps their rows to the saved utility-result rows. Cases
without LiRA still retain all utility results; the analysis handles these partial
audits and suites with LiRA disabled. More shadows provide more IN and OUT
observations per record at a higher training cost.

The result key `shadow_out` is retained for artifact compatibility and denotes
the gold-standard model retrained without the entire target client. It is not
used to construct the record-level shadow-OUT distributions. Analyze a suite
with:

```bash
python -m analysis.experiments_results path/to/suite
```

The command reads the configured repetitions and plots test accuracy and online
LiRA results. To write CSV summaries, run
`python validation/accuracy_audit/analyze_saved_cifar.py --input path/to/suite --output path/to/summary`.
For a `preferential_class` suite, `cifar_per_class_{per_run,summary}.csv` also
reports test accuracy for every class, plus forget accuracy and class-restricted
LiRA AUC/TPR for classes represented by the target client. This helps identify
class-distribution effects in pooled privacy scores.

The faithful shadow-bank path currently supports the centralized `sgd` runner.
The legacy runner now rejects its former single-benchmark approximation when a
test requests `LiRA`, rather than reporting it as the paper's attack.

## Bounded spectral unlearning

The spectral runner uses full Hessian-vector products and retains every entry of
`F = U.T @ H_target @ U`. It computes `B = Lambda**(-1/2) @ F @ Lambda**(-1/2)`
and the squared row norms of `U @ B`, without constructing a full Hessian.
Only positive Ritz values above the configured cutoff are inverted. Selection
resets the smallest descending prefix reaching the requested score mass in each
module layer (including its weight and bias); zero-mass layers select nothing.
The random baseline samples the same number of scalar coordinates in each layer,
and both methods restore selected values from the original model's pretraining
state. This score is a selection
statistic, not a certificate of post-reset privacy.

The default uses a fixed random sample of up to 512 actual training records and
512 target records, rank 10, and at most `NUM_POWER_ITERS + 1` full-curvature
passes plus one target-curvature pass. The sample is reused on every pass, and
held-out validation data is excluded. The score is computed once per trained
model and reused throughout its percentage sweep. Full-data curvature is still
available by setting both sample limits to `0` or `None`.

These are explicit empirical-curvature and rank approximations. Compare sample
size, rank, cutoff and forgetting/utility outcomes before drawing conclusions. The
existing diagonal runners remain separate historical variants; their optional
stochastic correction is not part of the document's core score.

Relevant initial configuration keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `spectral_max_samples` | 512 | Full-training curvature sample cap |
| `spectral_target_max_samples` | 512 | Target curvature sample cap |
| `spectral_rank` | 10 | Requested subspace dimension |
| `spectral_num_power_iters` | `NUM_POWER_ITERS`, otherwise 5 | Maximum subspace updates |
| `spectral_seed` | 0 | Fixed sample and subspace seed |
| `spectral_eigenvalue_min` | 1e-6 | Absolute positive-curvature cutoff |
| `spectral_eigenvalue_rtol` | 1e-5 | Cutoff relative to the largest positive Ritz value |
| `spectral_power_tolerance` | 1e-3 | Relative eigenpair residual stopping tolerance |
| `spectral_curvature_backend` | `full` | `block` explicitly selects the BackPACK block approximation |
| `spectral_hmp_chunk_size` | 1 | Simultaneous product directions |

Each repetition saves `score_diagnostics.pkl` with the sample indices, retained
eigenvalues, projected target curvature, sum-rule error, eigenpair residuals and
actual operator-pass count. `stage_timings.pkl` separates original training,
gold retraining, score construction and initial evaluation. Per-case timings
separate selection/recovery from the random baseline and privacy/utility
inference; `unlearning_with_score_seconds` charges the shared score once to an
individual deletion. The suite timing file records shadow-bank and total time.

The `spectral_wip` launch configuration defaults to **3 repetitions** and performs
LiRA at **3 of the 10 sweep points** (0%, 55.56%, 100% score mass). The training
epochs and shadow training-set sizes are unchanged. Override these budgets:

```bash
NUM_TESTS=3 LIRA_SHADOW_MODELS=8 NUM_POWER_ITERS=3 python -m experiments.configs.spectral_wip
# Utility-only development run:
NUM_TESTS=1 LIRA_SHADOW_MODELS=0 NUM_POWER_ITERS=2 python -m experiments.configs.spectral_wip
```

## TMLR experiment suites

`queue.sh` runs the experiments for the core-score paper as a Slurm array, one
suite of `experiments/configs/tmlr_paper.py` per task (`python -m
experiments.configs.tmlr_paper --list` prints the indices). Every suite uses five
clients, 20 original/gold pairs, a 64-shadow online-LiRA bank and the same seeds,
so core-score and diagonal suites share splits:

- CIFAR-10 `preferential_mix` sweep, strength 1 (preferential) to 0 (stratified
  IID), plus Dirichlet(0.8); FashionMNIST at strengths 1, 0.5 and 0. Strength is the
  target's share of its exclusive class (20% to 100%); unlike the Dirichlet
  partitioner, every strength keeps all training records;
- the diagonal TIS on matched splits, scored on trained records only
  (`diagonal_population: training`) without stochastic correction;
- rank, solver-iteration and curvature-sample sensitivity, and recovery budget;
- the core-optimal subspace (`spectral_subspace: generalized`) at strengths 1, 0.5
  and 0, its damping and rank ablations on the IID split, and top-curvature scores
  on the same Gauss-Newton curvature (`spectral_curvature_backend: ggn`).

For a fixed rank, the core `Tr(B^2)` is maximized by the top generalized
eigenvectors of `(H_T, H)`, the directions with the largest relative curvature
`mu = v^T H_T v / v^T H v`, not by the top eigenvectors of `H`. The generalized
option estimates `H = (1 - a) H_R + a H_T` from a retained sample plus the target
sample itself (`a` is the target's training fraction), so with Gauss-Newton
curvature `0 <= mu <= 1/a` and `a * mu` is the target's curvature share. Block
LOBPCG solves the pencil `(H_T, H + delta I)` with curvature products only, for at
most `spectral_num_power_iters` iterations. In near-interpolating networks many
directions sit at the bound; `delta = spectral_damping * lambda_max(H)` acts as a
prior precision that ranks them by curvature. The control sample is drawn outside
`H`'s retained sample. `cifar_score_diagnostics.csv` reports the largest eigenvalue
of `B` against `relative_curvature_bound`.

Main suites sweep 14 score masses under both reset values (`initial`, `zero`, always
the same for the score and random arms), add full-weight recovery
(`recovery_scope: all`) at six masses, and a no-reset fine-tuning baseline. Core
suites also score an equal-size sample of retained records in the same basis; the
log-score correlation and top-decile overlap with the target score measure how
target-specific the ranking is. Checkpoints of the initialization, original and
gold models are saved (`save_models: baseline`).

Download the datasets once on a login node with `python -m
experiments.configs.tmlr_paper --download`, commit the code, then submit
`sbatch queue.sh` (or e.g. `sbatch --array=0-5 queue.sh`). Results go to
`stat_tests/TMLR/<suite>_job_<jobid>`, with `suite.json`, the source revision and
diff, and CSV summaries in `summary/`. `cifar_score_diagnostics.csv` reports the
largest eigenvalue of B, the control agreement and the eigensolver time separately
from the per-request time, because the eigenbasis does not depend on the target.

The earlier diagonal CIFAR-10 random-client sweeps (stochastic correction off and
on, 10 clients, all client records) remain available as
`sbatch queue_diagonal_validation.sh`; their results are under
`stat_tests/CIFAR/diagonal_stochastic_<off|on>_job_<jobid>`.

The spectral benchmark remains available through
`python -m experiments.configs.cifar_full_validation --output <new-directory>`.
It uses rank 20 and fixed 2,048-record full and target curvature samples.

## Checks before a long run

### Recreate the CREATE environment

Create the environment from an interactive GPU allocation so package
installation does not consume login-node resources. From the repository root:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  --time=01:00:00 --pty /bin/bash -l
module purge
module load python/3.11.6-gcc-13.2.0
module load cuda
virtualenv venv -p "$(which python3)"
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  -r requirements-create-torch.txt
python -m pip install -r requirements-create.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python -m unittest discover -s tests -v
exit
sbatch queue.sh
```

The checked-in versions match the environment used for the local regression
and runtime validation. If the environment is created somewhere other than
`venv` in the repository root, submit with
`sbatch --export=ALL,VENV_PATH=/absolute/path/to/venv queue.sh`.

Within an allocated GPU job, using the same environment as `queue.sh`:

```bash
srun venv/bin/python -m experiments.spectral_smoke --device cuda --backend full --image-size 64 --batch-size 128 --rank 20
```

This checks one ResNet18 full-Hessian matrix product at the same 64×64 image size
as the CIFAR training transform. It does not train a LiRA bank. Use `--backend block`
only to check the optional block approximation. A small local check is:

```bash
python -m experiments.spectral_smoke --device cpu --batch-size 2 --rank 2
python -m unittest discover -s tests -v
```

The regression suite includes exact dense-Hessian comparisons, noncommuting
core scores and basis invariance, selection boundaries/ties, frozen-weight
gradients, real tiny shadow training, and complete small experiments with
recovery, saved artifacts and sparse/no-LiRA analysis. The optional poisoning
experiments additionally require `adversarial-robustness-toolbox`.

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

Phase-2 refactoring decisions, the reproducibility baseline, and validation evidence are recorded in [`docs/refactoring/`](docs/refactoring/). The executable code—not either manuscript—is the behavior oracle for this structural phase.

## Online LiRA privacy evaluation

The revised diagonal and spectral runners implement the online likelihood-ratio
attack from Carlini et al. A reusable bank of shadow models is trained once per
experiment suite. For every audit record, the saved membership mask identifies
the shadows trained with that record (shadow-IN) and without it (shadow-OUT).
The rest of each shadow's training set is sampled independently from the common
data pool, and every shadow uses the same number of records as the attacked
model. The attack fits per-record Gaussian means to the stable logit confidence
and computes `log p(score | IN) - log p(score | OUT)` for every trained,
unlearned, and gold-retrained model.

Set `num_shadow_models` (default `64`, use `0` to disable), `lira_seed`, and
`lira_global_variance` in the initial experiment configuration. The global
variance option pools within-record residuals separately for IN and OUT; the
per-record means are never pooled. The suite-level `lira_shadow_bank.npz`
contains the shadow scores and explicit membership mask, while each `test_<n>`
directory contains the candidate scores for the evaluated models.

The result key `shadow_out` is retained for artifact compatibility and denotes
the gold-standard model retrained without the entire target client. It is not
used to construct the record-level shadow-OUT distributions. Analyze a suite
with:

```powershell
.\.venv\Scripts\python.exe -m analysis.experiments_results <suite-directory>
```

The faithful shadow-bank path currently supports the centralized `sgd` runner.
The legacy runner now rejects its former single-benchmark approximation when a
test requests `LiRA`, rather than reporting it as the paper's attack.

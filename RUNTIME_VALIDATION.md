# Runtime and core-score validation

Validated locally on 2026-09-21 against the previous code at `939ab74`.
Environment: macOS arm64, Python 3.12, PyTorch 2.14.0, torchvision 0.29.0,
BackPACK 1.7.1. All measurements below are CPU measurements, not cluster/GPU
forecasts. [Raw measurements](validation/runtime_2026-09-21.json) are included.

## Main changes

- The refactor introduced a default bank of 64 full LiRA shadow trainings even
  when no case requested LiRA. The runner now starts a bank only for explicit
  audits, defaults to 8 shadows, and reuses it across repetitions. Epoch counts
  and shadow training-set sizes are unchanged.
- The shipped spectral sweep uses 3 repetitions instead of 20, with LiRA only at
  the first, middle and last score-mass settings. These are exploratory defaults;
  reduced repetitions/shadows reduce statistical precision. `NUM_TESTS` and
  `LIRA_SHADOW_MODELS` restore larger budgets. `LIRA_SHADOW_MODELS=0` disables LiRA.
- The old spectral estimator repeatedly applied per-parameter BackPACK block
  products across the complete dataset, then discarded off-diagonal target
  correlations. The default now uses full autograd Hessian-vector products,
  preserves the full projected target matrix, and caps full/target curvature
  samples at 512 each. It saves the sampled indices, numerical cutoffs, actual
  pass counts, residuals and sum-rule diagnostics. Curvature samples remain fixed
  across passes and exclude validation examples.
- The core allocation follows document equations (55)–(58):
  `F = U.T H_target U`, `B = Lambda^(-1/2) F Lambda^(-1/2)`, and
  `score_i = sum_k (U B)_ik^2`. Positive-support filtering is explicit. The
  common `2 alpha_target^2` prefactor is recorded separately and omitted from
  within-request rankings. The sum of scores matches `||B||_F^2` numerically.
- Selection now takes the smallest prefix reaching the requested score mass,
  including ties correctly. Zero-score groups and a 0% request select nothing.
  Recovery uses indexed scatter instead of rebuilding sparse masks; frozen
  weights, selected gradients, original parameter names and model modes are
  tested. Duplicate zero-reset evaluations are reused.
- Saved timings separate the deletion operation (score, selection, recovery)
  from initial training, full gold retraining, random baselines and LiRA. Sparse
  privacy results carry explicit case indices, and analysis handles disabled
  LiRA without inventing privacy values.

## Cost counts for the active CIFAR configuration

These are counts per suite/array task, not measured hour estimates.

| Work | Previous default | New default |
| --- | ---: | ---: |
| LiRA shadow trainings | 64 | 8 |
| Original + gold model trainings | 40 (20 repetitions) | 6 (3 repetitions) |
| Total full model trainings | 104 | 14 |
| LiRA cases per repetition | 10 | 3 |
| Curvature full-data sample | 50,000 | up to 512 actual training records |
| Curvature target sample | 5,000 | up to 512 |

With 3 power updates and rank 10, the previous four full passes plus one target
pass visited 205,000 records; the capped path visits at most 2,560, before
multiplying by rank. This is about 80× fewer record visits, independent of the
operator implementation improvement. It is an approximation and does not imply
identical scores. The existing three-task array still runs three such suites.

## Numerical and integration checks

Command: `python -m unittest discover -s tests -v`.

**37 tests ran: 36 passed and 1 CUDA-only check was skipped.** Coverage includes:

- Full products against an explicit dense Hessian, including cross-parameter
  terms, uneven batches, repeated calls and direction chunks of 1/2/3.
- Noncommuting target curvature, the diagonal limiting case, invariance under
  orthogonal basis rotations, positive-support thresholds, the sum rule,
  constant gradients and residual-based early stopping.
- Selection boundaries, ties, zero mass, exact parameter counts, frozen
  coordinates and optimizer updates against a dense reference.
- An actual eight-shadow toy training bank with balanced memberships and correct
  nonmember exclusion; disabled audits and failures in sparse case sequences.
- Four complete toy repetitions (two with LiRA, two without), including original
  and gold training, score construction, recovery, saved artifacts, utility and
  privacy plots. Analysis rejects mismatched successful cases across repetitions.

ResNet18 full-product smoke checks passed with 11,181,642 parameters, two 64×64
images, and ranks 2 and 10. The rank-10 check returned a finite
`[11181642, 10]` result in 4.9 seconds.

## Local performance measurements

ResNet18, two 64×64 images, rank 2, one direction per product call, two CPU
threads, one warmup plus two measured calls:

| Curvature implementation | Mean product time |
| --- | ---: |
| Previous BackPACK block operator | 24.037 s |
| Full autograd operator | 0.491 s |

The new path was about **49× faster in this small CPU check**. The two backends
compute different curvature operators; the full operator is separately checked
against exact dense Hessians. This ratio must not be extrapolated to the entire
GPU job or multiplied blindly by the sampling reduction.

For selected-parameter recovery on a 1024×1024 linear layer with batch 32 and one
CPU thread, matching forward outputs and gradients:

| Reconstruction | Forward + backward | Registered state storage |
| --- | ---: | ---: |
| Previous sparse mask | 2.113 ms | 7.62 MB |
| Indexed scatter | 0.757 ms | 5.50 MB |

That is about **2.8× faster** for this layer. The separate 32×32 check improved
from 0.253 ms to 0.145 ms.

## Real-data MNIST check

The repository CNN was trained for 3 epochs on 2,048 random MNIST records. The
target held 256 records; recovery used the other 1,792. Evaluation used 512 test
records. Training seed was 32, fixed curvature sampling seeds were 3/4, subspace
seed was 45, rank was 4, and two power updates were allowed. This bounded check
used the same fixed-random-sample helper as the runner.

| Full curvature sample | Target sample | Score time |
| --- | ---: | ---: |
| 128 | 128 | 2.68 s |
| 512 | 256 | 8.02 s |
| All 2,048 | All 256 | 28.75 s |

At 512 records, a 10% score-mass request selected 2,924 weights. Recovery took
1.05 seconds, with test accuracy changing from 91.99% to 92.58%. The cosine
similarity of the 512-record and full-2,048-record score vectors was 0.938; this
is evidence of similarity, not equality of rankings. Sum-rule relative errors
were at most 2.2e-7. The truncated two-update subspaces had not met the 1e-3
convergence tolerance; that status and their residuals were saved explicitly.

The 3-epoch training took 5.16 seconds, so score plus recovery was **not faster
than training in this deliberately tiny run**. This test checks implementation,
retained utility and sampling sensitivity; it does not establish privacy,
forgetting effectiveness, or a speed advantage for every dataset/training budget.

## Remaining empirical validation

The full CIFAR/ResNet GPU suite was not run from this laptop. Use the saved stage
timings to measure the actual deletion cost separately from the experiment's
training and audit costs. Compare sample/rank/cutoff settings and assess forgetting,
retained utility and LiRA before making publication-level claims. The historical
diagonal implementations remain separate variants, including their optional
stochastic correction; they are not presented as the document's new core.

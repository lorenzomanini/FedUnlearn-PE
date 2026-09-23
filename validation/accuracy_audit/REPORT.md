# CIFAR accuracy audit and bounded recovery fix

The current run does not achieve both useful accuracy and retraining-like forgetting. Small resets leave the target almost perfectly classified; larger resets damage retained performance; the nearly complete zero reset collapses to chance. The score's matrix algebra is correct for its sampled, restricted subspace. That does **not** validate either the sampling/rank approximation or the zero-reset intervention.

A reproducible defect in that intervention is now fixed through an explicit optional initialization-based reset. Zeroing both BatchNorm scale and offset before a ReLU gives exactly zero gradients for that channel; restoring the selected scale to its usual initial value restores gradients. This proves a failure mechanism and its local repair, **not** that the full CIFAR accuracy problem has been resolved. No trained CIFAR checkpoints or images are present locally, so the matched CIFAR ablation remains unrun.

## 1. Audit of the previous changes

The attributable algorithm changes are the diff from `939ab74` to `a95658c`, checked against the prior task history. Existing modifications to README, analysis scripts and queue were preserved. The exhaustive inventory, origins, source references and benchmark caveats are in [speed_audit.md](speed_audit.md).

| Previous change | Departure from the PDF? | Accuracy consequence |
|---|---|---|
| Full/target curvature each capped at 512 records | Additional data approximation to the specified empirical Hessians | Can materially change ratios and masks; samples cover 1.13% of actual training and 10.24% of target records. Main new accuracy-sensitive shortcut. |
| Fixed sample/order and seeded subspace reused through operator passes | Compatible with a fixed operator and subspace | Makes the sampled calculation consistent; does not establish representative sampling. |
| BackPACK parameter-block products replaced by full autograd HVPs; reuse gradient graph within each batch | Corrects a prior full-H mismatch | Cross-tensor derivatives are restored. Ranking can change because the old operator was different. |
| Retain all projected target correlations, rather than only its diagonal | Corrects a prior formula mismatch | Required by equations 55–58; reverting this would change the theoretical method. |
| Explicit positive-support cutoff and skip target products for rejected directions | Matches the PDF's support restriction; threshold is a numerical choice | All 10 directions survived in these runs, so removal did not cause this failure. |
| Residual early stopping at 0.001 | Permitted numerical approximation | Never triggered in the supplied runs. All four full-H calls were used. |
| Default queue reduced from three suites at 2/3/4 updates to one at 3 | Budget choice, not algebra | Less sensitivity testing. The middle 3-update setting already existed; not a universal 5-to-3 reduction. |
| Repetitions 20 to 3 | Evaluation precision only | Does not lower individual-model accuracy; gives weaker uncertainty estimates. |
| LiRA shadows 64 to 8, conditional creation, optionally disabled | Privacy evaluation only | Four IN/four OUT observations per record; noisier privacy evidence. Enabling/disabling the bank can alter train/validation splitting, so ablations must preserve saved indices. |
| LiRA only at sweep cases 0, 5, 9 | Evaluation coverage only | Leaves privacy unknown at seven utility settings; does not damage predictions. |
| Reuse evaluations of identical zero-reset models and LiRA calls | Equivalent deterministic evaluation | No model change. |
| Sparse reconstruction replaced by indexed scatter; remove duplicate trainable template | Algebraically equivalent selected-coordinate optimization | Verified against a dense masked reference. Mode propagation changed separately, below. |
| Chunk residual/score assembly and avoid unnecessary eigenvector CPU copy | Equivalent memory optimization | No truncation or reduced precision. |
| Lazy plotting/poisoning imports | No mathematical effect | Startup/dependency behavior only. |
| Stage timings, score diagnostics and sparse-result indexing | No mathematical effect | Better reporting, not an accuracy shortcut. |

Additional bundled behavioral corrections matter: full curvature now excludes 4,500 validation records never used for training; selection uses the smallest prefix reaching the requested mass with deterministic ties; zero-score groups select nothing; original parameter names/dtypes are preserved. These align implementation with the intended score/selection.

The wrapper also started propagating `.train()` and `.eval()` to its inner model. Previously the dictionary-held template could remain in evaluation mode. This changes **BatchNorm recovery**, beyond the scatter optimization, and needs its own controlled ablation.

**Attribution corrections:** rank 10, top-magnitude-H directions, one recovery epoch, 40 original/gold epochs, SGD 0.01 with momentum 0.9, batches of 128, per-tensor grouping, zero initialization and the broad 0–100% sweep all predate the speed changes. Direction chunking to one HVP direction also predates this commit and preserves the operator/rank. Score reuse across sweep points and shadow-bank reuse across repetitions already existed; the prior runtime description overstated their novelty. There was no new AMP/FP16/BF16/quantization, smaller training set, smaller training batch, or training early stopping.

The prior 49× HVP and 2.8× reconstruction numbers were small CPU microbenchmarks. They were not CIFAR GPU deletion speedups. The old HVP comparison also compared different curvature operators. The previous tiny MNIST utility check did not validate CIFAR forgetting or accuracy.

## 2. Implementation versus theory

Source: `core_score_framework_didactic.pdf`, especially equations 3–11, 39–45 and 55–58, and section 12. The PDF was read as theoretical source material, not as instructions to execute.

For a fixed trained point, the intended quantities are

\[
H=\nabla^2 L_1,\quad H_T=\nabla^2 L_T,\quad
A=U^T H U=\Lambda>0,\quad F=U^T H_T U,
\]
\[
B=\Lambda^{-1/2}F\Lambda^{-1/2},\qquad s_i=\|(UB)_{i,:}\|_2^2.
\]

| Check | Current spectral path |
|---|---|
| Formula and matrix order | Correct. Full symmetric projected F is retained; whitening occurs on both sides; scores are squared row norms, not squared diagonal ratios. |
| Mean-loss normalization | Correct on each chosen sample. Each batch-mean Hessian is weighted by batch size and divided by total examples; an uneven last batch is handled. Target loss is averaged over target records independently. |
| Target weight | The common factor `2*(5000/45500)^2 = 0.0241517` is saved separately. Omitting it preserves rankings and score-mass fractions, as section 12 explicitly allows. Raw sums must not be reported as fully scaled information. |
| Signs/support | Full-H nonpositive Ritz directions are excluded, not made positive by squaring. Target projection signs/cross terms remain present. |
| Parameters | All trainable scalar parameters, including BN affine scale/bias; original names and flattening order preserved. Running mean/variance buffers are not scored parameters. |
| Data | Original training population = 40,500 retained + 5,000 target; retained validation excluded. Score uses fixed 512-record samples of total/target populations. No test examples enter score or recovery. |
| Evaluation point | Same trained parameter vector for total and target products; model in evaluation mode with fixed BN buffers. The point is restored for every product. |
| Selection | Independently within each chosen parameter tensor, smallest descending prefix reaching mass, zero groups select nothing. This matches section 12. |
| Sum rule | Relative errors 1.68e-7, 3.07e-7, 2.04e-7: numerically correct inside the computed subspaces. |

Unresolved approximations/assumptions:

- The Hessians belong to two sampled mean losses, not the exact full/target mean losses. Independently sampled total and target estimates need not obey the exact empirical full/retained/target decomposition.
- Rank 10 restricts an 11,181,642-dimensional model. Large eigenvalues need not capture large target/full-curvature ratios. This was already an approximation, and the PDF expressly calls for rank sensitivity checks.
- All runs used three updates/four full-H passes and **none converged**. Maximum relative residuals were 0.170, 0.210 and 0.194, versus tolerance 0.001. A positive projected subspace still defines the PDF's surrogate; these residuals are evidence against claiming accurate top eigendirections, not evidence that the matrix formula is wrong.
- Training uses batch-statistic BatchNorm; score uses evaluation-mode running statistics. The stationary-point assumption `grad L1 = 0` for that evaluation-mode loss has not been checked. BN buffers can contain target influence and are outside the scored parameter vector. Their recovery updates are an additional empirical state change.
- The discarded mean-response and uncertainty-correction terms are absent by the deliberate definition of the **core**. Their absence is not a coding error, but the core cannot be described as all information or a privacy certificate.
- The PDF explicitly separates local score allocation from the effect of a large reset. It does not prove that resetting gamma fraction of score mass erases gamma fraction of information, or that low forget accuracy is equivalent to retraining.

Preexisting mismatches in the separate **diagonal** path (not used by these CIFAR runs): it uses ratios of sums, adding a constant target-fraction-squared relative to the PDF's mean ratios; this is harmless for within-request ranking. It additionally zeroes negative target diagonals, although the squared ratio in equation 50 allows negative target curvature. Its optional stochastic correction is not this PDF's core, GGN changes the curvature model, and its runner still includes retained validation examples in score data. These should not be conflated with the corrected spectral path or silently changed as part of this fix.

## 3. What the supplied CIFAR results show

The following are means over three runs. Retained training accuracy uses the **40,500 actual retained training records**, not all 45,000 non-target records. The 4,500 held-out retained examples are tabulated separately in the CSV. Test size is 10,000; forget size is 5,000. `retrained` in existing sweep files means **one epoch recovering selected coordinates**, whereas `shadow_out` is the 40-epoch gold retraining baseline.

| Model / score mass | Parameters reset | Test accuracy | Retained train accuracy | Forget accuracy |
|---|---:|---:|---:|---:|
| Original / 0% | 0% | 76.66% | 100.00% | 100.00% |
| Gold retraining | — | 74.72% | 100.00% | 75.18% |
| 11.11% | 1.21% | 73.23% | 99.78% | 99.40% |
| 22.22% | 3.35% | 72.32% | 99.33% | 96.81% |
| 33.33% | 6.36% | 70.69% | 96.33% | 89.99% |
| 44.44% | 10.37% | 68.51% | 87.75% | 79.17% |
| 55.56% | 15.65% | 64.41% | 76.61% | 68.65% |
| 66.67% | 22.69% | 59.73% | 66.89% | 59.73% |
| 77.78% | 32.49% | 57.42% | 62.14% | 57.45% |
| 88.89% | 47.69% | 54.04% | 57.05% | 53.56% |
| 100% | 99.99% | 10.00% | 10.07% | 9.83% |

At the smallest nonzero mask, test accuracy falls immediately to **37.30%**, then recovers to 73.23%. Random reset of roughly the same global parameter count gives 74.95% before recovery and 75.85% after it. This shows strong functional damage from the score-selected intervention even at low parameter count, although stronger target suppression is also expected from informative coordinates.

At 55.56% mass, versus gold, the recovered model loses **10.31 percentage points of test accuracy**, **23.39 points of retained training accuracy**, and **6.53 points of forget accuracy**. Retained held-out accuracy is 65.10% versus gold 75.34%, so the loss is not just a training-set artifact. Test accuracy SD is small relative to that gap; individual runs show the same pattern.

The random control at that mass/count recovers to 72.54% test, 97.00% retained train and 88.70% forget accuracy. It preserves more utility but retains more membership signal. It is only approximately global-count matched: the implementation distributes random resets uniformly by tensor size rather than matching score-selected counts separately in each tensor. It is not a clean layer-matched proof that random ranking is superior.

| LiRA model | AUC | TPR at 1% FPR |
|---|---:|---:|
| Original | 0.8563 | 28.35% |
| Gold | 0.4978 | 1.09% |
| Score + recovery, 55.56% mass | 0.5105 | 1.21% |
| Random + recovery, matched approximate count | 0.5939 | 1.61% |
| Score + recovery, 100% mass | 0.4978 | 1.19% |

Privacy is measured only at 0%, 55.56% and 100% mass. Near-chance LiRA at 100% comes with a useless 10%-accuracy model; it is not a useful unlearning success. At intermediate mass, low leakage coexists with substantial retained damage. At small mass, 99.4% forget accuracy versus 75.18% gold is consistent with under-unlearning; those points lack LiRA measurements, so accuracy alone is not a complete privacy verdict.

The 55.56% model disagrees with gold on 32.43% of test labels; original-to-gold disagreement is already 22.54%, and random-to-gold is 25.86%. These are behavioral differences between independently trained networks. No checkpoints or full logits were saved here, so parameter distance and KL/probability distance cannot be computed. The artifact field called `loss` is actually **true-class logit margin**, not cross-entropy. Cross-entropy cannot be reconstructed from it; logged train/validation CE values are batch averages.

Result alignment checks passed: all ten sweep cases, LiRA mapping `[0,5,9]`, finite arrays, zero-reset outputs identical to original, all target records in LiRA positives, and no original training records in LiRA negatives. The failure is not explained by a missing-case or plotting alignment bug.

Reproduce the numeric audit from the repository root:

```bash
.venv/bin/python validation/accuracy_audit/analyze_saved_cifar.py
```

Outputs: `cifar_utility_{per_run,summary}.csv`, `cifar_privacy_{per_run,summary}.csv`, `cifar_timings_{per_run,summary}.csv`, `cifar_score_diagnostics.csv`, and `cifar_audit_metadata.json`, all alongside this report. SDs use `ddof=0`; they are descriptive spread across three models, not confidence intervals from independent shadow banks.

## 4. Ranked diagnosis and cheap isolation plan

1. **Destructive zero reset plus inadequate recovery is the strongest supported cause.** The damage appears immediately at small masks, gets worse monotonically with mass, and one retained-data epoch cannot restore it. A Conv–BN–ReLU regression reproduces an exact dead-channel mechanism: selected gamma=beta=0 means the ReLU input is identically zero and both selected gradients remain zero. Fresh BN gamma=1 restores nonzero gradients. This mechanism is proven in a small network; the number and contribution of such channels in the saved CIFAR masks cannot be measured because those masks/checkpoints were not saved. Nearly complete zero reset also destroys deep-network learning paths irrespective of score sampling.
2. **512-record curvature and rank-10 restriction may rank shared useful features rather than target-specific influence.** This is the main new speed approximation, combined with inherited severe rank restriction. All eigensolvers are unconverged; no same-checkpoint sample/rank stability comparison exists. It is plausible, not causally isolated from current artifacts.
3. **The BatchNorm mode correction changed recovery behavior.** The old inner template could stay in evaluation mode; the new one trains BN and updates running statistics. The two modes optimize different recovery behavior. Linear-layer scatter equivalence tests did not validate this difference on ResNet. Reverting mode globally would conceal this issue, so test it separately.
4. **The intervention is being asked to do more than the local theory establishes.** Random-client examples share features with retained examples. Allocating a local lower-bound core does not isolate a target-only circuit, especially for large resets. No algebraic error in the PDF was identified that licenses replacing its method.
5. **Evaluation/baseline limitations affect interpretation, not the observed collapse.** Original and gold models are heavily fit (100% retained train versus about 75% test); eight shared shadows and three repetitions limit privacy confidence. These cannot explain the 10–65 point utility losses.

Run one matched original checkpoint first, preserving the saved partition, learning rate, batch size and seeds. Use 44.44% mass initially, near the crossing from under- to over-suppression; repeat at 55.56% only if needed. The ablation order is:

| Comparison | Change exactly one factor | What it isolates |
|---|---|---|
| A vs B | Zero versus initialization reset, one epoch, 512 samples, 3 updates | Dead channels/reset values |
| B vs C | Same initialized reset, one versus up to five epochs | Recovery budget; inspect retained validation curve |
| C vs D | Train versus frozen BN mode | Bundled wrapper mode change |
| C vs E | 512 versus 2,048 samples, same rank/updates | Curvature sampling |
| Best stable setting vs F | 3 versus 5 updates, same sample/rank | Subspace iteration budget |
| Only if still justified | Rank 10 versus 20 at fixed samples/updates | Restricted-subspace coverage |

The last two steps are conditional, not a recommendation to launch a full grid. Each candidate includes its score cost; multiple ablations together are research cost and must not be advertised as one cheap deletion. Saved coordinates allow overlap checks across score settings. Reuse the existing shadow bank for the final candidate's LiRA, without retraining shadows. Never choose an epoch using forget/test accuracy; use retained validation, then audit forgetting separately. More recovery can restore target generalization and can also restore membership leakage, so both must be measured.

## 5. Fix delivered, visible diffs and cost

The implemented **optional** repair preserves the spectral score, positive support and selection rule. It changes the empirical reset value from zero to a fresh model's ordinary initialization on **exactly the same selected coordinates**. This is not a change to the PDF derivation: section 12 leaves the reset intervention empirical and does not require zero. It does change the experimental intervention and is therefore explicit in configuration/results.

| File | Change and reason |
|---|---|
| `fisherunlearn/unlearning.py` | Add optional `reset_reference`; copy selected values from a fresh model/state. Initialize selected trainable vectors from those values so the wrapper does not silently zero them again. Preserve all unselected weights and BN buffers at reset. Reject wrong-shaped references and buffer selection in reference mode. |
| `experiments/config.py` | Add explicit `reset_strategy: zero|initial`. |
| `experiments/runner.py` | Pass the same copied fresh initialization through reset and recovery, and to the random comparator. Record strategy and counts per tensor. Existing default remains zero for reproducibility; opt in to initial mode. |
| `experiments/cifar_recovery_ablation.py` | New checkpoint-only command: fixed saved splits/seeds, selected-coordinate recovery with persistent SGD momentum, optional BN mode comparison, epoch-wise retained validation, saved output checkpoint/mask, and reused-bank LiRA. No original/gold/shadow training. |
| Tests | Preserve zero-mode behavior; verify reference values/frozen coordinates and dense-update equivalence; reproduce dead BN/ReLU gradients and their repair; exercise both runner strategies, budget failure and the checkpoint command on a toy dataset. |

The full source/test additions and modifications are in [changes.patch](changes.patch). A compact view of the behavioral change:

```diff
- reset_parameters(model, selected)                 # selected values become zero
+ reset_parameters(model, selected, reset_reference=fresh_state)

- retrain_params[key] = nn.Parameter(param.new_zeros(count))
+ retrain_params[key] = nn.Parameter(reset_state[name].reshape(-1)[indices].clone())

- UnlearnNet(reset_model, selected)
+ UnlearnNet(reset_model, selected, reset_reference=fresh_state)
```

The first isolation run needs the original checkpoint for `test_0`. From the repository root on the machine that has it and the CIFAR data:

```bash
python -m experiments.cifar_recovery_ablation \
  --suite stat_tests/CIFAR --repetition 0 \
  --checkpoint /path/to/test_0_original_model.pth \
  --output validation/cifar_zero_1epoch \
  --reset-strategy zero --epochs 1

python -m experiments.cifar_recovery_ablation \
  --suite stat_tests/CIFAR --repetition 0 \
  --checkpoint /path/to/test_0_original_model.pth \
  --output validation/cifar_initial_1epoch \
  --reset-strategy initial --epochs 1
```

Both default to 44.44% mass, 512/512 samples, rank 10, three updates and identical seeds. Then repeat the initialized version with `--epochs 5` and a fresh output directory. Add `--batchnorm-mode frozen` only for the BN comparison. Test larger sampling with `--max-samples 2048 --target-max-samples 2048`, separately from `--power-iters 5`. Do not use another repetition's checkpoint: the command checks its predictions/margins on 128 saved training records, which is a useful consistency check rather than a cryptographic identity proof.

For existing sweep callers the explicit candidate change is:

```diff
 test_params_dict = {
-    'retrain_epochs': 1,
+    'retrain_epochs': 5,
+    'reset_strategy': 'initial',
 }
```

That profile change is **proposed, not silently applied to the shipped sweep**. The first paired one-epoch ablation should establish whether initialization helps before spending more recovery time. No automatic exclusion of BN weights, GGN substitution, damping, full-network finetuning, or change to score normalization was introduced.

Measured GPU timing at 55.56% mass: score 6.26 s, selection 1.31 s, recovery 13.73 s, **21.30 s total**; gold retraining 472.78 s. Forecasts use that observed cost:

| Per-deletion candidate | Estimated time | Fraction of 40-epoch gold |
|---|---:|---:|
| Current 512 samples / 3 updates / 1 recovery epoch | 21.30 s measured | 4.50% |
| Same score / 3 recovery epochs | 48.77 s | 10.32% |
| Same score / 5 recovery epochs | 76.23 s | 16.12% |
| 2,048 samples / 5 updates / 5 recovery epochs | 105.25 s | 22.26% |

The last estimate scales score work by `(2048*(6+1))/(512*(4+1)) = 5.6`, then adds selection plus five measured recovery epochs. Fixed overhead, early convergence, mask size, initialization behavior and GPU load can change these times; they are forecasts, not guarantees. Increasing samples to the entire 45,500/5,000 populations would remove much of the intended speed advantage and is not the proposed fix.

The ablation command defaults to a deletion budget of **25% of that repetition's saved gold time**, with at most five epochs. It checks the budget between score/selection/recovery phases and batches, fails when exhausted, and does not label an incomplete candidate successful. An in-flight score call or batch can overrun; this is not a preemptive scheduler wall-time guarantee. Score, mask/reset, recovery, retained validation and model export are charged to deletion. Final test/forget/LiRA reporting and disk serialization are recorded separately. Include those too if the deployment SLA includes auditing. The existing full-sweep evaluation takes about 56 s per case because it evaluates four models; the shared eight-shadow training cost was 4,151 s. Neither should be hidden in a one-model deletion comparison.

Proposed acceptance screen: test/retained-heldout accuracy within two percentage points of gold, retained-training damage within two points, and forgetting/LiRA comparable to gold across repeated models. These are suggested engineering thresholds, not a theorem or a user-specified acceptance limit. Do not accept restored utility alone or chance-level LiRA from a collapsed model. If no candidate meets both utility and forgetting within budget, the method has not met the requested goal; extending runtime beyond retraining is not an acceptable fallback.

## Validation and remaining limit

The original 37-test suite passed (one CUDA-only test skipped). After the patch, the complete suite additionally verifies initialization recovery and budget behavior; the final test result is recorded in `validation_checks.txt`. The CLI help and diff whitespace checks were exercised. The original CIFAR arrays were only read; existing user edits were preserved.

**Remaining empirical work:** run the paired checkpoint ablation on the original CIFAR weights. Those weights/data are absent from this workspace. The code repair and diagnosis are concrete; restored CIFAR accuracy and the final production setting cannot honestly be claimed until that run passes utility, forgetting and measured-cost checks.

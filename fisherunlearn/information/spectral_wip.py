import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from backpack import backpack, extend
from backpack.core.derivatives.adaptive_avg_pool_nd import AdaptiveAvgPool2dDerivatives
from backpack.custom_module.branching import SumModule
from backpack.extensions import HMP
from backpack.extensions.curvmatprod.hmp.batchnorm1d import HMPBatchNorm1d
from backpack.extensions.curvmatprod.hmp.hmpbase import HMPBase
from backpack.extensions.module_extension import ModuleExtension


class _HMPSum(ModuleExtension):
    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        return backproped


class _GraphHMP(HMP):
    def __init__(self):
        super().__init__()
        self.set_module_extension(SumModule, _HMPSum())
        self.set_module_extension(nn.BatchNorm2d, HMPBatchNorm1d())
        self.set_module_extension(
            nn.AdaptiveAvgPool2d,
            HMPBase(AdaptiveAvgPool2dDerivatives()),
        )

    def accumulate_backpropagated_quantities(self, existing, other):
        return lambda mat: existing(mat) + other(mat)


# ============================================================
# Helpers
# ============================================================

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def get_param_info(params):
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    total = sum(numels)
    return shapes, numels, total


def flatten_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def set_params_from_flat(params, flat):
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(flat[offset:offset + n].reshape_as(p))
            offset += n


def split_columns_to_param_blocks(V, shapes, numels):
    """
    V: [P, r] flat block of r vectors
    returns list with one tensor per parameter:
        [r, *param.shape]
    """
    out = []
    offset = 0
    r = V.shape[1]
    for shape, n in zip(shapes, numels):
        block = V[offset:offset + n, :]              # [n, r]
        block = block.transpose(0, 1).reshape(r, *shape)
        out.append(block)
        offset += n
    return out


def merge_param_blocks_to_columns(blocks):
    """
    blocks: list of tensors, each [r, *param.shape]
    returns flat matrix [P, r]
    """
    cols = []
    for block in blocks:
        r = block.shape[0]
        cols.append(block.reshape(r, -1).transpose(0, 1))  # [n, r]
    return torch.cat(cols, dim=0)


def named_tensors_like_params(model, tensors):
    out = {}
    j = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = tensors[j]
            j += 1
    return out


def unflatten_like(vec, shapes, numels):
    out = []
    offset = 0
    for shape, n in zip(shapes, numels):
        out.append(vec[offset:offset + n].reshape(shape))
        offset += n
    return out


# ============================================================
# BackPACK block-Hessian matrix product over a whole loader
# ============================================================

def make_block_hessian_matvec(model, dataloader, loss_fn, device, hmp_chunk_size=1):
    """
    Returns a function matmat(V) that computes block-diagonal Hessian
    times multiple vectors at once.

    V has shape [P, r]
    output has shape [P, r]

    The Hessian is that of the average loss over the whole dataloader,
    but approximated by BackPACK's block-diagonal HMP.

    Process at most hmp_chunk_size direction vectors in each BackPACK call.
    Convolution weight products otherwise build intermediates proportional to
    both the data batch size and the number of directions, which can exceed
    CUDA's 32-bit indexing limit. Chunking preserves the operator and rank.
    """
    if isinstance(hmp_chunk_size, bool) or not isinstance(hmp_chunk_size, int) or hmp_chunk_size < 1:
        raise ValueError("hmp_chunk_size must be a positive integer")

    model = extend(model)
    loss_fn = extend(loss_fn)

    params = get_trainable_params(model)
    shapes, numels, total_params = get_param_info(params)
    theta0 = flatten_params(params).to(device)

    def matmat(V):
        # keep evaluation point fixed
        set_params_from_flat(params, theta0)

        V = V.to(device=device, dtype=theta0.dtype)
        r = V.shape[1]

        out = torch.zeros(total_params, r, device=device, dtype=theta0.dtype)
        total_count = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)   # batch mean

            with backpack(_GraphHMP()):
                loss.backward()

            bs = x.shape[0]
            for start in range(0, r, hmp_chunk_size):
                stop = min(start + hmp_chunk_size, r)
                V_blocks = split_columns_to_param_blocks(V[:, start:stop], shapes, numels)
                HV_blocks = [
                    p.hmp(v_block.contiguous()).detach()
                    for p, v_block in zip(params, V_blocks)
                ]
                HV = merge_param_blocks_to_columns(HV_blocks)
                out[:, start:stop].add_(HV, alpha=bs)
            total_count += bs

        return out / total_count

    return matmat, shapes, numels, total_params


def make_hessian_matvec(model, dataloader, loss_fn, device, hmp_chunk_size=1):
    """Exact full-Hessian products, sharing one gradient graph per data batch.

    Unlike BackPACK's block HMP, this includes derivatives between different
    parameter tensors. Each call uses the same parameter point in evaluation
    mode. The loader must describe a fixed dataset with deterministic transforms:
    changing samples between calls changes the operator during subspace iteration.
    ``hmp_chunk_size`` bounds simultaneous second-derivative directions; one is
    the conservative default for convolutional models.
    """
    if isinstance(hmp_chunk_size, bool) or not isinstance(hmp_chunk_size, int) or hmp_chunk_size < 1:
        raise ValueError("hmp_chunk_size must be a positive integer")
    if getattr(loss_fn, "reduction", "mean") != "mean":
        raise ValueError("Hessian estimation requires a mean-reduced loss")
    params = get_trainable_params(model)
    if not params:
        raise ValueError("Hessian estimation requires trainable parameters")
    shapes, numels, total_params = get_param_info(params)
    theta0 = flatten_params(params).to(device)

    def matmat(V):
        if V.ndim != 2 or V.shape[0] != total_params:
            raise ValueError(f"directions must have shape [{total_params}, rank]")
        set_params_from_flat(params, theta0)
        V = V.to(device=device, dtype=theta0.dtype)
        out = torch.zeros_like(V)
        total_count = 0
        training_states = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            with torch.enable_grad():
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    loss = loss_fn(model(x), y)
                    gradients = torch.autograd.grad(
                        loss, params, create_graph=True, allow_unused=True,
                    )
                    active = [i for i, g in enumerate(gradients)
                              if g is not None and g.requires_grad]
                    bs = x.shape[0]
                    for start in range(0, V.shape[1], hmp_chunk_size):
                        stop = min(start + hmp_chunk_size, V.shape[1])
                        if not active:
                            continue
                        directions = split_columns_to_param_blocks(
                            V[:, start:stop], shapes, numels,
                        )
                        batched = stop - start > 1
                        products = torch.autograd.grad(
                            tuple(gradients[i] for i in active), params,
                            grad_outputs=tuple(directions[i] if batched else directions[i][0]
                                               for i in active),
                            retain_graph=stop < V.shape[1],
                            allow_unused=True, is_grads_batched=batched,
                        )
                        offset = 0
                        for n, product in zip(numels, products):
                            if product is not None:
                                columns = product.detach().reshape(stop - start, n).T
                                out[offset:offset + n, start:stop].add_(columns, alpha=bs)
                            offset += n
                    total_count += bs
                    del loss, gradients
        finally:
            for module, was_training in training_states:
                module.training = was_training
        if not total_count:
            raise ValueError("cannot estimate curvature from an empty loader")
        return out / total_count

    return matmat, shapes, numels, total_params


def make_ggn_matvec(model, dataloader, loss_fn, device, hmp_chunk_size=1):
    """Generalized Gauss-Newton products J^T (d2 loss / d outputs2) J V, matrix-free.

    Positive semidefinite for convex losses such as cross-entropy and squared error,
    so curvature shares between subsets lie in [0, 1]. Only reverse mode is used:
    J v is the derivative of the pullback J^T w with respect to w. Loader, mode and
    chunking semantics match ``make_hessian_matvec``.
    """
    if isinstance(hmp_chunk_size, bool) or not isinstance(hmp_chunk_size, int) or hmp_chunk_size < 1:
        raise ValueError("hmp_chunk_size must be a positive integer")
    if getattr(loss_fn, "reduction", "mean") != "mean":
        raise ValueError("Gauss-Newton estimation requires a mean-reduced loss")
    params = get_trainable_params(model)
    if not params:
        raise ValueError("Gauss-Newton estimation requires trainable parameters")
    shapes, numels, total_params = get_param_info(params)
    theta0 = flatten_params(params).to(device)

    def matmat(V):
        if V.ndim != 2 or V.shape[0] != total_params:
            raise ValueError(f"directions must have shape [{total_params}, rank]")
        set_params_from_flat(params, theta0)
        V = V.to(device=device, dtype=theta0.dtype)
        out = torch.zeros_like(V)
        total_count = 0
        training_states = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            with torch.enable_grad():
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    dual = torch.zeros_like(outputs, requires_grad=True)
                    pullback = torch.autograd.grad(
                        outputs, params, grad_outputs=dual, create_graph=True, allow_unused=True,
                    )
                    active = [i for i, g in enumerate(pullback)
                              if g is not None and g.requires_grad]
                    detached = outputs.detach().requires_grad_(True)
                    output_gradient = torch.autograd.grad(
                        loss_fn(detached, y), detached, create_graph=True,
                    )[0]
                    bs = x.shape[0]
                    for start in range(0, V.shape[1], hmp_chunk_size):
                        stop = min(start + hmp_chunk_size, V.shape[1])
                        if not active:
                            continue
                        directions = split_columns_to_param_blocks(
                            V[:, start:stop], shapes, numels,
                        )
                        batched = stop - start > 1
                        output_tangent = torch.autograd.grad(
                            tuple(pullback[i] for i in active), dual,
                            grad_outputs=tuple(directions[i] if batched else directions[i][0]
                                               for i in active),
                            retain_graph=True, is_grads_batched=batched,
                        )[0]
                        output_curvature = torch.autograd.grad(
                            output_gradient, detached, grad_outputs=output_tangent,
                            retain_graph=True, is_grads_batched=batched,
                        )[0]
                        products = torch.autograd.grad(
                            outputs, params, grad_outputs=output_curvature,
                            retain_graph=True, allow_unused=True, is_grads_batched=batched,
                        )
                        offset = 0
                        for n, product in zip(numels, products):
                            if product is not None:
                                columns = product.detach().reshape(stop - start, n).T
                                out[offset:offset + n, start:stop].add_(columns, alpha=bs)
                            offset += n
                    total_count += bs
                    del outputs, dual, pullback, detached, output_gradient
        finally:
            for module, was_training in training_states:
                module.training = was_training
        if not total_count:
            raise ValueError("cannot estimate curvature from an empty loader")
        return out / total_count

    return matmat, shapes, numels, total_params


# ============================================================
# Block power iteration / subspace iteration
# ============================================================

def top_eigenspace_block_power(
    matmat, total_params, rank, device, dtype, num_iters=8,
    tolerance=1e-3, return_diagnostics=False,
):
    """Rayleigh--Ritz subspace iteration with a measured residual stopping rule.

    Power iteration finds large-magnitude curvature directions; negative Ritz
    values are explicitly excluded by the score estimator. It does not guarantee
    finding every positive direction, nor the largest target-information ratios.
    At most ``num_iters + 1`` operator calls are used, including final refinement.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= total_params:
        raise ValueError("rank must be an integer between 1 and the parameter count")
    if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 0:
        raise ValueError("num_iters must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    Q = torch.randn(total_params, rank, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    converged = False
    started = time.perf_counter()
    for iteration in range(num_iters + 1):
        HQ = matmat(Q)
        projected = Q.T @ HQ
        projected = (projected + projected.T) * 0.5
        evals, small_basis = torch.linalg.eigh(projected)
        order = torch.argsort(evals, descending=True)
        evals, small_basis = evals[order], small_basis[:, order]
        # Compute exact Ritz residual norms without another operator pass or a
        # second parameter-count x rank temporary.
        residual_gram = torch.zeros_like(projected)
        for start in range(0, total_params, 65536):
            stop = start + 65536
            residual = HQ[start:stop] - Q[start:stop] @ projected
            residual_gram.add_(residual.T @ residual)
        residual_norms = torch.diagonal(small_basis.T @ residual_gram @ small_basis).clamp_min(0).sqrt()
        denominator = evals.abs().clamp_min(torch.finfo(dtype).eps)
        relative_residuals = residual_norms / denominator
        converged = bool(torch.all(relative_residuals <= tolerance))
        print(
            f"Curvature subspace pass {iteration + 1}/{num_iters + 1}: "
            f"max relative residual={relative_residuals.max().item():.3g}",
            flush=True,
        )
        if converged or iteration == num_iters:
            break
        Q, _ = torch.linalg.qr(HQ, mode="reduced")
    evecs = Q @ small_basis
    diagnostics = {
        "operator_calls": iteration + 1,
        "power_iterations": iteration,
        "converged": converged,
        "eigenpair_residual_norms": residual_norms.detach().cpu(),
        "eigenpair_relative_residuals": relative_residuals.detach().cpu(),
        "ritz_eigenvalues": evals.detach().cpu(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if return_diagnostics:
        return evals, evecs, diagnostics
    return evals, evecs


def _top_pencil_ritz(gram_a, gram_b, count, cutoff=1e-10):
    """Largest Ritz pairs of a small pencil; nearly B-dependent directions are dropped."""
    gram_a, gram_b = (gram_a + gram_a.T) * 0.5, (gram_b + gram_b.T) * 0.5
    weights, vectors = torch.linalg.eigh(gram_b)
    keep = weights > cutoff * weights.max().clamp_min(torch.finfo(weights.dtype).tiny)
    if int(keep.sum()) < count:
        return None
    whitening = vectors[:, keep] * weights[keep].rsqrt()
    values, rotation = torch.linalg.eigh(whitening.T @ gram_a @ whitening)
    order = torch.argsort(values, descending=True)[:count]
    return values[order], whitening @ rotation[:, order]


def _combine(blocks, which, coefficients):
    """Linear combination of stacked column blocks without concatenating them."""
    result, offset = None, 0
    for block in blocks:
        width = block[which].shape[1]
        term = block[which] @ coefficients[offset:offset + width]
        result = term if result is None else result + term
        offset += width
    return result


def top_generalized_eigenspace(
    pencil, total_params, rank, device, dtype, num_iters=100,
    tolerance=1e-3, return_diagnostics=False, guard=None,
):
    """Block LOBPCG for the largest eigenpairs of A x = mu B x, with B positive definite.

    ``pencil(V)`` returns ``(A V, B V)``; each iteration applies it to one new residual
    block, so neither matrix is formed or inverted. Ritz vectors are B-orthonormal and
    ``||A x - mu B x|| / (|mu| ||B x||) <= tolerance`` for every pair stops early.
    At most ``num_iters + 1`` pencil calls are used, including the initial block.
    Eigenvalues clustered at the top (as in near-interpolating networks) converge
    slowly; the returned residuals then document how far the block is from exact.
    ``guard`` extra vectors (default rank // 4, at least one) are iterated but not
    returned, so the last wanted pair does not trade places with the next one.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= total_params:
        raise ValueError("rank must be an integer between 1 and the parameter count")
    if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 0:
        raise ValueError("num_iters must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if guard is None:
        guard = max(1, rank // 4)
    if isinstance(guard, bool) or not isinstance(guard, int) or guard < 0:
        raise ValueError("guard must be a nonnegative integer")
    size = min(total_params, rank + guard)
    started = time.perf_counter()
    X = torch.randn(total_params, size, device=device, dtype=dtype)
    AX, BX = pencil(X)
    calls = 1
    ritz = _top_pencil_ritz((X.T @ AX).double(), (X.T @ BX).double(), size)
    if ritz is None:
        raise ValueError("the initial block is rank deficient in the B inner product")
    values, rotation = ritz
    rotation = rotation.to(dtype)
    X, AX, BX = X @ rotation, AX @ rotation, BX @ rotation
    previous = None
    tiny = torch.finfo(dtype).tiny
    for iteration in range(num_iters + 1):
        residual = AX - BX * values.to(dtype)
        residual_norms = residual.norm(dim=0)
        relative_residuals = residual_norms / (
            values.abs().to(dtype) * BX.norm(dim=0)).clamp_min(torch.finfo(dtype).eps)
        converged = bool(torch.all(relative_residuals[:rank] <= tolerance))
        print(
            f"Generalized subspace pass {iteration + 1}/{num_iters + 1}: "
            f"max relative residual={relative_residuals[:rank].max().item():.3g}",
            flush=True,
        )
        if converged or iteration == num_iters:
            break
        # A unit-scaled residual, B-orthogonal to X, keeps the search space well conditioned.
        residual = residual - X @ (BX.T @ residual)
        residual = residual / residual.norm(dim=0).clamp_min(tiny)
        AR, BR = pencil(residual)
        calls += 1
        blocks = [(X, AX, BX), (residual, AR, BR)] + ([previous] if previous else [])
        gram = lambda which: torch.cat([
            torch.cat([left[0].T @ right[which] for right in blocks], dim=1) for left in blocks
        ]).double()
        ritz = _top_pencil_ritz(gram(1), gram(2), size)
        if ritz is None and previous:
            # Restart without the previous directions when the search space degenerates.
            blocks = blocks[:2]
            ritz = _top_pencil_ritz(gram(1), gram(2), size)
        if ritz is None:
            break
        values, coefficients = ritz
        coefficients = coefficients.to(dtype)
        X, AX, BX = (_combine(blocks, which, coefficients) for which in range(3))
        tail = coefficients[size:]
        P, AP, BP = (_combine(blocks[1:], which, tail) for which in range(3))
        scale = P.norm(dim=0).clamp_min(tiny)
        previous = (P / scale, AP / scale, BP / scale)
    diagnostics = {
        "operator_calls": calls,
        "power_iterations": iteration,
        "converged": converged,
        "block_size": size,
        "eigenpair_residual_norms": residual_norms[:rank].detach().cpu(),
        "eigenpair_relative_residuals": relative_residuals[:rank].detach().cpu(),
        "ritz_eigenvalues": values[:rank].detach().cpu(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    values, X = values[:rank].to(dtype), X[:, :rank]
    if return_diagnostics:
        return values, X, diagnostics
    return values, X


def core_score_from_subspace(
    basis, full_projection, target_projection,
    eigenvalue_threshold=1e-6, relative_eigenvalue_threshold=1e-5,
):
    """Compute row norms of U A^(-1/2) F A^(-1/2), keeping all of F.

    ``full_projection`` is either the eigenvalue vector in ``basis`` or the
    symmetric matrix U.T H U. The latter form makes invariance under orthogonal
    changes of subspace basis directly testable. Eigenvalues at or below the
    absolute/relative positive threshold are excluded, never damped or squared
    into admissibility. A wholly excluded subspace gives zero score mass.
    """
    for name, value in (("eigenvalue_threshold", eigenvalue_threshold),
                        ("relative_eigenvalue_threshold", relative_eigenvalue_threshold)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    rank = basis.shape[1]
    if target_projection.shape != (rank, rank):
        raise ValueError("target projection must be square with the subspace rank")
    if full_projection.ndim == 1:
        if full_projection.shape != (rank,):
            raise ValueError("eigenvalues must match the subspace rank")
        evals = full_projection
    elif full_projection.shape == (rank, rank):
        evals, rotation = torch.linalg.eigh((full_projection + full_projection.T) * 0.5)
        basis = basis @ rotation
        target_projection = rotation.T @ target_projection @ rotation
    else:
        raise ValueError("full projection must be a rank-vector or a square rank-matrix")
    if not torch.isfinite(evals).all() or not torch.isfinite(target_projection).all():
        raise ValueError("curvature projections must be finite")
    maximum = max(evals.max().item(), 0.0) if evals.numel() else 0.0
    threshold = max(eigenvalue_threshold, relative_eigenvalue_threshold * maximum)
    keep = evals > threshold
    evals = evals[keep]
    if not bool(keep.all()):
        basis = basis[:, keep]
        target_projection = target_projection[keep][:, keep]
    target_projection = (target_projection + target_projection.T) * 0.5
    inv_sqrt = evals.rsqrt()
    whitened = target_projection * inv_sqrt[:, None] * inv_sqrt[None, :]
    scores = torch.empty(basis.shape[0], device=basis.device, dtype=basis.dtype)
    for start in range(0, basis.shape[0], 65536):
        projected_rows = basis[start:start + 65536] @ whitened
        scores[start:start + 65536] = projected_rows.square().sum(dim=1)
    score_sum = scores.sum()
    core_trace = whitened.square().sum()
    return {
        "diag_flat": scores,
        "evecs": basis,
        "evals_H": evals,
        "projected_target": target_projection,
        "whitened_target": whitened,
        "retained_mask": keep,
        "diagnostics": {
            "requested_rank": rank,
            "retained_rank": evals.numel(),
            "eigenvalue_threshold": threshold,
            "absolute_eigenvalue_threshold": eigenvalue_threshold,
            "relative_eigenvalue_threshold": relative_eigenvalue_threshold,
            "score_sum": score_sum.item(),
            "core_trace": core_trace.item(),
            "sum_rule_absolute_error": (score_sum - core_trace).abs().item(),
            "sum_rule_relative_error": ((score_sum - core_trace).abs() /
                                        core_trace.abs().clamp_min(torch.finfo(basis.dtype).tiny)).item(),
        },
    }


def score_agreement(first, second, top_fraction=0.1):
    """Agreement of two coordinate scores: log-score correlation and top-set overlap.

    Values near one mean the ranking barely depends on which records were scored.
    """
    positive = (first > 0) & (second > 0)
    correlation = float("nan")
    if int(positive.sum()) > 1:
        a = first[positive].double().log()
        b = second[positive].double().log()
        a, b = a - a.mean(), b - b.mean()
        norm = a.norm() * b.norm()
        if norm > 0:
            correlation = (a @ b / norm).item()
    count = max(1, int(first.numel() * top_fraction))
    top_first = torch.topk(first, count).indices
    top_second = torch.topk(second, count).indices
    overlap = torch.isin(top_first, top_second).double().mean().item()
    return {"log_score_correlation": correlation, "top_fraction": top_fraction,
            "top_set_overlap": overlap}


def estimate_core_score(
    model, loader_full, loader_a, loss_fn, rank, device,
    num_power_iters=5, hmp_chunk_size=1, curvature_backend="full",
    eigenvalue_threshold=1e-6, relative_eigenvalue_threshold=1e-5,
    power_tolerance=1e-3, return_evecs=False, loader_control=None,
):
    """Matrix-free noncommuting core score from document equations (55)--(58).

    The default uses the full empirical-loss Hessian. ``curvature_backend='ggn'``
    uses positive semidefinite Gauss-Newton curvature and ``'block'`` explicitly
    opts into BackPACK's parameter-block approximation. Loader sample
    caps, rank and thresholds define an approximate restricted curvature model and
    should be reported with results. The common information factor 2 alpha_T**2
    is omitted because it does not affect within-request score-mass selection.
    ``loader_control`` optionally scores a non-target sample in the same basis; its
    agreement with the target score measures how target-specific the ranking is.
    """
    started = time.perf_counter()
    for name, value in (("eigenvalue_threshold", eigenvalue_threshold),
                        ("relative_eigenvalue_threshold", relative_eigenvalue_threshold)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    original_names = [name for name, p in model.named_parameters() if p.requires_grad]
    original_params = get_trainable_params(model)
    if not original_params:
        raise ValueError("score estimation requires trainable parameters")
    if curvature_backend == "full":
        operator_factory = make_hessian_matvec
    elif curvature_backend == "ggn":
        operator_factory = make_ggn_matvec
    elif curvature_backend == "block":
        model = extend(model, use_converter=True)
        operator_factory = make_block_hessian_matvec
    else:
        raise ValueError("curvature_backend must be 'full', 'ggn' or 'block'")
    params = get_trainable_params(model)
    if [id(p) for p in params] != [id(p) for p in original_params]:
        # A converted graph can rename modules, but changing parameter order
        # would invalidate the caller's masks. Refuse an ambiguous mapping.
        raise ValueError("curvature graph conversion changed parameter identity/order")
    dtype = params[0].dtype
    H_matmat, shapes, numels, total_params = operator_factory(
        model, loader_full, loss_fn, device, hmp_chunk_size=hmp_chunk_size,
    )
    Ha_matmat, _, _, _ = operator_factory(
        model, loader_a, loss_fn, device, hmp_chunk_size=hmp_chunk_size,
    )
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        evals, basis, eig_diagnostics = top_eigenspace_block_power(
            H_matmat, total_params, rank, device, dtype,
            num_iters=num_power_iters, tolerance=power_tolerance, return_diagnostics=True,
        )
        # Exclude nonpositive directions before evaluating the target Hessian.
        cutoff = max(eigenvalue_threshold, relative_eigenvalue_threshold *
                     max(evals.max().item(), 0.0))
        keep = evals > cutoff
        positive_basis = basis[:, keep]
        if bool(keep.any()):
            target_products = Ha_matmat(positive_basis)
            target_projection = positive_basis.T @ target_products
            del target_products
        else:
            target_projection = basis.new_empty((0, 0))
        result = core_score_from_subspace(
            positive_basis, evals[keep], target_projection,
            eigenvalue_threshold=eigenvalue_threshold,
            relative_eigenvalue_threshold=relative_eigenvalue_threshold,
        )
        control = None
        if loader_control is not None and bool(keep.any()):
            control_started = time.perf_counter()
            control_matmat, _, _, _ = operator_factory(
                model, loader_control, loss_fn, device, hmp_chunk_size=hmp_chunk_size,
            )
            control_scores = core_score_from_subspace(
                positive_basis, evals[keep], positive_basis.T @ control_matmat(positive_basis),
                eigenvalue_threshold=eigenvalue_threshold,
                relative_eigenvalue_threshold=relative_eigenvalue_threshold,
            )
            control = {
                "projected_target": control_scores["projected_target"].detach().cpu(),
                "whitened_target": control_scores["whitened_target"].detach().cpu(),
                "core_trace": control_scores["diagnostics"]["core_trace"],
                **score_agreement(result["diag_flat"], control_scores["diag_flat"]),
                "elapsed_seconds": time.perf_counter() - control_started,
            }
            del control_scores
    finally:
        for module, was_training in training_states:
            module.training = was_training
    diag_flat = result["diag_flat"].detach().cpu()
    diag_tensors = [t for t in unflatten_like(diag_flat, shapes, numels)]
    # Legacy keys remain available; these diagonal diagnostics are not used to
    # compute scores when the projected target curvature has correlations.
    evals_Ha = result["projected_target"].diagonal()
    diagnostics = {
        **result["diagnostics"],
        "requested_rank": rank,
        "eigenvalue_threshold": cutoff,
        "curvature_backend": curvature_backend,
        "subspace": {key: value.tolist() if torch.is_tensor(value) else value
                     for key, value in eig_diagnostics.items()},
        "retained_eigenpair_residual_norms": eig_diagnostics["eigenpair_residual_norms"][keep.cpu()].tolist(),
        "retained_eigenpair_relative_residuals": eig_diagnostics["eigenpair_relative_residuals"][keep.cpu()].tolist(),
        "target_operator_calls": int(bool(keep.any())),
        "full_hvp_directions": eig_diagnostics["operator_calls"] * rank,
        "target_hvp_directions": int(keep.sum().item()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if control is not None:
        diagnostics["control"] = control
    return {
        "diag_flat": diag_flat,
        "diag_tensors": diag_tensors,
        "diag_by_name": dict(zip(original_names, diag_tensors)),
        "evals_H": result["evals_H"].detach().cpu(),
        "evals_Ha": evals_Ha.detach().cpu(),
        "ratios_sq": (evals_Ha / result["evals_H"]).square().detach().cpu(),
        "projected_target": result["projected_target"].detach().cpu(),
        "whitened_target": result["whitened_target"].detach().cpu(),
        "diagnostics": diagnostics,
        "evecs": result["evecs"].detach().cpu() if return_evecs else None,
    }


def _largest_eigenvalue(matmat, total_params, device, dtype, num_iters):
    """Rayleigh-quotient power estimate of a positive semidefinite operator's top eigenvalue."""
    vector = torch.randn(total_params, 1, device=device, dtype=dtype)
    vector = vector / vector.norm()
    value = 0.0
    for _ in range(num_iters):
        product = matmat(vector)
        value = (vector.T @ product).item()
        norm = product.norm()
        if norm == 0:
            break
        vector = product / norm
    return max(value, 0.0)


def estimate_generalized_core_score(
    model, loader_retained, loader_target, loss_fn, rank, device, target_fraction,
    num_iters=100, hmp_chunk_size=1, curvature_backend="ggn", damping=1e-4,
    eigenvalue_threshold=1e-6, relative_eigenvalue_threshold=1e-5,
    tolerance=1e-3, return_evecs=False, loader_control=None, damping_power_iters=20,
):
    """Core score in the core-optimal subspace: top generalized eigenvectors of (H_T, H).

    For a rank-r subspace, Tr(B^2) is at most the sum of the r largest squared
    relative curvatures mu = v^T H_T v / v^T H v, attained by the top generalized
    eigenvectors (Poincare separation); the top eigenvectors of H need not attain it.
    H = (1 - a) H_R + a H_T is estimated from a retained sample and the target sample
    itself (a = target_fraction), so with Gauss-Newton curvature 0 <= mu <= 1/a, and
    a * mu is the target's share of the curvature along v. Block LOBPCG solves the
    pencil (H_T, H + delta I) with delta = damping * lambda_max(H); delta acts as a
    prior precision, ranks directions by curvature where many share the bound, and
    only lowers mu. The basis is scored exactly as in ``estimate_core_score``.
    """
    started = time.perf_counter()
    if not 0 < target_fraction < 1:
        raise ValueError("target_fraction must lie strictly between 0 and 1")
    if curvature_backend != "ggn":
        raise ValueError("the generalized subspace needs positive semidefinite 'ggn' curvature")
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    for name, value in (("eigenvalue_threshold", eigenvalue_threshold),
                        ("relative_eigenvalue_threshold", relative_eigenvalue_threshold)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    params = get_trainable_params(model)
    if not params:
        raise ValueError("score estimation requires trainable parameters")
    dtype = params[0].dtype
    target_matmat, shapes, numels, total_params = make_ggn_matvec(
        model, loader_target, loss_fn, device, hmp_chunk_size=hmp_chunk_size,
    )
    retained_matmat, _, _, _ = make_ggn_matvec(
        model, loader_retained, loss_fn, device, hmp_chunk_size=hmp_chunk_size,
    )

    def curvature(V):
        target_products = target_matmat(V)
        return target_products, (1 - target_fraction) * retained_matmat(V) + target_fraction * target_products

    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        scale = _largest_eigenvalue(lambda V: curvature(V)[1], total_params, device, dtype,
                                    damping_power_iters)
        delta = damping * scale

        def pencil(V):
            target_products, full_products = curvature(V)
            return target_products, full_products + delta * V

        relative_curvatures, vectors, eig_diagnostics = top_generalized_eigenspace(
            pencil, total_params, rank, device, dtype, num_iters=num_iters,
            tolerance=tolerance, return_diagnostics=True,
        )
        basis, _ = torch.linalg.qr(vectors, mode="reduced")
        del vectors
        target_products, full_products = curvature(basis)
        full_projection = basis.T @ full_products
        full_projection = (full_projection + full_projection.T) * 0.5
        target_projection = basis.T @ target_products
        del target_products, full_products
        thresholds = dict(eigenvalue_threshold=eigenvalue_threshold,
                          relative_eigenvalue_threshold=relative_eigenvalue_threshold)
        result = core_score_from_subspace(basis, full_projection, target_projection, **thresholds)
        control = None
        if loader_control is not None:
            control_started = time.perf_counter()
            control_matmat, _, _, _ = make_ggn_matvec(
                model, loader_control, loss_fn, device, hmp_chunk_size=hmp_chunk_size,
            )
            control_scores = core_score_from_subspace(
                basis, full_projection, basis.T @ control_matmat(basis), **thresholds,
            )
            control = {
                "projected_target": control_scores["projected_target"].detach().cpu(),
                "whitened_target": control_scores["whitened_target"].detach().cpu(),
                "core_trace": control_scores["diagnostics"]["core_trace"],
                **score_agreement(result["diag_flat"], control_scores["diag_flat"]),
                "elapsed_seconds": time.perf_counter() - control_started,
            }
            del control_scores
    finally:
        for module, was_training in training_states:
            module.training = was_training
    diag_flat = result["diag_flat"].detach().cpu()
    diag_tensors = list(unflatten_like(diag_flat, shapes, numels))
    evals_Ha = result["projected_target"].diagonal()
    diagnostics = {
        **result["diagnostics"],
        "subspace_mode": "generalized",
        "requested_rank": rank,
        "curvature_backend": curvature_backend,
        "target_fraction": target_fraction,
        "relative_curvature_bound": 1 / target_fraction,
        "damping": damping,
        "damping_absolute": delta,
        "curvature_scale": scale,
        "generalized_eigenvalues": relative_curvatures.detach().cpu().tolist(),
        "subspace": {key: value.tolist() if torch.is_tensor(value) else value
                     for key, value in eig_diagnostics.items()},
        "retained_eigenpair_residual_norms": eig_diagnostics["eigenpair_residual_norms"].tolist(),
        "retained_eigenpair_relative_residuals": eig_diagnostics["eigenpair_relative_residuals"].tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if control is not None:
        diagnostics["control"] = control
    return {
        "diag_flat": diag_flat,
        "diag_tensors": diag_tensors,
        "diag_by_name": dict(zip(names, diag_tensors)),
        "evals_H": result["evals_H"].detach().cpu(),
        "evals_Ha": evals_Ha.detach().cpu(),
        "ratios_sq": (evals_Ha / result["evals_H"]).square().detach().cpu(),
        "projected_target": result["projected_target"].detach().cpu(),
        "whitened_target": result["whitened_target"].detach().cpu(),
        "diagnostics": diagnostics,
        "evecs": result["evecs"].detach().cpu() if return_evecs else None,
    }


def estimate_diag_commuting_backpack(
    model, loader_full, loader_a, loss_fn, rank, device,
    num_power_iters=8, hmp_chunk_size=1, **kwargs,
):
    """Compatibility name for the corrected noncommuting core estimator.

    Despite the historical name, the default is now the full Hessian and target
    correlations are always retained. Pass ``curvature_backend='block'`` to opt
    into the old curvature approximation. Legacy callers still receive eigenvectors.
    """
    kwargs.setdefault("return_evecs", True)
    return estimate_core_score(
        model, loader_full, loader_a, loss_fn, rank, device,
        num_power_iters=num_power_iters, hmp_chunk_size=hmp_chunk_size, **kwargs,
    )

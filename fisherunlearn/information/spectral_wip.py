import torch
from torch.utils.data import DataLoader
from backpack import backpack, extend
from backpack.extensions import HMP


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

def make_block_hessian_matvec(model, dataloader, loss_fn, device):
    """
    Returns a function matmat(V) that computes block-diagonal Hessian
    times multiple vectors at once.

    V has shape [P, r]
    output has shape [P, r]

    The Hessian is that of the average loss over the whole dataloader,
    but approximated by BackPACK's block-diagonal HMP.
    """
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

            with backpack(HMP()):
                loss.backward()

            V_blocks = split_columns_to_param_blocks(V, shapes, numels)

            HV_blocks = []
            for p, v_block in zip(params, V_blocks):
                HV_blocks.append(p.hmp(v_block).detach())

            HV = merge_param_blocks_to_columns(HV_blocks)

            bs = x.shape[0]
            out += bs * HV
            total_count += bs

        return out / total_count

    return matmat, shapes, numels, total_params


# ============================================================
# Block power iteration / subspace iteration
# ============================================================

def top_eigenspace_block_power(matmat, total_params, rank, device, dtype, num_iters=8):
    """
    Compute top-r eigenvectors of a symmetric operator using block power iteration.
    """
    Q = torch.randn(total_params, rank, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(Q, mode="reduced")

    for i in range(num_iters):
        print(f"Iter {i+1}/{num_iters}", end="\r")
        Z = matmat(Q)                   # [P, r]
        Q, _ = torch.linalg.qr(Z, mode="reduced")
    print("Finished iter")

    # Rayleigh-Ritz refinement
    HQ = matmat(Q)                      # [P, r]
    B = Q.T @ HQ                        # [r, r]
    evals, U_small = torch.linalg.eigh(B)

    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    U_small = U_small[:, order]

    evecs = Q @ U_small                 # [P, r]
    return evals, evecs


# ============================================================
# Main estimator under commuting assumption
# ============================================================

def estimate_diag_commuting_backpack(
    model,
    loader_full,
    loader_a,
    loss_fn,
    rank,
    device,
    num_power_iters=8,
):
    """
    Fast approximation of diag((H^+ H_a)^2) under:
      - H and H_a commute
      - H and H_a are approximated by BackPACK block-diagonal Hessians

    Steps:
      1. top eigenvectors of block-diagonal H via block power iteration
      2. project block-diagonal H_a onto those eigenvectors
      3. reconstruct diagonal:
           sum_k ((lambda_a_k / lambda_k)^2 * u_k^2)
    """
    params = get_trainable_params(model)
    dtype = params[0].dtype

    # block-diagonal Hessian operators
    H_matmat, shapes, numels, total_params = make_block_hessian_matvec(
        model, loader_full, loss_fn, device
    )
    Ha_matmat, _, _, _ = make_block_hessian_matvec(
        model, loader_a, loss_fn, device
    )

    # top eigenpairs of H
    evals_H, evecs = top_eigenspace_block_power(
        matmat=H_matmat,
        total_params=total_params,
        rank=rank,
        device=device,
        dtype=dtype,
        num_iters=num_power_iters,
    )  # evals_H: [r], evecs: [P, r]
    # eigenvalues of H_a on the same vectors
    HaU = Ha_matmat(evecs)                      # [P, r]
    evals_Ha = torch.sum(evecs * HaU, dim=0)   # [r], Rayleigh quotients

    ratios_sq = (evals_Ha / evals_H) ** 2      # [r]

    diag_flat = torch.sum((evecs ** 2) * ratios_sq.unsqueeze(0), dim=1)  # [P]

    diag_tensors = [t.detach().cpu() for t in unflatten_like(diag_flat, shapes, numels)]

    return {
        "diag_flat": diag_flat.detach().cpu(),
        "diag_tensors": diag_tensors,
        "diag_by_name": named_tensors_like_params(model, diag_tensors),
        "evals_H": evals_H.detach().cpu(),
        "evals_Ha": evals_Ha.detach().cpu(),
        "ratios_sq": ratios_sq.detach().cpu(),
        "evecs": evecs.detach().cpu(),
    }

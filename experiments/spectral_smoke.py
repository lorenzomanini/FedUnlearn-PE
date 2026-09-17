"""Check ResNet18 spectral HMP on one synthetic batch before a long suite.

Run inside an allocated GPU job:
    python -m experiments.spectral_smoke
"""

import argparse
import time

import torch
from backpack import extend
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

from fisherunlearn.information.spectral_wip import make_block_hessian_matvec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--hmp-chunk-size", type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.rank, args.hmp_chunk_size) < 1:
        parser.error("batch size, rank, and HMP chunk size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run this check inside an allocated GPU job")

    torch.manual_seed(2026)
    model = extend(resnet18(num_classes=10).to(device).eval(), use_converter=True)
    dataset = TensorDataset(
        torch.randn(args.batch_size, 3, 32, 32),
        torch.randint(10, (args.batch_size,)),
    )
    matmat, _, _, n_params = make_block_hessian_matvec(
        model, DataLoader(dataset, batch_size=args.batch_size),
        nn.CrossEntropyLoss(), device, hmp_chunk_size=args.hmp_chunk_size,
    )
    directions = torch.randn(n_params, args.rank, device=device)
    print(
        f"ResNet18 HMP: device={device}, batch={args.batch_size}, "
        f"rank={args.rank}, chunk={args.hmp_chunk_size}", flush=True,
    )
    start = time.perf_counter()
    products = matmat(directions)
    if products.shape != directions.shape or not torch.isfinite(products).all().item():
        raise RuntimeError("HMP returned an invalid shape or non-finite values")
    print(f"PASS: {tuple(products.shape)} finite products in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()

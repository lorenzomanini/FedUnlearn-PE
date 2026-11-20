"""Forget one CIFAR-100 fine class while predicting its 20 superclasses.

Prepare the data once on a login node with --download; Slurm workers stay offline.
Each seed trains one original and three independent retained-data gold models.
"""

import argparse
import json
from pathlib import Path


SEEDS = tuple(range(3000, 3015))
# Pilot seeds were analysed before PRIMARY_CONTRAST was fixed: exploratory only.
PILOT_SEEDS = (3000, 3001, 3002)
# 3003-3009 (n=7) until the 2026-11-20 amendment recorded in PRIMARY_CONTRAST.
CONFIRMATORY_SEEDS = tuple(range(3003, 3015))
REGULARIZED_SEEDS = (3003, 3004, 3005)
CONFIG = {
    "dataset_name": "cifar100_superclass",
    "model_name": "resnet18",
    "num_classes": 20,
    "distribution_type": "fine_class",
    "num_clients": 6,
    "target_client": 0,
    "target_fine_class": 0,
    "swap_fine_class": 53,
    # pickup_truck (vehicles_1): retained control from an unrelated superclass.
    "cross_fine_class": 58,
    "download": False,
    "save_models": False,
    "num_gold_models": 3,
    "parameter_budgets": [8.0, 10.0, 15.0],
    "recovery_epochs": 1,
    "batch_size": 128,
    "eval_batch_size": 256,
    "train_epochs": 40,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "augment": False,
    "drop_last": False,
    "spectral_rank": 20,
    "spectral_num_power_iters": 50,
    "spectral_max_samples": 2048,
    "spectral_target_max_samples": 512,
    "spectral_seed": 0,
    "spectral_curvature_backend": "ggn",
    "spectral_eigenvalue_min": 1e-6,
    "spectral_eigenvalue_rtol": 1e-5,
    "spectral_power_tolerance": 1e-3,
}
# Regularized base model: 32-px crop with 4-px padding and horizontal flip before
# the 64-px resize, plus weight decay, for original, gold and recovery training.
# Scoring and evaluation always use un-augmented images.
VARIANTS = {"standard": {}, "regularized": {"augment": True, "weight_decay": 5e-4}}

# Pre-specified on 2026-11-13; amended once on 2026-11-20 (seed count only, see
# "amendments"), before any confirmatory seed was trained. Do not edit after
# confirmatory results exist; analysis/prepare_cifar100_tmlr.py reads this dictionary.
PRIMARY_CONTRAST = {
    "endpoint": "target_accuracy_closure",
    "endpoint_definition": (
        "apple accuracy closure, NFS_TargetAcc (paper Eq. 55): "
        "(acc_original - acc_arm) / (acc_original - mean acc of the seed's golds) "
        "on the 100 apple test images"),
    "arm": "core", "arm_budget_percent": 10.0,
    "control": "random", "control_budgets_percent": [8.0, 10.0, 15.0, 20.0, 30.0, 40.0],
    "matching": "retained_accuracy_percent",
    "matching_definition": (
        "within each seed, linearly interpolate the control's closure against its "
        "retained test accuracy (all non-apple test images) and evaluate it at the "
        "arm's retained accuracy; no extrapolation: a seed whose control points do "
        "not bracket the arm is excluded and reported"),
    "statistic": "per-seed difference, arm minus interpolated control",
    "test": "two-sided one-sample t-test of the per-seed differences against 0",
    "alpha": 0.05,
    "robustness_check": "exact two-sided sign-flip test on the same differences",
    "seeds": list(CONFIRMATORY_SEEDS),
    "amendments": [{
        "date": "2026-11-20",
        "change": "confirmatory seeds 3003-3009 (n=7) -> 3003-3014 (n=12); nothing else",
        "reason": ("power: pilot controls (job 64858158) gave per-seed SD 0.13 and a "
                   "run-to-run noise floor of ~0.2 closure, so n=7 has ~0.45 power at the "
                   "pilot mean +0.165 if SD=0.2; n=12 has ~0.74"),
        "data_status": ("no confirmatory seed had produced results: job 64686505 tasks "
                        "3-12 failed before training"),
    }],
    "variant": "standard",
    "exploratory": ("pilot seeds 3000-3002, the regularized variant and every other "
                    "contrast are reported descriptively with t-intervals"),
}


def check_data(download=False):
    """Fail before training if the offline dataset or its partition is unavailable."""
    from experiments.datasets import get_clients_subsets, get_datasets

    config = {**CONFIG, "download": download}
    train, test = get_datasets(config)
    clients = get_clients_subsets(train, config)
    target = config["target_fine_class"]
    swap = config["swap_fine_class"]
    cross = config["cross_fine_class"]
    if target == swap or train.fine_to_coarse[target] != train.fine_to_coarse[swap]:
        raise ValueError("The target and retained swap must be different fine classes in one superclass")
    if train.fine_to_coarse[target] == train.fine_to_coarse[cross]:
        raise ValueError("The cross control must come from a different superclass")
    print(json.dumps({
        "training_records": len(train),
        "test_records": len(test),
        "target_records": len(clients[config["target_client"]]),
        "target_fine_class": train.fine_classes[target],
        "swap_fine_class": train.fine_classes[swap],
        "cross_fine_class": train.fine_classes[cross],
        "superclass": train.classes[train.fine_to_coarse[target]],
        "cross_superclass": train.classes[train.fine_to_coarse[cross]],
    }, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--download", action="store_true", help="download CIFAR-100 once on a login node")
    actions.add_argument("--preflight", action="store_true", help="validate cached data without downloading or training")
    actions.add_argument("--describe", action="store_true", help="print the protocol as JSON without loading data")
    actions.add_argument("--smoke", action="store_true", help="check the scoring path on synthetic data")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="standard")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, help="new directory for this seed's artifacts")
    parser.add_argument("--save-models", action="store_true",
                        help="also retain weights, masks and scores for checkpoint-based follow-ups")
    args = parser.parse_args(argv)
    config = {**CONFIG, **VARIANTS[args.variant], "variant": args.variant,
              "save_models": args.save_models}

    if args.download or args.preflight:
        check_data(download=args.download)
    elif args.describe:
        print(json.dumps({"experiment": "cifar100_within_class", "seeds": SEEDS,
                          "seed": args.seed, "variant": args.variant, "config": config,
                          "primary_contrast": PRIMARY_CONTRAST}, indent=2))
    elif args.smoke:
        from experiments.within_class import smoke

        smoke(config, args.device)
    elif args.output is None:
        parser.error("--output is required to run the experiment")
    else:
        from experiments.within_class import run

        run(config, args.output, args.seed, args.device)


if __name__ == "__main__":
    main()

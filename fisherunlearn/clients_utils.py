"""Compatibility facade for the original client-data helper imports."""

from .data import (
    concatenate_subsets,
    create_poisoned_data,
    poisoning_data,
    random_split_subset,
    split_dataset_by_class_distribution,
)

__all__ = [
    "split_dataset_by_class_distribution",
    "concatenate_subsets",
    "random_split_subset",
    "create_poisoned_data",
    "poisoning_data",
]

"""Compatibility facade for the spectral-WIP estimator."""

from fisherunlearn.information.spectral_wip import (
    estimate_diag_commuting_backpack,
    flatten_params,
    get_param_info,
    get_trainable_params,
    make_block_hessian_matvec,
    merge_param_blocks_to_columns,
    named_tensors_like_params,
    set_params_from_flat,
    split_columns_to_param_blocks,
    top_eigenspace_block_power,
    unflatten_like,
)

__all__ = [
    "get_trainable_params",
    "get_param_info",
    "flatten_params",
    "set_params_from_flat",
    "split_columns_to_param_blocks",
    "merge_param_blocks_to_columns",
    "named_tensors_like_params",
    "unflatten_like",
    "make_block_hessian_matvec",
    "top_eigenspace_block_power",
    "estimate_diag_commuting_backpack",
]

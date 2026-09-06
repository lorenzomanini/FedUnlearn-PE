"""Compatibility facade for the FedUnlearn scientific API."""

from . import runtime
from .information import (
    compute_client_information,
    find_informative_params,
    plot_information_parameters_tradeoff,
)
from .privacy import mia_attack
from .runtime import set_device, set_info_batch_size, set_mia_batch_size
from .unlearning import UnlearnNet, reset_parameters

__all__ = [
    "DEVICE",
    "INFO_BATCH_SIZE",
    "MIA_BATCH_SIZE",
    "set_device",
    "set_info_batch_size",
    "set_mia_batch_size",
    "compute_client_information",
    "plot_information_parameters_tradeoff",
    "find_informative_params",
    "reset_parameters",
    "UnlearnNet",
    "mia_attack",
]


def __getattr__(name):
    if name in {"DEVICE", "INFO_BATCH_SIZE", "MIA_BATCH_SIZE"}:
        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

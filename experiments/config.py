from dataclasses import dataclass
from typing import Any, Literal, Mapping, NotRequired, TypedDict


class LegacyInitParamsDict(TypedDict):
    test_name: str
    dataset_name: Literal["mnist", "cifar10", "FashionMNIST", "cifar100"]
    num_clients: int
    num_classes: int
    distribution_type: Literal[
        "preferential_class", "uniform", "dirichlet", "random", "categorical"
    ]
    model_name: Literal["simple_cnn", "resnet18"]
    loss_name: Literal["cross_entropy", "mse"]
    trainer_name: Literal["sgd"]
    train_epochs: int
    target_client: int
    num_tests: int
    info_use_converter: bool
    use_FIM: bool
    hessian_method: Literal["diag_hessian", "diag_ggn", "diag_ggn_mc"]
    poison: bool
    local_epochs: int
    participation_rate: float
    lr: float


class RevisedInitParamsDict(LegacyInitParamsDict):
    learning_rate: float
    momentum: float
    num_shadow_models: NotRequired[int]
    lira_seed: NotRequired[int]
    lira_global_variance: NotRequired[bool]
    spectral_rank: NotRequired[int]
    spectral_num_power_iters: NotRequired[int]
    spectral_max_samples: NotRequired[int | None]
    spectral_target_max_samples: NotRequired[int | None]
    spectral_seed: NotRequired[int]
    spectral_curvature_backend: NotRequired[Literal["full", "block"]]
    spectral_eigenvalue_min: NotRequired[float]
    spectral_eigenvalue_rtol: NotRequired[float]
    spectral_power_tolerance: NotRequired[float]
    spectral_hmp_chunk_size: NotRequired[int]


class TestParamsDict(TypedDict):
    subtest: int
    unlearning_method: Literal["information", "parameters"]
    unlearning_percentage: float
    retrain_epochs: int
    reset_strategy: NotRequired[Literal["zero", "initial"]]
    tests: list[str]
    mia_classifier_types: list[Literal["nn", "logistic", "svm"]]
    whitelist: list[str]
    blacklist: list[str]


@dataclass(frozen=True)
class RuntimeConfig:
    generation: Literal["legacy", "revised_diagonal", "spectral_wip"]
    device: Any
    info_batch_size: int
    mia_batch_size: int
    eval_batch_size: int
    train_batch_size: int
    num_power_iters: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    values: Mapping[str, Any]

    def to_legacy_dict(self):
        return dict(self.values)


@dataclass(frozen=True)
class UnlearningCase:
    values: Mapping[str, Any]

    def to_legacy_dict(self):
        return dict(self.values)

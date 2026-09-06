import functools

from torch import nn
from torchvision.models import resnet18


class FLNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )


class FLNet2(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 43),
        )


class FeedForwardNN(nn.Sequential):
    def __init__(self, input_size, num_classes):
        dim_step = input_size // 4
        hidden_sizes = [
            input_size - dim_step,
            input_size - 2 * dim_step,
            input_size - 3 * dim_step,
        ]
        super().__init__(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], num_classes),
        )


def create_resnet(init_params_dict):
    model = resnet18(num_classes=init_params_dict["num_classes"])
    if init_params_dict["dataset_name"] == "mnist":
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    return model


def get_model_class(init_params_dict):
    model_name = init_params_dict["model_name"]
    if model_name == "simple_cnn":
        init_params_dict["info_use_converter"] = False
        return FLNet
    elif model_name == "resnet18":
        init_params_dict["info_use_converter"] = True
        return functools.partial(
            create_resnet, init_params_dict=init_params_dict.copy()
        )
    elif model_name == "complex_cnn":
        init_params_dict["info_use_converter"] = False
        return FLNet2
    elif model_name == "feedforward_nn":
        input_dim = init_params_dict["input_dim"]
        num_classes = init_params_dict["num_classes"]
        init_params_dict["info_use_converter"] = False
        return functools.partial(
            FeedForwardNN, input_size=input_dim, num_classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def get_loss_class(init_params_dict):
    loss_name = init_params_dict["loss_name"]
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss
    elif loss_name == "mse":
        return nn.MSELoss
    else:
        raise ValueError("Unsupported loss name")

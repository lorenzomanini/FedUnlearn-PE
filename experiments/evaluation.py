import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def compute_accuracy(model, dataset, device, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    tqdm_bar = tqdm(
        total=len(dataloader), desc="Computing accuracy", unit="batch", leave=False
    )
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tqdm_bar.update(1)
    tqdm_bar.close()
    model.cpu()
    model.train(was_training)
    return correct / total


class PerformanceEvaluator:
    def __init__(self, model, dataset, device, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.outputs = []
        self.labels = []
        was_training = model.training
        model.to(device)
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                self.outputs.append(model(images).cpu())
                self.labels.append(labels.cpu())
        self.outputs = torch.cat(self.outputs)
        self.labels = torch.cat(self.labels)
        model.cpu()
        model.train(was_training)

    def _labels_outputs_subset(self, subset):
        if subset is not None:
            subset_indices = subset.indices
            subset_outputs = self.outputs[subset_indices]
            subset_labels = self.labels[subset_indices]
        else:
            subset_outputs = self.outputs
            subset_labels = self.labels
        return subset_outputs, subset_labels

    def get_accuracy(self, subset=None):
        subset_outputs, subset_labels = self._labels_outputs_subset(subset)
        _, predicted = torch.max(subset_outputs.data, 1)
        correct = (predicted == subset_labels).sum().item()
        return correct / len(subset_labels)

    def get_losses(self, subset=None):
        subset_outputs, subset_labels = self._labels_outputs_subset(subset)
        return nn.CrossEntropyLoss(reduction="none")(
            subset_outputs, subset_labels
        )

    def offline_lira_score(self, local_means, global_var, subset=None):
        subset_outputs, subset_labels = self._labels_outputs_subset(subset)
        loss = nn.CrossEntropyLoss(reduction="none")(
            subset_outputs, subset_labels
        )

        def cdf(x, mean, var):
            return 0.5 * (
                1 + torch.erf((x - mean) / (torch.sqrt(2 * var)))
            )

        # Cross-entropy is lower for members, so membership is the OUT upper tail.
        return 1 - cdf(loss, local_means, global_var)


def logit_margin(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    zy = z.gather(1, y.unsqueeze(1)).squeeze(1)
    top2_vals, top2_idx = z.topk(k=2, dim=1)
    top1_is_true = top2_idx[:, 0] == y
    z_other = torch.where(top1_is_true, top2_vals[:, 1], top2_vals[:, 0])
    return zy - z_other


def logit_confidence(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stable logit of the true-class confidence used by online LiRA."""
    zy = z.gather(1, y.unsqueeze(1)).squeeze(1)
    other = z.masked_fill(
        torch.nn.functional.one_hot(y, num_classes=z.shape[1]).bool(),
        -torch.inf,
    )
    return zy - torch.logsumexp(other, dim=1)


def evaluate_model(model, dataset, device, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    margins = []
    was_training = model.training
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                preds.append(torch.argmax(output, dim=1).cpu())
                margins.append(logit_margin(output, labels).cpu())
    finally:
        model.cpu()
        model.train(was_training)
    return {
        "pred": torch.cat(preds).numpy(),
        "loss": torch.cat(margins).numpy(),
    }


def evaluate_lira(model, dataset, device, batch_size):
    """Evaluate the paper's stable logit-confidence statistic on a fixed model."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    was_training = model.training
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                scores.append(logit_confidence(model(images), labels).cpu())
    finally:
        model.cpu()
        model.train(was_training)
    return torch.cat(scores).numpy()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader

from . import runtime


def mia_attack(
    model, member_dataset, nonmember_dataset, classifier_type="logistic", plot=False
):
    member_loader = DataLoader(
        member_dataset, runtime.MIA_BATCH_SIZE, shuffle=False
    )
    nonmember_loader = DataLoader(
        nonmember_dataset, runtime.MIA_BATCH_SIZE, shuffle=False
    )
    model.eval()
    if classifier_type not in ["logistic", "svm", "linear", "nn"]:
        raise ValueError(
            "Invalid classifier_type: choose 'logistic', 'svm', 'linear', or 'nn'."
        )

    def get_features(model, dataloader):
        model.eval()
        model.to(runtime.DEVICE)
        confs, losses, entropies = [], [], []
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(runtime.DEVICE), y.to(runtime.DEVICE)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                conf, _ = probs.max(dim=1)
                loss = ce_loss(logits, y)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
                confs.extend(conf.cpu().numpy())
                losses.extend(loss.cpu().numpy())
                entropies.extend(entropy.cpu().numpy())
        return np.stack([losses], axis=1)

    X_member = get_features(model, member_loader)
    X_nonmember = get_features(model, nonmember_loader)
    X = np.concatenate([X_member, X_nonmember])
    y = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])

    if classifier_type in ["logistic", "svm", "linear"]:
        if classifier_type == "logistic":
            clf = LogisticRegression(max_iter=1000)
        elif classifier_type == "svm":
            clf = SVC(probability=True)
        elif classifier_type == "linear":
            clf = SGDClassifier(loss="log_loss", max_iter=1000)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        y_score = clf.predict_proba(X)[:, 1]
        y_true = y
        auc = roc_auc_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)
    elif classifier_type == "nn":
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1, 16)
                self.fc2 = nn.Linear(16, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return torch.sigmoid(self.fc2(x))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model_nn = SimpleNN().to(runtime.DEVICE)
        optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(
            runtime.DEVICE
        )
        y_train_tensor = torch.tensor(
            y_train.reshape(-1, 1), dtype=torch.float32
        ).to(runtime.DEVICE)
        model_nn.train()
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model_nn(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        model_nn.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(
            runtime.DEVICE
        )
        y_test_tensor = torch.tensor(
            y_test.reshape(-1, 1), dtype=torch.float32
        ).to(runtime.DEVICE)
        with torch.no_grad():
            y_score = model_nn(X_test_tensor).cpu().numpy().flatten()
            y_pred = (y_score > 0.5).astype(int)
        auc = roc_auc_score(y_test, y_score)
        acc = accuracy_score(y_test, y_pred)

    if plot:
        fpr, tpr, _ = roc_curve(y if classifier_type != "nn" else y_test, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Membership Inference ROC Curve ({classifier_type})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"[MIA: {classifier_type}] AUC = {auc:.4f}, Accuracy = {acc:.4f}")
    return {"roc_auc": auc, "accuracy": acc}

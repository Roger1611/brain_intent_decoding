# %%
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# add repo root to python path
sys.path.append(str(Path.cwd().parents[1]))

# %%
from src.data.bnci_loader import load_bnci_processed

from src.models.deepconvnet import DeepConvNet
from src.models.projection_head import ProjectionHead

from src.training.trainer import Trainer

from src.evaluation.ess import compute_ess
from src.utils.seed import set_seed

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 7, 123]

LAMBDA_ICRR = 0.1

EPOCHS = 25
LR = 1e-3

# %%
DATA_PATH = Path("../../datasets/bnci_dataset/processed/preprocessed_BNCI.npz")

X, y, meta = load_bnci_processed(DATA_PATH)

print("Dataset shape:", X.shape)
print("Metadata:", meta)

chans = meta["n_channels"]
samples = meta["n_times"]
n_classes = len(meta["label_map"])

print("Channels:", chans)
print("Samples:", samples)
print("Classes:", n_classes)

# %%
from pathlib import Path

DATA_PATH = Path("../../datasets/bnci_dataset/processed/preprocessed_BNCI.npz")

X, y, meta = load_bnci_processed(DATA_PATH)

print("Dataset shape:", X.shape)
print("Metadata:", meta)

# correct keys from metadata
chans = meta["n_channels"]
samples = meta["n_times"]
n_classes = len(meta["label_map"])

print("Channels:", chans)
print("Samples:", samples)
print("Classes:", n_classes)

# %%
class DeepConvNetProjection(torch.nn.Module):

    def __init__(self, proj_dim=128):

        super().__init__()

        self.backbone = DeepConvNet(
            chans=chans,
            samples=samples,
            classes=n_classes
        )

        with torch.no_grad():
            dummy = torch.zeros(1, chans, samples)
            _, z = self.backbone(dummy, return_embedding=True)
            emb_dim = z.shape[1]

        self.projection = ProjectionHead(emb_dim, proj_dim=proj_dim)
        self.classifier = torch.nn.Linear(proj_dim, n_classes)

    def forward(self, x, return_embedding=False):

        _, z = self.backbone(x, return_embedding=True)
        z_proj = self.projection(z)
        logits = self.classifier(z_proj)

        if return_embedding:
            return logits, z_proj

        return logits

# %%
def run_experiment(model_builder, lambda_icrr=0.0, batch_size=32):

    accs = []
    esses = []

    for seed in SEEDS:

        print("Running seed:", seed)
        set_seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y
        )

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = model_builder().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=DEVICE,
            lambda_icrr=lambda_icrr
        )

        for _ in range(EPOCHS):
            trainer.train_epoch(train_loader)

        model.eval()
        preds = []
        embeddings = []

        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(DEVICE)
                logits, z = model(xb, return_embedding=True)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                embeddings.append(z.cpu().numpy())

        preds = np.concatenate(preds)
        embeddings = np.concatenate(embeddings)
        acc = (preds == y_test).mean()

        ess = compute_ess(embeddings, y_test)
        print(f"seed={seed} acc={acc:.4f} ess={ess:.4f}")

        accs.append(acc)
        esses.append(ess)

    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_ess": float(np.mean(esses)),
        "std_ess": float(np.std(esses))
    }

# %%
def baseline_model():
    return DeepConvNet(chans=chans, samples=samples, classes=n_classes)


def projection_model():
    return DeepConvNetProjection()


def icrr_model():
    return DeepConvNet(chans=chans, samples=samples, classes=n_classes)


def projection_icrr_model():
    return DeepConvNetProjection()

# %%
results = {}

print("Baseline")
results["baseline"] = run_experiment(
    baseline_model,
    lambda_icrr=0.0
)

print("Projection Only")
results["projection"] = run_experiment(
    projection_model,
    lambda_icrr=0.0
)

print("ICRR Only")
results["icrr"] = run_experiment(
    icrr_model,
    lambda_icrr=LAMBDA_ICRR
)

print("Projection + ICRR")
results["projection_icrr"] = run_experiment(
    projection_icrr_model,
    lambda_icrr=LAMBDA_ICRR
)

results

# %%
for k, v in results.items():

    print("\n", k)

    print("Accuracy:",
          f"{v['mean_acc']:.4f} ± {v['std_acc']:.4f}")

    print("ESS:",
          f"{v['mean_ess']:.4f} ± {v['std_ess']:.4f}")



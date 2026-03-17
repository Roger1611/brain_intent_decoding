import numpy as np
from pathlib import Path


def load_bnci_processed(path: Path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    meta = d["meta"].item()
    return X, y, meta


def load_bnci_all_subjects(path: Path):

    d = np.load(path, allow_pickle=True)
    X           = d["X"].astype(np.float32)
    y           = d["y"].astype(np.int64)
    subject_ids = d["subject_ids"].astype(np.int64)
    meta        = d["meta"].item()
    return X, y, subject_ids, meta
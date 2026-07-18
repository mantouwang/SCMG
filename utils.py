import os
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DATASET_FILES = {
    "CPDB": "CPDB_multiomics.h5",
    "STRING": "STRING_multiomics.h5",
    "IREF": "IRef_multiomics.h5",
    "PCNET": "PCNet_multiomics.h5",
    "PATHNET": "PATHNET_multiomics.h5",
    "GGNET": "GGNET_multiomics.h5",
}


def canonical_dataset(name):
    key = str(name).strip().upper()
    if key not in DATASET_FILES:
        choices = ", ".join(DATASET_FILES)
        raise ValueError(f"Unknown dataset '{name}'. Choose one of: {choices}")
    return key


def dataset_path(data_path, dataset):
    return Path(data_path) / DATASET_FILES[canonical_dataset(dataset)]


def compute_auprc(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def load_bulk_data(data_path, dataset, device):
    h5_path = dataset_path(data_path, dataset)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as handle:
        network = handle["network"][:]
        source, target = np.nonzero(network)
        edge_index = torch.tensor(np.vstack([source, target]), dtype=torch.long)

        if "features/bulk" in handle:
            x = torch.from_numpy(handle["features/bulk"][:]).float()
        elif "features_raw/bulk" in handle:
            x = torch.from_numpy(handle["features_raw/bulk"][:]).float()
        else:
            raise KeyError("Cannot find bulk features in the H5 file")

        y = torch.from_numpy(
            np.logical_or(
                np.logical_or(handle["y_test"][:], handle["y_val"][:]),
                handle["y_train"][:],
            )
        ).int()
        train_mask = torch.from_numpy(handle["mask_train"][:])
        val_mask = torch.from_numpy(handle["mask_val"][:])
        test_mask = torch.from_numpy(handle["mask_test"][:])

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.bulk_dim = int(x.shape[1])
    return data.to(device)


@dataclass
class ScrnaDataset:
    y: np.ndarray
    lognorm_matrices: tuple
    raw_matrices: tuple
    source_masks: tuple


def _decode(values):
    return np.asarray(
        [
            value.decode(errors="replace") if isinstance(value, bytes) else str(value)
            for value in values
        ]
    )


def load_scrna_dataset(h5_path):
    with h5py.File(h5_path, "r") as handle:
        y = handle["labels"][:].reshape(-1).astype(np.float32)
        lognorm_matrices = []
        raw_matrices = []
        source_masks = []
        source_order = None
        for stage in ("normal", "precancer", "cancer"):
            lognorm = handle[f"features/scrna/{stage}"][:].astype(np.float32)
            raw = handle[f"features_raw/scrna/{stage}"][:].astype(np.float32)
            source_labels = _decode(handle[f"cell_metadata/{stage}/source"][:])
            if lognorm.shape != raw.shape or lognorm.shape[1] != source_labels.size:
                raise ValueError(f"scRNA matrix/metadata alignment failed for {stage}")

            current_sources = tuple(dict.fromkeys(source_labels.tolist()))
            if source_order is None:
                source_order = current_sources
            elif set(current_sources) != set(source_order):
                raise ValueError("The scRNA stages do not contain the same sources")

            source_masks.append(
                np.stack(
                    [source_labels == source for source in source_order], axis=0
                ).astype(np.float32)
            )
            lognorm_matrices.append(lognorm)
            raw_matrices.append(raw)

    return ScrnaDataset(
        y,
        tuple(lognorm_matrices),
        tuple(raw_matrices),
        tuple(source_masks),
    )


def scrna_matrix_tensors(dataset, device):
    return {
        "lognorm_matrices": tuple(
            torch.from_numpy(matrix).to(device)
            for matrix in dataset.lognorm_matrices
        ),
        "raw_matrices": tuple(
            torch.from_numpy(matrix).to(device)
            for matrix in dataset.raw_matrices
        ),
        "source_masks": tuple(
            torch.from_numpy(mask).to(device)
            for mask in dataset.source_masks
        ),
    }


def make_folds(data, random_state=42, number=5):
    all_mask = (data.train_mask | data.val_mask | data.test_mask).cpu().numpy()
    labels = data.y.squeeze()[all_mask.squeeze()].cpu().numpy()
    indices = np.arange(all_mask.shape[0])[all_mask.squeeze()]
    splitter = StratifiedKFold(n_splits=number, shuffle=True, random_state=random_state)
    folds = []
    for train_index, test_index in splitter.split(indices, labels):
        train_mask = np.full_like(all_mask, False)
        test_mask = np.full_like(all_mask, False)
        train_mask[indices[train_index]] = True
        test_mask[indices[test_index]] = True
        folds.append((train_mask, test_mask))
    return folds

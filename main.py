import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import RuntimeScRNAFeatureExtractor, SCMG
from utils import (
    compute_auprc,
    dataset_path,
    load_bulk_data,
    load_scrna_dataset,
    make_folds,
    scrna_matrix_tensors,
    set_seed,
)


warnings.filterwarnings("ignore")

DATASETS = ("CPDB", "STRING", "IREF", "PCNET", "GGNET", "PATHNET")
DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "SCMG_DATA_PATH",
        Path(__file__).resolve().parent / "data" / "PPI",
    )
)

# Training configuration used for the released model.
SEED = 42
BULK_EPOCHS = 1200
SCRNA_EPOCHS = 140
FUSION_EPOCHS = 100
BULK_LR = 8e-4
SCRNA_LR = 2.4e-4
FUSION_LR = 1e-3
GRAD_CLIP = 5.0


def _set_trainable(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def _activate_phase(model, phase):
    _set_trainable(model, False)
    if phase == "bulk":
        _set_trainable(model.bulk, True)
    elif phase == "scrna":
        _set_trainable(model.sc_models, True)
    elif phase == "fusion":
        _set_trainable(model.mixer, True)
    else:  # pragma: no cover
        raise ValueError(f"Unknown training phase: {phase}")


def _train_phase(model, phase, epochs, learning_rate, x_bulk, edge_index, sc_batch, y, train_mask, pos_weight):
    _activate_phase(model, phase)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        final_logit, bulk_logit, scrna_logit = model(x_bulk, edge_index, sc_batch)
        if phase == "bulk":
            logits = bulk_logit[train_mask]
        elif phase == "scrna":
            logits = scrna_logit[train_mask]
        else:
            logits = final_logit[train_mask]
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y[train_mask],
            pos_weight=pos_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            GRAD_CLIP,
        )
        optimizer.step()


def train_fold(bulk_data, extracted_scrna, train_mask_np, test_mask_np, fold_index, device):
    set_seed(SEED + fold_index)
    train_mask_cpu = torch.from_numpy(train_mask_np).to(dtype=torch.bool)
    sc_batch = RuntimeScRNAFeatureExtractor().normalize_for_fold(
        extracted_scrna, train_mask_cpu, device=device
    )
    train_mask = train_mask_cpu.to(device=device)
    test_mask = torch.from_numpy(test_mask_np).to(device=device, dtype=torch.bool)
    y = bulk_data.y.view(-1).float()

    positive = y[train_mask].sum()
    negative = train_mask.sum() - positive
    pos_weight = (negative / (positive + 1e-6)).clamp(0.5, 5.0)

    model = SCMG().to(device)
    x_bulk = bulk_data.x.float()
    edge_index = bulk_data.edge_index

    _train_phase(
        model,
        "bulk",
        BULK_EPOCHS,
        BULK_LR,
        x_bulk,
        edge_index,
        sc_batch,
        y,
        train_mask,
        pos_weight,
    )
    _train_phase(
        model,
        "scrna",
        SCRNA_EPOCHS,
        SCRNA_LR,
        x_bulk,
        edge_index,
        sc_batch,
        y,
        train_mask,
        pos_weight,
    )
    _train_phase(
        model,
        "fusion",
        FUSION_EPOCHS,
        FUSION_LR,
        x_bulk,
        edge_index,
        sc_batch,
        y,
        train_mask,
        pos_weight,
    )

    model.eval()
    with torch.no_grad():
        final_logit, _, _ = model(x_bulk, edge_index, sc_batch)
        y_true = y[test_mask].cpu().numpy().astype(int)
        y_score = torch.sigmoid(final_logit[test_mask]).cpu().numpy()
    return compute_auprc(y_true, y_score)


def run(dataset, data_path, device):
    bulk_data = load_bulk_data(data_path, dataset, device)
    h5_path = dataset_path(data_path, dataset)
    scrna_data = load_scrna_dataset(h5_path)

    bulk_labels = bulk_data.y.cpu().numpy().reshape(-1).astype(np.int64)
    scrna_labels = scrna_data.y.reshape(-1).astype(np.int64)
    if bulk_labels.shape != scrna_labels.shape or not np.array_equal(bulk_labels, scrna_labels):
        raise ValueError(f"Bulk/scRNA label alignment failed for {dataset}")

    raw_scrna = scrna_matrix_tensors(scrna_data, torch.device("cpu"))
    extractor = RuntimeScRNAFeatureExtractor()
    extracted_scrna = extractor(raw_scrna)
    del raw_scrna, scrna_data

    fold_aupr = []
    for fold_index, (train_mask, test_mask) in enumerate(make_folds(bulk_data)):
        fold_aupr.append(
            train_fold(
                bulk_data,
                extracted_scrna,
                train_mask,
                test_mask,
                fold_index,
                device,
            )
        )
    return float(np.mean(fold_aupr))


def parse_args():
    parser = argparse.ArgumentParser(description="Train SCMG on one prepared dataset.")
    parser.add_argument("dataset_name", nargs="?", choices=DATASETS)
    parser.add_argument("--dataset", dest="dataset_option", choices=DATASETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset_option or args.dataset_name or "CPDB"
    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )
    mean_aupr = run(dataset, args.data_path, device)
    print(f"Mean Final AUPR: {mean_aupr:.10f}")


if __name__ == "__main__":
    main()

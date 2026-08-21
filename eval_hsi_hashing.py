#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.houston_dataset import HoustonPatchDataset, stratified_split
from vts_hsi_model import VTSHSIModel


def resolve_device(arg_device: str | None = None) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def binary_hash_from_logits(logits):
    return torch.where(logits >= 0, torch.tensor(1), torch.tensor(-1)).cpu().numpy()


def hash_collapse_diagnostic(logits, batch_size=None):
    logits = logits.detach()
    signs = torch.sign(logits)
    binary = torch.where(logits >= 0, torch.ones_like(logits, dtype=torch.int8), -torch.ones_like(logits, dtype=torch.int8))
    mean_per_bit = binary.float().mean(dim=0)
    unique_codes = torch.unique(binary, dim=0).shape[0]
    print("mean per-bit sign:", mean_per_bit.cpu().numpy())
    print("unique codes in batch:", unique_codes)
    if (mean_per_bit.abs() > 0.8).any() or unique_codes < max(2, min(10, logits.shape[0] // 10)):
        print("Warning: collapse diagnostic suggests a degenerate hash code distribution.")
    return binary, signs


def mean_average_precision(query_hash, database_hash, query_labels, database_labels):
    query_hash = query_hash.astype(np.int8)
    database_hash = database_hash.astype(np.int8)
    ap_scores = []
    for i in range(len(query_hash)):
        q = query_hash[i]
        dists = np.count_nonzero(q != database_hash, axis=1)
        order = np.argsort(dists)
        retrieved_labels = database_labels[order]
        q_label = query_labels[i]
        hits = np.where(retrieved_labels == q_label)[0]
        if len(hits) == 0:
            ap_scores.append(0.0)
            continue
        precisions = []
        for rank_pos, idx in enumerate(hits + 1, start=1):
            precisions.append(np.mean(retrieved_labels[:idx] == q_label))
        ap_scores.append(np.sum(precisions) / max(len(hits), 1))
    return float(np.mean(ap_scores))


def collect_hashes(model, loader, device):
    model.eval()
    hashes = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            logits = model(images)
            bits = torch.where(logits >= 0, torch.tensor(1), torch.tensor(-1)).cpu().numpy()
            hashes.append(bits)
            labels.append(batch_labels.numpy())
    return np.concatenate(hashes), np.concatenate(labels)


def evaluate(args):
    args.device = str(resolve_device(args.device))
    dataset = HoustonPatchDataset(args.dataset_root, scene=args.scene, patch_size=args.patch_size)
    split = stratified_split(dataset.labels, train_fraction=0.5, query_per_class=100)

    model = VTSHSIModel(
        num_tokens=args.num_tokens,
        token_dim=args.token_dim,
        feature_channels=args.feature_channels,
        output_dim=args.output_dim,
        spectral_pooling=args.spectral_pooling,
        use_all_tokens=args.use_all_tokens,
        hash_bit_length=args.hash_bit_length,
        vit_model_name=args.vit_model_name,
        input_channels=args.input_channels,
        dropout=args.dropout,
    ).to(args.device)

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(state)
    else:
        print(f"Warning: checkpoint not found at {checkpoint}; using untrained model for evaluation.")

    query_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, split["query"].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    db_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, split["database"].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    q_hash, q_labels = collect_hashes(model, query_loader, args.device)
    db_hash, db_labels = collect_hashes(model, db_loader, args.device)
    mAP = mean_average_precision(q_hash, db_hash, q_labels, db_labels)
    print(f"mAP@{args.hash_bit_length}: {mAP:.4f}")
    return mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HSI VTS hashing model")
    parser.add_argument("--dataset_root", type=str, default="data")
    parser.add_argument("--scene", type=str, default="Houston2013")
    parser.add_argument("--patch_size", type=int, default=13)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--feature_channels", type=int, default=8)
    parser.add_argument("--output_dim", type=int, default=768)
    parser.add_argument("--spectral_pooling", type=str, choices=["mean", "attention"], default="attention")
    parser.add_argument("--use_all_tokens", action="store_true", default=True)
    parser.add_argument("--cls_only", dest="use_all_tokens", action="store_false")
    parser.add_argument("--hash_bit_length", type=int, default=32)
    parser.add_argument("--vit_model_name", type=str, default="ViT-B_16")
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    evaluate(args)

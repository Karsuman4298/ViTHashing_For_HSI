#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.houston_dataset import HoustonPatchDataset, stratified_split
from eval_hsi_hashing import binary_hash_from_logits, hash_collapse_diagnostic, mean_average_precision
from losses.csq import CSQLoss
from losses.dpn import DPNLoss
from vts_hsi_model import VTSHSIModel


def resolve_device(arg_device: str | None = None) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_band_stats(dataset: HoustonPatchDataset, indices):
    train_positions = np.asarray(indices, dtype=np.int64)
    if len(train_positions) == 0:
        raise ValueError("Train split is empty; cannot compute per-band normalization stats.")
    values = []
    for idx in train_positions:
        patch, _ = dataset[idx]
        values.append(patch.numpy())
    samples = np.stack(values, axis=0)
    mean = samples.mean(axis=(0, 2, 3))
    std = samples.std(axis=(0, 2, 3))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_dataset(dataset_root: str, scene: str, patch_size: int = 13):
    dataset = HoustonPatchDataset(dataset_root, scene=scene, patch_size=patch_size)
    labels = dataset.labels
    split = stratified_split(labels, train_fraction=0.5, query_per_class=100)
    band_mean, band_std = compute_band_stats(dataset, split["train"])
    dataset.set_band_stats(band_mean, band_std)
    return dataset, split


def make_loader(dataset: HoustonPatchDataset, indices, batch_size: int, shuffle: bool = False):
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_model(model, dataset, split, device, batch_size):
    model.eval()
    q_loader = make_loader(dataset, split["query"], batch_size, shuffle=False)
    db_loader = make_loader(dataset, split["database"], batch_size, shuffle=False)
    q_hash = []
    q_labels = []
    db_hash = []
    db_labels = []

    with torch.no_grad():
        for loader, hash_list, label_list in [(q_loader, q_hash, q_labels), (db_loader, db_hash, db_labels)]:
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                bits = torch.where(logits >= 0, torch.ones_like(logits, dtype=torch.int8), -torch.ones_like(logits, dtype=torch.int8)).to(device)
                hash_list.append(bits.cpu().numpy())
                label_list.append(labels.numpy())
    q_hash = np.concatenate(q_hash)
    q_labels = np.concatenate(q_labels)
    db_hash = np.concatenate(db_hash)
    db_labels = np.concatenate(db_labels)
    print("hash collapse diagnostic on database batch:")
    sample_logits = model(next(iter(db_loader))[0].to(device))
    hash_collapse_diagnostic(sample_logits)
    mAP = mean_average_precision(q_hash, db_hash, q_labels, db_labels)
    print(f"mAP@{len(q_hash[0])}: {mAP:.4f}")
    return mAP


def run_training(args):
    dataset_root = Path(args.dataset_root)
    dataset, split = build_dataset(str(dataset_root), args.scene, patch_size=args.patch_size)

    train_loader = make_loader(dataset, split["train"], args.batch_size, shuffle=True)
    query_loader = make_loader(dataset, split["query"], args.batch_size, shuffle=False)
    db_loader = make_loader(dataset, split["database"], args.batch_size, shuffle=False)

    args.device = str(resolve_device(args.device))
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.loss == "csq":
        criterion = CSQLoss(args.hash_bit_length, num_classes=int(dataset.labels.max()) + 1, bit_balance_weight=0.5, diversity_weight=0.5)
    elif args.loss == "dpn":
        criterion = DPNLoss(args.hash_bit_length)
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            logits = model(images)
            if args.loss == "csq":
                loss = criterion(logits, labels)
            else:
                loss = criterion(logits)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        if epoch % args.eval_every == 0 or epoch == 1:
            print(f"epoch={epoch:03d} loss={running_loss / len(train_loader.dataset):.4f}")

    save_dir = Path(args.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"{args.scene}_{args.loss}_{args.hash_bit_length}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "band_mean": dataset.band_mean,
        "band_std": dataset.band_std,
        "scene": args.scene,
        "hash_bit_length": args.hash_bit_length,
        "spectral_pooling": args.spectral_pooling,
        "use_all_tokens": args.use_all_tokens,
        "loss": args.loss,
    }, checkpoint_path)
    print(f"saved model to {checkpoint_path}")
    mAP = evaluate_model(model, dataset, split, args.device, args.batch_size)
    print("training complete")
    print({
        "scene": args.scene,
        "hash_bit_length": args.hash_bit_length,
        "spectral_pooling": args.spectral_pooling,
        "use_all_tokens": args.use_all_tokens,
        "loss": args.loss,
        "checkpoint": str(checkpoint_path),
        "mAP": mAP,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HSI VTS hashing model")
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
    parser.add_argument("--loss", type=str, choices=["csq", "dpn"], default="csq")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval_every", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    run_training(args)

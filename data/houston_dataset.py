from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _split_sanity_check(valid_positions, train_idx, query_idx, database_idx, min_radius=13):
    if len(query_idx) == 0:
        return True
    query_positions = valid_positions[query_idx]
    train_positions = valid_positions[train_idx]
    db_positions = valid_positions[database_idx]

    def min_distance(pos, other_positions):
        if len(other_positions) == 0:
            return np.inf
        dy = other_positions[:, 0] - pos[0]
        dx = other_positions[:, 1] - pos[1]
        return np.sqrt(dy * dy + dx * dx).min()

    for q in query_positions:
        min_train = min_distance(q, train_positions)
        min_db = min_distance(q, db_positions)
        if min_train <= min_radius or min_db <= min_radius:
            return False
    return True


HOUSTON_SCENES = {
    "Houston13": {"bands": 48, "classes": 7},
    "Houston18": {"bands": 48, "classes": 7},
}


class HoustonPatchDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, os.PathLike],
        scene: str = "Houston13",
        patch_size: int = 13,
        bands: Optional[int] = None,
        transform=None,
        band_mean: Optional[np.ndarray] = None,
        band_std: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.scene = scene
        self.patch_size = patch_size
        self.transform = transform
        self.band_mean = band_mean
        self.band_std = band_std
        self.eps = eps

        self.hsi = None
        self.labels = None
        self._load_scene(self.root_dir, bands=bands)
        if self.hsi.shape[1:] != self.labels.shape:
            raise ValueError(
                f"HSI spatial size {self.hsi.shape[1:]} does not match label map {self.labels.shape}"
            )

    def set_band_stats(self, band_mean: np.ndarray, band_std: np.ndarray):
        self.band_mean = np.asarray(band_mean, dtype=np.float32)
        self.band_std = np.asarray(band_std, dtype=np.float32)

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        if self.band_mean is None or self.band_std is None:
            return patch
        patch = patch.astype(np.float32, copy=False)
        if patch.shape[0] != self.band_mean.shape[0]:
            return patch
        return (patch - self.band_mean[:, None, None]) / (self.band_std[:, None, None] + self.eps)

    def _load_scene(self, root_dir: Path, bands: Optional[int] = None):
        scene_name = self.scene
        if scene_name not in HOUSTON_SCENES:
            raise ValueError(f"Unsupported Houston scene {scene_name}; expected one of {sorted(HOUSTON_SCENES)}")

        data_file = root_dir / "Houston" / f"{scene_name}.mat"
        label_file = root_dir / "Houston" / f"{scene_name}_7gt.mat"

        if not data_file.exists():
            raise FileNotFoundError(f"HSI scene file not found: {data_file}")
        if not label_file.exists():
            raise FileNotFoundError(f"GT scene file not found: {label_file}")

        with h5py.File(str(data_file), "r") as f:
            data = np.asarray(f["ori_data"], dtype=np.float32)
        with h5py.File(str(label_file), "r") as f:
            labels = np.asarray(f["map"], dtype=np.int64)

        if data.shape[0] != 48:
            data = np.transpose(data, (2, 0, 1)) if data.ndim == 3 and data.shape[0] != 48 else data

        if bands is not None and data.shape[0] != bands:
            raise ValueError(f"Requested bands={bands}, but HSI cube has {data.shape[0]} bands")

        self.hsi = data
        self.labels = labels

    def __len__(self):
        return int(np.sum(self.labels > 0))

    def __getitem__(self, idx):
        valid_positions = np.argwhere(self.labels > 0)
        y, x = valid_positions[idx]
        half = self.patch_size // 2
        y0 = max(0, y - half)
        y1 = min(self.hsi.shape[1], y + half + 1)
        x0 = max(0, x - half)
        x1 = min(self.hsi.shape[2], x + half + 1)

        patch = self.hsi[:, y0:y1, x0:x1]
        if patch.shape[1] != self.patch_size or patch.shape[2] != self.patch_size:
            pad_y0 = max(0, half - y)
            pad_y1 = max(0, (y + half + 1) - self.hsi.shape[1])
            pad_x0 = max(0, half - x)
            pad_x1 = max(0, (x + half + 1) - self.hsi.shape[2])
            patch = np.pad(
                patch,
                ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)),
                mode="reflect",
            )

        label = int(self.labels[y, x])
        patch = self._normalize_patch(patch)
        patch = torch.from_numpy(patch).float()
        if self.transform is not None:
            patch = self.transform(patch)
        return patch, label


def stratified_split(labels: Sequence[int], train_fraction: float = 0.5, query_per_class: int = 100, block_size: int = 32):
    labels = np.asarray(labels)
    valid_positions = np.argwhere(labels > 0)
    if len(valid_positions) == 0:
        return {"train": np.asarray([], dtype=np.int64), "query": np.asarray([], dtype=np.int64), "database": np.asarray([], dtype=np.int64), "classes": np.asarray([], dtype=np.int64)}

    width = labels.shape[1]
    train_max = min(60, width)
    query_min = max(80, min(width - 80, int(width * 0.38)))
    query_max = min(130, width)
    db_min = max(150, min(width, int(width * 0.70)))

    split_map = {"train": [], "query": [], "database": []}
    for idx, (y, x) in enumerate(valid_positions):
        if x < train_max:
            split_map["train"].append(idx)
        elif query_min <= x < query_max:
            split_map["query"].append(idx)
        elif x >= db_min:
            split_map["database"].append(idx)

    train_idx = np.asarray(split_map["train"], dtype=np.int64)
    query_idx = np.asarray(split_map["query"], dtype=np.int64)
    database_idx = np.asarray(split_map["database"], dtype=np.int64)

    if len(train_idx) > 0 and len(query_idx) > 0 and len(database_idx) > 0:
        if _split_sanity_check(valid_positions, train_idx, query_idx, database_idx, min_radius=13):
            valid_labels = labels[valid_positions[:, 0], valid_positions[:, 1]]
            classes = np.unique(valid_labels)
            return {
                "train": train_idx,
                "query": query_idx,
                "database": database_idx,
                "classes": classes,
            }

    raise RuntimeError("Unable to construct a spatially disjoint Houston split with the requested block-size margin.")


__all__ = ["HoustonPatchDataset", "HOUSTON_SCENES", "stratified_split"]

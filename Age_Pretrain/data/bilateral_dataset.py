# data/bilateral_dataset.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

from monai.data import PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped,
    ToTensord,
)

REQUIRED_COLUMNS = ("LEFT", "RIGHT", "age")


def _resolve_image_path(raw_path: str, root_dir: Path | None) -> Path:
    path = Path(str(raw_path).strip()).expanduser()
    if root_dir and not path.is_absolute():
        path = root_dir / path
    return path


def _validate_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_cols)}")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError("No valid rows remaining after dropping NaNs in required columns")
    return df


def _build_transforms(keys_images: Sequence[str], spatial_size: Tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=keys_images),
            EnsureChannelFirstd(keys=keys_images),
            #Orientationd(keys=keys_images, axcodes="RAS"),
            ScaleIntensityRanged(
                keys=keys_images,
                a_min=-1024,
                a_max=1024,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys=keys_images, spatial_size=spatial_size, mode="trilinear"),
            EnsureTyped(keys=keys_images),  # keep images as MetaTensor
            ToTensord(keys=["age"]),  # age as plain Tensor
        ]
    )


def _prepare_records(
    df: pd.DataFrame,
    root_path: Path | None,
    validate_files: bool,
) -> list[dict[str, float | str]]:
    data_list: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        left_path = _resolve_image_path(row["LEFT"], root_path)
        right_path = _resolve_image_path(row["RIGHT"], root_path)
        if validate_files and (not left_path.is_file() or not right_path.is_file()):
            continue
        record = {
            "LEFT": str(left_path),
            "RIGHT": str(right_path),
            "age": float(row["age"]),
        }
        data_list.append(record)

    if validate_files and not data_list:
        raise ValueError("No usable samples remain after dropping missing image files")
    return data_list


def create_datasets_from_csv(
    train_csv_path: str | os.PathLike[str],
    val_csv_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    root_dir: str | os.PathLike[str] | None = None,
    spatial_size: Sequence[int] = (256, 256, 64),
    validate_files: bool = True,
) -> Tuple[PersistentDataset, PersistentDataset]:
    """
    Build ``PersistentDataset`` instances for train and validation splits from explicit CSV files.

    Args:
        train_csv_path: CSV with ``LEFT``, ``RIGHT`` and ``age`` columns for training data.
        val_csv_path: CSV with ``LEFT``, ``RIGHT`` and ``age`` columns for validation data.
        cache_dir: Directory where MONAI ``PersistentDataset`` caches will be stored.
        root_dir: Optional directory prepended to relative paths in the CSV.
        spatial_size: Target spatial size passed to ``ResizeD``.
        validate_files: When ``True`` (default) drops rows referencing missing image files.
    """

    train_csv = Path(train_csv_path)
    val_csv = Path(val_csv_path)
    train_df = _validate_dataframe(train_csv)
    val_df = _validate_dataframe(val_csv)

    spatial_size = tuple(int(dim) for dim in spatial_size)
    if len(spatial_size) != 3:
        raise ValueError("spatial_size must define exactly three dimensions")

    root_path = Path(root_dir).expanduser() if root_dir else None
    train_records = _prepare_records(train_df, root_path, validate_files)
    val_records = _prepare_records(val_df, root_path, validate_files)
    if len(train_records) < 1:
        raise ValueError("Training CSV produced no usable samples")
    if len(val_records) < 1:
        raise ValueError("Validation CSV produced no usable samples")

    keys_images = ("LEFT", "RIGHT")
    train_transforms = _build_transforms(keys_images, spatial_size)
    val_transforms = _build_transforms(keys_images, spatial_size)

    cache_path = Path(cache_dir)
    train_cache_dir = cache_path / "train"
    val_cache_dir = cache_path / "val"
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PersistentDataset(
        data=train_records,
        transform=train_transforms,
        cache_dir=str(train_cache_dir),
    )

    val_ds = PersistentDataset(
        data=val_records,
        transform=val_transforms,
        cache_dir=str(val_cache_dir),
    )

    return train_ds, val_ds


def create_validation_dataset_from_csv(
    csv_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    root_dir: str | os.PathLike[str] | None = None,
    spatial_size: Sequence[int] = (256, 256, 64),
    validate_files: bool = True,
    cache_subdir: str = "val_infer",
) -> PersistentDataset:
    """
    Build a ``PersistentDataset`` from a CSV that already represents the validation split.

    Args:
        csv_path: CSV with ``LEFT``, ``RIGHT`` and ``age`` columns for validation data.
        cache_dir: Directory where MONAI ``PersistentDataset`` caches will be stored.
        root_dir: Optional directory prepended to relative paths in the CSV.
        spatial_size: Target spatial size passed to ``ResizeD``.
        validate_files: When ``True`` (default) drops rows referencing missing image files.
        cache_subdir: Folder name inside ``cache_dir`` dedicated to this dataset.
    """

    csv_path = Path(csv_path)
    df = _validate_dataframe(csv_path)
    spatial_size = tuple(int(dim) for dim in spatial_size)
    if len(spatial_size) != 3:
        raise ValueError("spatial_size must define exactly three dimensions")

    root_path = Path(root_dir).expanduser() if root_dir else None
    data_list = _prepare_records(df, root_path, validate_files)
    if not data_list:
        raise ValueError("Validation CSV produced no usable samples")

    transforms = _build_transforms(("LEFT", "RIGHT"), spatial_size)
    cache_path = Path(cache_dir) / cache_subdir
    cache_path.mkdir(parents=True, exist_ok=True)

    return PersistentDataset(
        data=data_list,
        transform=transforms,
        cache_dir=str(cache_path),
    )

# data/unilateral_dataset.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple, Union

import pandas as pd

from monai.data import PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped,
    ToTensord,
)


REQUIRED_COLUMNS = ("Path", "Label")


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


def _build_transforms(spatial_size: Tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=1024,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
            EnsureTyped(keys=["image"]),
            ToTensord(keys=["label"]),
        ]
    )


def _prepare_records(
    df: pd.DataFrame,
    root_path: Path | None,
    validate_files: bool,
) -> list[dict[str, Union[float, str]]]:
    """
    转换 DataFrame 为字典列表: [{"image": path, "label": int}, ...]
    """
    data_list: list[dict[str, Union[float, str]]] = []
    
    for _, row in df.iterrows():
        img_path = _resolve_image_path(row["Path"], root_path)
        
        if validate_files and not img_path.is_file():
            continue
            
        record = {
            "image": str(img_path),
            "label": int(row["Label"]),
        }
        data_list.append(record)

    if validate_files and not data_list:
        raise ValueError("No usable samples remain after dropping missing image files")
    
    return data_list


def create_unilateral_datasets(
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: str | None = None,
    cache_dir: str = "./cache",
    root_dir: str | None = None,
    spatial_size: Sequence[int] = (256, 256, 64),
) -> Tuple:

    train_df = _validate_dataframe(Path(train_csv_path))
    val_df = _validate_dataframe(Path(val_csv_path))

    root_path = Path(root_dir).expanduser() if root_dir else None
    
    train_records = _prepare_records(train_df, root_path, validate_files=True)
    val_records = _prepare_records(val_df, root_path, validate_files=True)

    spatial_size = tuple(int(dim) for dim in spatial_size)
    transforms = _build_transforms(spatial_size)

    cache_path = Path(cache_dir)
    train_cache = cache_path / "train_uni"
    val_cache = cache_path / "val_uni"
    
    train_cache.mkdir(parents=True, exist_ok=True)
    val_cache.mkdir(parents=True, exist_ok=True)

    train_ds = PersistentDataset(
        data=train_records,
        transform=transforms,
        cache_dir=str(train_cache),
    )

    val_ds = PersistentDataset(
        data=val_records,
        transform=transforms,
        cache_dir=str(val_cache),
    )

    if test_csv_path:
        test_df = _validate_dataframe(Path(test_csv_path))
        test_records = _prepare_records(test_df, root_path, validate_files=True)
        
        test_cache = cache_path / "test_uni"
        test_cache.mkdir(parents=True, exist_ok=True)
        
        test_ds = PersistentDataset(
            data=test_records,
            transform=transforms,
            cache_dir=str(test_cache),
        )
        return train_ds, val_ds, test_ds

    return train_ds, val_ds

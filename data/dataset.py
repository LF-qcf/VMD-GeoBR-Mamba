import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def find_date_column(df: pd.DataFrame):
    for name in ("Date", "date", "time", "Time", "datetime", "Datetime"):
        if name in df.columns:
            return name
    return None


def load_frame(csv_path: str, column: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found. Available columns: {list(df.columns)}")

    date_col = find_date_column(df)
    if date_col is None:
        df["date"] = pd.date_range("2000-01-01", periods=len(df), freq="min")
        date_col = "date"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[[date_col, column]].dropna()
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date", column: "imf3"})
    return df


class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = float(mean) if mean is not None else None
        self.std = float(std) if std is not None else None

    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        self.mean = float(np.nanmean(values))
        self.std = float(np.nanstd(values))
        if self.std < 1e-6 or not np.isfinite(self.std):
            self.std = 1.0

    def transform(self, values):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler mean/std are not initialized.")
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler mean/std are not initialized.")
        return values * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean, "std": self.std}


def split_indices(n, train_ratio=0.7, val_ratio=0.1):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def downsample_df(df, max_points):
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].copy()


class IMF3WindowDataset(Dataset):
    def __init__(self, values_norm, values_raw, dates, start, end, seq_len, pred_len):
        self.values_norm = np.asarray(values_norm, dtype=np.float32)
        self.values_raw = np.asarray(values_raw, dtype=np.float32)
        self.dates = np.asarray(dates)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.starts = []

        last_start = int(end) - self.seq_len - self.pred_len
        for idx in range(int(start), last_start + 1):
            self.starts.append(idx)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        s = self.starts[item]
        x_begin = s
        x_end = s + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len

        x_norm = self.values_norm[x_begin:x_end, None]
        y_norm = self.values_norm[y_begin:y_end]
        x_raw = self.values_raw[x_begin:x_end]
        y_raw = self.values_raw[y_begin:y_end]

        return (
            torch.from_numpy(x_norm).float(),
            torch.from_numpy(y_norm).float(),
            torch.from_numpy(x_raw).float(),
            torch.from_numpy(y_raw).float(),
            torch.tensor(s, dtype=torch.long),
        )

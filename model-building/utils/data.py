from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


class SplitData(NamedTuple):
    X_train:    np.ndarray
    y_train:    np.ndarray
    X_val:      np.ndarray
    y_val:      np.ndarray
    X_test:     np.ndarray
    y_test:     np.ndarray
    df_test:    pd.DataFrame
    val_start:  pd.Timestamp
    test_start: pd.Timestamp


def train_val_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "inflation_log",
) -> SplitData:
    max_date   = df["date"].max()
    test_start = max_date - pd.DateOffset(years=1) + pd.DateOffset(months=1)
    val_start  = test_start - pd.DateOffset(years=1)

    df_train = df[df["date"] < val_start]
    df_val   = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    df_test  = df[df["date"] >= test_start]

    print(f"Train:      {df_train['date'].min().date()} → {df_train['date'].max().date()}  ({len(df_train)} obs)")
    print(f"Validación: {df_val['date'].min().date()} → {df_val['date'].max().date()}  ({len(df_val)} obs)")
    print(f"Test:       {df_test['date'].min().date()} → {df_test['date'].max().date()}  ({len(df_test)} obs)")

    return SplitData(
        X_train    = df_train[feature_cols].values,
        y_train    = df_train[target_col].values,
        X_val      = df_val[feature_cols].values,
        y_val      = df_val[target_col].values,
        X_test     = df_test[feature_cols].values,
        y_test     = df_test[target_col].values,
        df_test    = df_test,
        val_start  = val_start,
        test_start = test_start,
    )


def load_food_inflation_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["TIME_PERIOD"])
    df = df.rename(columns={
        "REF_AREA":       "country",
        "REF_AREA_LABEL": "country_name",
        "TIME_PERIOD":    "date",
        "OBS_VALUE":      "inflation",
    })
    return df.sort_values(["country", "date"]).reset_index(drop=True)

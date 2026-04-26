import numpy as np
import pandas as pd
import joblib
import wandb
import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
_constants_spec = importlib.util.spec_from_file_location("root_constants", ROOT_DIR / "constants.py")
_constants_module = importlib.util.module_from_spec(_constants_spec)
assert _constants_spec is not None and _constants_spec.loader is not None
_constants_spec.loader.exec_module(_constants_module)
WANDB_PRODUCTION_ARTIFACT = _constants_module.WANDB_PRODUCTION_ARTIFACT
WANDB_PROJECT = _constants_module.WANDB_PROJECT
FOOD_INFLATION_CSV_FILENAME = _constants_module.FOOD_INFLATION_CSV_FILENAME

MODEL_DIR      = ROOT_DIR / "dev"
CSV_PATH       = ROOT_DIR / "model-building" / FOOD_INFLATION_CSV_FILENAME
ARTIFACT_NAME = WANDB_PRODUCTION_ARTIFACT

_model    = None
_le       = None
_features = None
_df_raw   = None
_df_feat  = None


def _build_features(df: pd.DataFrame, le) -> pd.DataFrame:
    df = df.copy()
    df["country_enc"] = le.transform(df["country"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    for lag in [1, 2, 3, 6, 12]:
        df[f"inflation_lag{lag}"] = df.groupby("country")["inflation"].shift(lag)
    df["rolling_mean_3"]  = df.groupby("country")["inflation"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df["rolling_mean_12"] = df.groupby("country")["inflation"].transform(
        lambda x: x.shift(1).rolling(12).mean()
    )
    df["yoy_change"] = df["inflation"] - df.groupby("country")["inflation"].shift(12)
    return df.dropna().reset_index(drop=True)


def is_ready() -> bool:
    return _model is not None


def get_country_options() -> list[dict]:
    if _df_raw is None:
        return []
    mapping = (
        _df_raw[["country", "country_name"]]
        .drop_duplicates()
        .sort_values("country_name")
    )
    return [{"code": r.country, "name": r.country_name} for r in mapping.itertuples()]


def load_model():
    global _model, _le, _features, _df_raw, _df_feat

    api      = wandb.Api()
    artifact = api.artifact(f"{api.default_entity}/{WANDB_PROJECT}/{ARTIFACT_NAME}")
    artifact_dir = artifact.download(root=str(MODEL_DIR))

    bundle    = joblib.load(Path(artifact_dir) / "model.pkl")
    _model    = bundle["model"]
    _le       = bundle["label_encoder"]
    _features = bundle["features"]

    df = pd.read_csv(CSV_PATH, parse_dates=["TIME_PERIOD"])
    df = df.rename(columns={
        "REF_AREA":       "country",
        "REF_AREA_LABEL": "country_name",
        "TIME_PERIOD":    "date",
        "OBS_VALUE":      "inflation",
    })
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    _df_raw  = df
    _df_feat = _build_features(df, _le)

    print(f"Model loaded — {len(_le.classes_)} countries, "
          f"data up to {_df_raw['date'].max().date()}")


def predict(country: str, year: int, month: int) -> float | None:
    if _model is None or country not in _le.classes_:
        return None

    target_date = pd.Timestamp(year=year, month=month, day=1)
    country_enc = int(_le.transform([country])[0])
    country_df  = _df_raw[_df_raw["country"] == country].sort_values("date")

    if country_df.empty:
        return None

    last_date = country_df["date"].max()

    if target_date <= last_date:
        row = _df_feat[(_df_feat["country"] == country) & (_df_feat["date"] == target_date)]
        if row.empty:
            return None
        pred = float(np.expm1(_model.predict(row[_features].values)[0]))
        return round(pred, 2)

    # Recursive future prediction
    recent = list(country_df["inflation"].tail(12).values)
    cursor = last_date + pd.DateOffset(months=1)
    while cursor <= target_date:
        n     = len(recent)
        lag1  = recent[-1]  if n >= 1  else 0
        lag2  = recent[-2]  if n >= 2  else 0
        lag3  = recent[-3]  if n >= 3  else 0
        lag6  = recent[-6]  if n >= 6  else 0
        lag12 = recent[-12] if n >= 12 else 0
        rm3   = np.mean(recent[-3:])  if n >= 3  else np.mean(recent)
        rm12  = np.mean(recent[-12:]) if n >= 12 else np.mean(recent)
        yoy   = recent[-1] - recent[-12] if n >= 12 else 0

        X    = np.array([[country_enc, cursor.year, cursor.month,
                          lag1, lag2, lag3, lag6, lag12, rm3, rm12, yoy]])
        pred = float(np.expm1(_model.predict(X)[0]))
        recent.append(pred)
        cursor += pd.DateOffset(months=1)

    return round(recent[-1], 2)

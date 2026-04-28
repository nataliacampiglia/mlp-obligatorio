import sys
import joblib
import pandas as pd
import wandb
from pathlib import Path


ROOT_DIR = Path(__file__).parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.constants import WANDB_PRODUCTION_ARTIFACT, WANDB_PROJECT
import core.predictor
from core.predictor import InflationPredictor
from monitoring.logger import log_prediction

# Pickle stores the class path at save time. Models saved before the move to core/
# reference 'predictor' as the module — this alias lets them deserialize correctly.
sys.modules.setdefault("predictor", core.predictor)

MODEL_DIR     = ROOT_DIR / "dev"
ARTIFACT_NAME = WANDB_PRODUCTION_ARTIFACT

_predictor: InflationPredictor | None = None


def is_ready() -> bool:
    return _predictor is not None


def get_country_options() -> list[dict]:
    if _predictor is None:
        return []
    return _predictor.countries()


def load_model():
    global _predictor

    api          = wandb.Api()
    artifact     = api.artifact(f"{api.default_entity}/{WANDB_PROJECT}/{ARTIFACT_NAME}")
    artifact_dir = artifact.download(root=str(MODEL_DIR))

    _predictor = joblib.load(Path(artifact_dir) / "model.pkl")
    print(f"Predictor loaded — {type(_predictor).__name__}, {len(_predictor.countries())} countries")


def predict(country: str, year: int, month: int) -> float | None:
    if _predictor is None:
        return None
    value = _predictor.predict(country, year, month)

    last_date = _predictor.last_date(country)
    last_value = _predictor.last_value(country)
    target = pd.Timestamp(year=year, month=month, day=1)

    months_from_last = None
    is_future = None
    if last_date is not None:
        months_from_last = (target.year - last_date.year) * 12 + (target.month - last_date.month)
        is_future = target > last_date

    log_prediction(
        country=country,
        year=year,
        month=month,
        prediction=value,
        model_version=ARTIFACT_NAME,
        last_known_date=last_date.strftime("%Y-%m-%d") if last_date is not None else None,
        last_known_value=last_value,
        months_from_last_known=months_from_last,
        country_in_training=last_date is not None,
        is_future_prediction=is_future,
    )
    return value

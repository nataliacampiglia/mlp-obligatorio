import sys
import numpy as np
import pandas as pd
import joblib
import wandb
from pathlib import Path


ROOT_DIR = Path(__file__).parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.constants import WANDB_PRODUCTION_ARTIFACT, WANDB_PROJECT
import core.predictor 
from core.predictor import InflationPredictor

# Pickle stores the class path at save time. Models saved before the move to core/
# reference 'predictor' as the module — this alias lets them deserialize correctly.
sys.modules.setdefault("predictor", core.predictor)

MODEL_DIR     = ROOT_DIR / "dev"
ARTIFACT_NAME = WANDB_PRODUCTION_ARTIFACT

_predictor: InflationPredictor | None = None
_version  = None


def is_ready() -> bool:
    return _predictor is not None


def get_model_version() -> str | None:
    return _version


def get_country_options() -> list[dict]:
    if _predictor is None:
        return []
    return _predictor.countries()


def load_model():
    global _predictor, _version

    api          = wandb.Api()
    artifact     = api.artifact(f"{api.default_entity}/{WANDB_PROJECT}/{ARTIFACT_NAME}")
    artifact_dir = artifact.download(root=str(MODEL_DIR))
    _version = artifact.version

    _predictor = joblib.load(Path(artifact_dir) / "model.pkl")
    print(f"Predictor loaded — {type(_predictor).__name__}, {len(_predictor.countries())} countries")


def predict(country: str, year: int, month: int) -> float | None:
    if _predictor is None:
        return None
    return _predictor.predict(country, year, month)

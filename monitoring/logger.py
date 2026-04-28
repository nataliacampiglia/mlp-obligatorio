import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

LOGS_DIR = Path(__file__).parent / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

_lock = Lock()


def log_prediction(
    country: str,
    year: int,
    month: int,
    prediction: float | None,
    model_version: str,
    last_known_date: str | None = None,
    last_known_value: float | None = None,
    months_from_last_known: int | None = None,
    country_in_training: bool | None = None,
    is_future_prediction: bool | None = None,
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "country": country,
        "year": year,
        "month": month,
        "prediction": prediction,
        "last_known_date": last_known_date,
        "last_known_value": last_known_value,
        "months_from_last_known": months_from_last_known,
        "country_in_training": country_in_training,
        "is_future_prediction": is_future_prediction,
    }
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with PREDICTIONS_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

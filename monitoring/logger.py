import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

LOGS_DIR = Path(__file__).parent / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

_lock = Lock()


def log_prediction(
    country: str,
    year: int,
    month: int,
    prediction: float | None,
    model_version: str,
    extra: dict[str, Any] | None = None,
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "country": country,
        "year": year,
        "month": month,
        "prediction": prediction,
    }
    if extra:
        record["extra"] = extra
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with PREDICTIONS_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

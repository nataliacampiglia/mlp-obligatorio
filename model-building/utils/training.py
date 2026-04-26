from typing import NamedTuple

import numpy as np
import wandb

from .metrics import compute_regression_metrics


class TrainResult(NamedTuple):
    test_metrics: dict
    mae:          float
    preds_test:   np.ndarray
    y_test_orig:  np.ndarray
    score:        float | None


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    log_target: bool = True,
    oob_score:  bool = False,
) -> TrainResult:
    model.fit(X_train, y_train)

    inv = np.expm1 if log_target else (lambda x: x)

    y_val_orig  = inv(y_val)
    preds_val   = inv(model.predict(X_val))
    val_metrics = compute_regression_metrics(y_val_orig, preds_val)
    print("--- Validación ---")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    score = None
    if oob_score:
        score = model.oob_score_
        print(f"  oob_score: {score:.4f}")

    wandb.log(
        {f"val/{k}": v for k, v in val_metrics.items()}
        | ({"val/oob_score": score} if score is not None else {})
    )

    y_test_orig  = inv(y_test)
    preds_test   = inv(model.predict(X_test))
    test_metrics = compute_regression_metrics(y_test_orig, preds_test)
    print("\n--- Test (último año) ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    return TrainResult(
        test_metrics = test_metrics,
        mae          = test_metrics["mae"],
        preds_test   = preds_test,
        y_test_orig  = y_test_orig,
        score        = score,
    )

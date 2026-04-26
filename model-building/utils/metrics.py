import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae":     float(mean_absolute_error(y_true, y_pred)),
        "rmse":    float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":      float(r2_score(y_true, y_pred)),
        "acc_1pp": float(np.mean(np.abs(y_pred - y_true) <= 1)),
        "acc_2pp": float(np.mean(np.abs(y_pred - y_true) <= 2)),
        "acc_5pp": float(np.mean(np.abs(y_pred - y_true) <= 5)),
    }

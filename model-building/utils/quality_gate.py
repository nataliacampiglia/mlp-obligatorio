import json

import wandb


def _get_test_mae(run) -> float | None:
    try:
        summary = run.summary_metrics
        if isinstance(summary, str):
            summary = json.loads(summary)
        if not isinstance(summary, dict):
            summary = getattr(run.summary, "_json_dict", {})
        if not isinstance(summary, dict):
            return None

        mae = (
            summary.get("test/mae")
            or summary.get("test.mae")
            or summary.get("test_mae")
            or summary.get("mae")
        )
        test_value = summary.get("test")
        if mae is None and isinstance(test_value, dict):
            mae = test_value.get("mae")

        return float(mae) if mae is not None else None
    except Exception:
        return None


def run_quality_gate(mae: float, project: str) -> bool:
    """Compare mae against all previous runs. Returns (promote_to_production, best_previous_mae)."""
    api      = wandb.Api()
    runs     = api.runs(project, order="-created_at")
    all_runs = [r for r in runs if _get_test_mae(r) is not None]
    print(all_runs)
    if len(all_runs) > 1:
        best_mae = min(_get_test_mae(r) for r in all_runs)
        print(f"Best previous MAE: {best_mae}")
        print(f"Current MAE: {mae}")
        print(f"Promote to production: {mae < best_mae}")
        return mae < best_mae

    return True

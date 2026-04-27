import json

import joblib
import wandb

from core.constants import WANDB_ALIAS_LATEST, WANDB_ALIAS_PRODUCTION, WANDB_ALIAS_STAGING


def register_model(
    predictor,
    model_name: str,
    metrics: dict,
    promote_to_production: bool,
    description: str = "",
) -> list[str]:
    """Serialize predictor to model.pkl, write metrics.json, and register as W&B artifact."""
    joblib.dump(predictor, "model.pkl")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    aliases = [WANDB_ALIAS_STAGING, WANDB_ALIAS_LATEST]
    if promote_to_production:
        aliases.append(WANDB_ALIAS_PRODUCTION)

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=description,
        metadata=metrics,
    )
    artifact.add_file("model.pkl")
    wandb.log_artifact(artifact, aliases=aliases)
    return aliases

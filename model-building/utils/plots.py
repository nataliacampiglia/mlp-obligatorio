import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


def plot_country_inflation(df: pd.DataFrame, country: str) -> None:
    pais = df[df["country"] == country].sort_values("date")
    plt.figure(figsize=(12, 4))
    plt.plot(pais["date"], pais["inflation"], linewidth=1.5)
    plt.title(f"Inflación alimentaria mensual — {pais['country_name'].iloc[0]}")
    plt.xlabel("Fecha")
    plt.ylabel("Inflación (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pred_vs_real(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicción vs Real en Test",
    wandb_key: str = "charts/pred_vs_real",
    n: int = 300,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.scatter(y_true[:n], y_pred[:n], alpha=0.4, s=10)
    lim = [y_true[:n].min(), y_true[:n].max()]
    plt.plot(lim, lim, "r--", linewidth=1)
    plt.xlabel("Inflación real (%)")
    plt.ylabel("Inflación predicha (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({wandb_key: wandb.Image(plt)})
    plt.show()


def plot_country_forecast(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    country: str,
    wandb_key: str,
    history_months: int = 24,
) -> None:
    historico = df[df["country"] == country].sort_values("date").tail(history_months)
    nombre    = df[df["country"] == country]["country_name"].iloc[0]

    plt.figure(figsize=(13, 4))
    plt.plot(historico["date"], historico["inflation"],
             label="Histórico", linewidth=1.5, color="steelblue")
    plt.plot(forecast["date"], forecast["inflation_pred"],
             label="Predicción (12 meses)", linewidth=1.5, linestyle="--",
             color="tomato", marker="o", markersize=4)
    plt.axvline(historico["date"].iloc[-1], color="gray", linestyle=":", linewidth=1)
    plt.title(f"Inflación alimentaria — {nombre} ({country})")
    plt.xlabel("Fecha")
    plt.ylabel("Inflación (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({wandb_key: wandb.Image(plt)})
    plt.show()


def plot_country_test_predictions(
    df: pd.DataFrame,
    model,
    df_test: pd.DataFrame,
    test_start: pd.Timestamp,
    feature_cols: list[str],
    country: str,
    wandb_key: str,
) -> None:
    nombre    = df[df["country"] == country]["country_name"].iloc[0]
    historico = df[(df["country"] == country) & (df["date"] < test_start)].sort_values("date")

    df_test_c = df_test[df_test["country"] == country].sort_values("date")
    preds     = np.expm1(model.predict(df_test_c[feature_cols].values))

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(historico["date"], historico["inflation"],
            label="Histórico", color="steelblue", linewidth=1.5)
    ax.plot(df_test_c["date"], df_test_c["inflation"],
            label="Real (test)", color="steelblue", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.plot(df_test_c["date"], preds,
            label="Predicción (test)", color="tomato", linewidth=1.5)
    ax.axvline(test_start, color="gray", linestyle=":", linewidth=1)
    ax.text(test_start + pd.DateOffset(days=5), ax.get_ylim()[0], "inicio test",
            color="gray", fontsize=8, va="bottom")
    ax.set_title(f"{nombre} ({country}) — Histórico completo y Predicción vs Real en Test")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Inflación alimentaria (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({wandb_key: wandb.Image(plt)})
    plt.show()

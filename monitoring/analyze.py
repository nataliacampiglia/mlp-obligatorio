"""
Análisis de logs de predicciones.

Pendiente de implementar. Esqueleto pensado para 5 paneles:

    1. Performance histórica (Modo A — necesita ground truth):
       MAE rolling sobre predicciones cuya fecha ya pasó, comparando contra
       el valor real del CSV de FAO.

    2. Sanity de outputs (Modo B — sin ground truth):
       Distribución de predicciones loggeadas vs rango p5-p95 del training.

    3. Drift de inputs (Modo B):
       PSI / KS test de las features loggeadas vs distribución del training.

    4. OOD detection (Modo B):
       Score por predicción: país conocido?, fecha dentro de extrapolación
       razonable?, lags dentro de p1-p99 del training?

    5. Modelo vs naive baseline (Modo B):
       Predicción del modelo vs último valor conocido del país.

Recomendado: levantar como Streamlit app (`streamlit run analyze.py`).
"""
from pathlib import Path

import pandas as pd

LOGS_PATH = Path(__file__).parent / "logs" / "predictions.jsonl"


def load_logs() -> pd.DataFrame:
    if not LOGS_PATH.exists():
        return pd.DataFrame()
    return pd.read_json(LOGS_PATH, lines=True)


if __name__ == "__main__":
    df = load_logs()
    print(f"Loaded {len(df)} prediction logs")
    print(df.head())

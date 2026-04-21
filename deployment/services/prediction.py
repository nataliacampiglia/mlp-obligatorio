from pathlib import Path

MODEL_PATH = Path(__file__).parents[2] / "dev" / "model.pkl"


def load_model():
    """Carga el artefacto model.pkl desde dev/. Retorna None si no existe."""
    pass


def predict(country: str, year: int, month: int) -> float | None:
    """
    Predice la inflación en alimentos para el par (país, año, mes).

    Retorna None si el modelo no está disponible todavía.
    """
    pass

# Deployment — Predictor de Inflación en Alimentos

Aplicación web construida con **FastAPI** que expone un formulario para predecir la inflación de precios de alimentos dado un país y un mes/año.

---

## Estructura del proyecto

```
deployment/
├── main.py                  # Punto de entrada de la app FastAPI
├── constants.py             # Lista de países y otras constantes
├── services/
│   └── prediction.py        # Servicio de predicción (lee el modelo desde ../dev/)
├── static/
│   └── index.html           # Formulario de una sola página (servido como contenido estático)
├── pyproject.toml           # Configuración de Poetry
├── poetry.lock
└── README.md
```

---

## Gestión de dependencias con Poetry

### Instalación de Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Inicialización del proyecto

```bash
cd deployment
poetry init          # genera pyproject.toml interactivamente
```

### Agregar dependencias

```bash
poetry add fastapi "uvicorn[standard]" pydantic
```

Agregar dependencias específicas del modelo una vez implementado el servicio de predicción:

```bash
poetry add scikit-learn joblib        # ejemplo según el modelo utilizado
```

### Instalar dependencias y activar el entorno

```bash
poetry install       # crea el virtualenv e instala todo lo declarado en pyproject.toml
poetry shell         # activa el virtualenv
```

### `pyproject.toml` de referencia

```toml
[tool.poetry]
name = "food-inflation-predictor"
version = "0.1.0"
description = "Predictor de inflación en alimentos"
authors = ["Tu Nombre <tu@email.com>"]

[tool.poetry.dependencies]
python = "^3.10"
fastapi = "*"
uvicorn = {extras = ["standard"], version = "*"}
pydantic = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

---

## Descripción de la aplicación

### Página estática (`static/index.html`)

La UI es un formulario HTML con estilo minimalista similar al siguiente diseño:

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│         Inflación en Alimentos                       │
│                                                      │
│   Fecha        [ YYYY-MM  ________________ ]         │
│                                                      │
│   País         [ Argentina            ▼   ]          │
│                                                      │
│                    [ Submit ]                        │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │ Predicción:                                    │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

Componentes:
- **Campo de fecha** — input `type="month"` (selector nativo de mes/año, sin día).
- **Selector de país** — `<select>` poblado desde el endpoint `/countries`.
- **Botón Submit** — azul, envía al endpoint `/predict` vía `fetch`.
- **Caja de resultado** — muestra la predicción debajo del botón, con borde visible.

### Constantes (`constants.py`)

```python
COUNTRIES = [
    "Argentina",
    "Bolivia",
    "Brasil",
    "Chile",
    "Colombia",
    "Ecuador",
    "Paraguay",
    "Perú",
    "Uruguay",
    "Venezuela",
    # agregar más según sea necesario
]
```

### Endpoints

| Método | Ruta         | Descripción                                                              |
|--------|--------------|--------------------------------------------------------------------------|
| GET    | `/`          | Sirve `static/index.html`                                                |
| GET    | `/countries` | Retorna la lista de países disponibles como JSON                         |
| POST   | `/predict`   | Recibe `{country, year, month}` y retorna el valor de inflación predicho |

### Servicio de predicción (`services/prediction.py`)

Lee el modelo entrenado desde `../dev/` y lo utiliza para realizar inferencia.

```python
# services/prediction.py

def load_model():
    """Carga el artefacto de modelo más reciente desde la carpeta dev."""
    pass


def predict(country: str, year: int, month: int) -> float:
    """
    Predice la inflación de precios de alimentos para el país y mes/año dados.

    Parámetros
    ----------
    country : str  — nombre del país (debe estar en COUNTRIES)
    year    : int  — año de 4 dígitos
    month   : int  — mes entre 1 y 12

    Retorna
    -------
    float — valor de inflación en alimentos predicho
    """
    pass
```

### App FastAPI (`main.py`)

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from constants import COUNTRIES
from services.prediction import predict

app = FastAPI(title="Predictor de Inflación en Alimentos")
app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictRequest(BaseModel):
    country: str
    year: int
    month: int  # 1-12


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/countries")
def get_countries():
    return {"countries": COUNTRIES}


@app.post("/predict")
def run_prediction(req: PredictRequest):
    value = predict(req.country, req.year, req.month)
    return {"country": req.country, "year": req.year, "month": req.month, "inflation": value}
```

---

## Ejecución local

```bash
poetry install
poetry run uvicorn main:app --reload --port 8000
```

Abrir `http://localhost:8000` en el navegador.

---

## Carga del modelo

El servicio de predicción espera encontrar el modelo en `../dev/`. Todo modelo entrenado y validado en la etapa `model-building` es promovido a `dev/` antes de iniciar este servicio. El servicio lee el artefacto más reciente al iniciarse mediante `load_model()`.

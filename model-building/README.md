# model-building

Etapa de experimentación y entrenamiento de modelos candidatos.

## Contenido

| Archivo | Descripción |
|---|---|
| `food_price_inflation.csv` | Dataset de inflación alimentaria mensual (FAO) |
| `food_inflation_model.ipynb` | Notebook de entrenamiento y evaluación |
| `model.pkl` | Artefacto del modelo entrenado (generado al correr el notebook) |
| `metrics.json` | Métricas del modelo (MAE, RMSE, R², Accuracy por tolerancia) |

## Flujo

1. Entrenar el modelo en el notebook
2. Evaluar métricas en validación y test (split temporal)
3. Si las métricas superan el umbral → promover modelo a `dev/`

## Criterio de promoción

El modelo se promueve si en el set de test (último año de datos) cumple:

- MAE aceptable para el rango de inflación del país
- Accuracy (±5pp) > umbral definido por el equipo


Datos a gurdar de cada modelo:
- descricion del modelo: ej random forest, profundidad
- path del modelo
- score
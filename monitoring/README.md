# Monitoring

Cierra el feedback loop de CD4ML: cada predicción del servicio se loggea, y desde estos logs se analiza el comportamiento del modelo en producción.

## Estructura

```
monitoring/
├── logger.py       # API de logging (llamada desde deployment/)
├── analyze.py      # Stub de análisis — pendiente de implementar
├── logs/
│   └── predictions.jsonl   # un evento por línea (gitignored)
└── README.md
```

## Qué se loguea

Cada llamada a `predict()` escribe un evento en `logs/predictions.jsonl`:

```json
{"timestamp": "...", "model_version": "...", "country": "URY", "year": 2030, "month": 6, "prediction": 8.4}
```

Formato JSONL para poder appendear sin parsear y leer con `pd.read_json(lines=True)`.

## Qué se analiza

Dos modos de monitoreo, complementarios:

- **Modo A — con ground truth**: para predicciones de fechas que ya pasaron, se compara contra el valor real del CSV de FAO. Métrica: MAE rolling.
- **Modo B — sin ground truth**: para predicciones futuras (ej. Uruguay 2030). Se monitorea drift de inputs, sanity de outputs, OOD, comparación con baseline naive.

Detalle en `analyze.py`.

## Stack en producción real

Fowler (CD4ML) sugiere EFK (Elasticsearch + FluentD + Kibana) para log aggregation y dashboards. Acá simplificamos a JSONL local + análisis offline para la demo.

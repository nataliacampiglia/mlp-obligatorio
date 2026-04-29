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

Todo se deriva exclusivamente de los logs (sin ground truth, sin CSV externo). Por eso no medimos accuracy: para fechas futuras (ej. Uruguay 2030) **el ground truth no existe todavía**. En producción real, eso se resolvería con un segundo tipo de evento (`{type: "actual", ...}`) loggeado cuando llega el dato real, pero queda fuera del alcance de la demo.

Lo que sí podemos monitorear desde logs puros:

- **Usage analytics**: volumen, top países, fechas predichas, versiones servidas.
- **Distribución de outputs**: ¿las predicciones caen en un rango sano? Auto-calibrado con p5/p95 de los propios logs.
- **Anomaly detection**: predicciones con z-score alto contra el promedio del país en los logs.
- **OOD scoring**: país conocido + cuánta extrapolación temporal hay (derivado de `months_from_last_known`).
- **Modelo vs naive baseline**: cuánto se aleja la predicción del último valor conocido del país.
- **Drill-down**: inspección puntual de una predicción con todos sus flags.

Run:

```bash
streamlit run monitoring/analyze.py
```

## Stack en producción real

Fowler (CD4ML) sugiere EFK (Elasticsearch + FluentD + Kibana) para log aggregation y dashboards. Acá simplificamos a JSONL local + análisis offline para la demo.

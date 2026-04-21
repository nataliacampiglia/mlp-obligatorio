# Continuous Delivery for Machine Learning — Inflación Alimentaria

Implementación práctica de los principios del artículo [Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html) de Martin Fowler, aplicados a la predicción de inflación alimentaria mensual por país.

---

## Contexto

El artículo plantea que aplicar Continuous Delivery a sistemas de Machine Learning requiere coordinar tres ejes que cambian simultáneamente: el **código**, los **datos** y el **modelo**. A diferencia del software tradicional, una aplicación de ML puede degradarse sin que nadie toque una sola línea de código, simplemente porque los datos del mundo cambiaron.

Para representar este concepto de forma concreta, estructuramos el proyecto en **carpetas que mapean directamente las etapas del pipeline de CD4ML**. Cada carpeta es un componente autónomo con una responsabilidad clara, de la misma manera en que en un pipeline de entrega continua cada etapa tiene un propósito definido antes de pasar a la siguiente.

---

## Estructura del proyecto

```
mlp-obligatorio/
├── model-building/     # Experimentación y entrenamiento
├── dev/                # Integración y validación
├── deployment/         # Empaquetado y entrega
└── prod/               # Ambiente productivo
```

### `model-building/`
Etapa de experimentación. Aquí vive el notebook donde se entrena el modelo, se evalúan métricas y se generan los artefactos (`model.pkl`, `metrics.json`). Es el equivalente al "experimento en rama" que describe el artículo: trabajo exploratorio que todavía no está en producción.

Un modelo se promueve a `dev/` si supera el umbral de métricas definido.

### `dev/`
Ambiente de integración. Una vez que el modelo supera los umbrales, se validan los datos de entrada, se corren tests del modelo y se verifica que el pipeline completo funcione de punta a punta. Refleja el principio del artículo de que *"los modelos deben ser testeados, no solo entrenados"*.

Un modelo se promueve a `deployment/` si es mejor que el mejor modelo registrado en dev.

### `deployment/`
Contiene la lógica para empaquetar y exponer el modelo como servicio. El artículo describe tres patrones posibles (modelo embebido, servicio independiente, modelo como dato); esta carpeta abstrae ese paso de entrega. Aquí el modelo deja de ser un artefacto de experimentación y se convierte en algo consumible por otras aplicaciones.

### `prod/`
Ambiente productivo. Es el destino final del pipeline y el punto desde donde se monitorea el comportamiento real del modelo. El artículo enfatiza que el ciclo no termina en el deploy: los datos que genera producción deben realimentar el entrenamiento, cerrando el loop de mejora continua.

---

## Por qué carpetas

La decisión de usar carpetas como mecanismo de representación es intencional: es la forma más directa de mostrar que **CD4ML no es una herramienta, es una disciplina organizativa**. Cada carpeta impone una frontera explícita que obliga a pensar qué entra, qué sale y qué condiciones deben cumplirse para avanzar de una etapa a la siguiente, exactamente como lo haría un pipeline de CI/CD tradicional aplicado al mundo del Machine Learning.

---

## Dataset

`food_price_inflation.csv` — inflación alimentaria mensual por país (2001–2025), fuente FAO.

| Columna | Descripción |
|---|---|
| `REF_AREA` | Código ISO del país |
| `REF_AREA_LABEL` | Nombre del país |
| `TIME_PERIOD` | Fecha (primer día del mes) |
| `OBS_VALUE` | Inflación alimentaria (%) |

## Modelo

Dado el par **(país, fecha)** predice la inflación alimentaria para los próximos 12 meses usando un `RandomForestRegressor` entrenado con features de país codificado, año y mes. El split es temporal: train → validación → test (último año).

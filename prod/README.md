# prod

Ambiente productivo. Contiene el modelo activo que sirve predicciones reales.

## Responsabilidades

- Servir el modelo con mejor performance registrado
- Monitorear el comportamiento del modelo en producción
- Realimentar el ciclo: los datos nuevos generados en prod vuelven a `model-building/` para el próximo ciclo de entrenamiento

## Loop de mejora continua

```
prod/ → nuevos datos → model-building/ → dev/ → deployment/ → prod/
```

Este ciclo es el núcleo del Continuous Delivery for ML: el modelo mejora de forma incremental y controlada, sin interrupciones del servicio.

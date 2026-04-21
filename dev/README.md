# dev

Ambiente de integración y validación. Todo modelo que supera los umbrales de `model-building/` es promovido a esta etapa.

## Responsabilidades

- Validar el schema y calidad de los datos de entrada
- Correr tests de integración del pipeline completo
- Comparar el modelo candidato contra el mejor modelo registrado

## Criterio de promoción a deployment

El modelo se promueve a `deployment/` si sus métricas en dev son mejores que las del modelo actualmente en producción.

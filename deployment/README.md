# deployment

Empaquetado y exposición del modelo como servicio consumible.

## Responsabilidades

- Leer el modelo promovido desde `dev/`
- Exponer una interfaz para predecir inflación dado `(país, fecha)`
- Versionar cada deploy para permitir rollback

## Criterio de activación en prod

El modelo deployado se activa en `prod/` si pasa las validaciones de integración y su accuracy supera el 80% con tolerancia ±5pp.

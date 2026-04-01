# Actividad14-clases

Estructura simple para trabajar un flujo de ciencia de datos con Python.

## Estructura

- `data/raw/`: dataset original sin modificar.
- `data/processed/`: salidas limpias y transformadas.
- `notebooks/`: notebook principal del flujo de trabajo.

## Archivo principal

- Dataset sugerido: `data/raw/datos.csv`
- Notebook principal: `notebooks/flujo_principal.ipynb`

## Flujo recomendado

1. Cargar el dataset original desde `data/raw/`.
2. Explorar columnas, tipos y calidad de datos.
3. Limpiar nulos, duplicados y outliers.
4. Transformar variables con escalado, codificacion e ingenieria de caracteristicas.
5. Guardar resultados en `data/processed/`.

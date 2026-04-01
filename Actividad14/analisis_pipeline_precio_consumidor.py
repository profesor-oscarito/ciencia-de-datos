import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("Actividad14/precio_consumidor_2025.csv")


def normalize_col_name(col: str) -> str:
    """Normalize raw CSV column names to snake_case ascii."""
    col = col.replace('"', "").replace("\ufeff", "")
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("ascii")
    col = col.lower().strip()
    col = re.sub(r"[^a-z0-9]+", "_", col).strip("_")
    return col


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df.columns = [normalize_col_name(c) for c in df.columns]

    # Some files include a BOM that can distort 'anio' as 'ianio'.
    for c in list(df.columns):
        if c.endswith("anio") and c != "anio":
            df = df.rename(columns={c: "anio"})

    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce")
    df["fecha_termino"] = pd.to_datetime(df["fecha_termino"], errors="coerce")
    df["precio_promedio"] = pd.to_numeric(
        df["precio_promedio"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df = df.dropna(subset=["precio_promedio", "fecha_inicio"]).copy()
    df["dia_anio"] = df["fecha_inicio"].dt.dayofyear

    return df


def split_temporal(df: pd.DataFrame, quantile: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    df = df.sort_values("fecha_inicio").copy()
    cut_date = df["fecha_inicio"].quantile(quantile)
    train_df = df[df["fecha_inicio"] <= cut_date].copy()
    test_df = df[df["fecha_inicio"] > cut_date].copy()
    return train_df, test_df, cut_date


def build_preprocessor(features: list[str], X: pd.DataFrame) -> ColumnTransformer:
    cat_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_features = [c for c in features if c not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    return preprocessor


def evaluate_pipeline(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(root_mean_squared_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred)),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH}")

    df = load_and_clean(DATA_PATH)
    train_df, test_df, cut_date = split_temporal(df, quantile=0.8)

    y_train = train_df["precio_promedio"]
    y_test = test_df["precio_promedio"]

    features_without_bounds = [
        "anio",
        "mes",
        "semana",
        "id_region",
        "region",
        "sector",
        "tipo_de_punto_monitoreo",
        "grupo",
        "producto",
        "unidad",
        "dia_anio",
    ]

    features_with_bounds = features_without_bounds + ["precio_minimo", "precio_maximo"]

    X_train_a = train_df[features_without_bounds]
    X_test_a = test_df[features_without_bounds]
    prep_a = build_preprocessor(features_without_bounds, X_train_a)

    model_candidates = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0, random_state=0),
    }

    print("=== RESUMEN DEL DATASET ===")
    print(f"Filas/columnas: {df.shape}")
    print(f"Rango de fechas: {df['fecha_inicio'].min().date()} a {df['fecha_inicio'].max().date()}")
    print(f"Train/Test temporal: {train_df.shape} / {test_df.shape} (corte: {cut_date.date()})")
    print("Cardinalidad (producto, sector, region, grupo, unidad):")
    print(
        df["producto"].nunique(),
        df["sector"].nunique(),
        df["region"].nunique(),
        df["grupo"].nunique(),
        df["unidad"].nunique(),
    )

    print("\n=== ESCENARIO A: SIN precio_minimo/precio_maximo ===")
    for name, model in model_candidates.items():
        pipe = Pipeline(steps=[("prep", prep_a), ("model", model)])
        metrics = evaluate_pipeline(pipe, X_train_a, y_train, X_test_a, y_test)
        print(f"{name:12s} -> MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} R2={metrics['R2']:.4f}")

    print("\n=== ESCENARIO B: INCLUYENDO precio_minimo/precio_maximo ===")
    X_train_b = train_df[features_with_bounds]
    X_test_b = test_df[features_with_bounds]
    prep_b = build_preprocessor(features_with_bounds, X_train_b)
    pipe_b = Pipeline(steps=[("prep", prep_b), ("model", Ridge(alpha=1.0, random_state=0))])
    metrics_b = evaluate_pipeline(pipe_b, X_train_b, y_train, X_test_b, y_test)
    print(f"ridge+bounds -> MAE={metrics_b['MAE']:.2f} RMSE={metrics_b['RMSE']:.2f} R2={metrics_b['R2']:.4f}")

    midpoint = (df["precio_minimo"] + df["precio_maximo"]) / 2
    midpoint_mae = np.mean(np.abs(midpoint - df["precio_promedio"]))
    print(f"Referencia simple ((min+max)/2) MAE={midpoint_mae:.2f}")

    print("\n=== INTERPRETACION ===")
    print("- Para un ejercicio realista de prediccion, usar objetivo: precio_promedio")
    print("- Variables recomendadas: producto, region, sector, tipo_de_punto_monitoreo, grupo, unidad, semana/mes/dia_anio")
    print("- Evitar precio_minimo y precio_maximo como predictores si el objetivo es estimar precio_promedio antes del levantamiento")


if __name__ == "__main__":
    main()

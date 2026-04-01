from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path("Actividad14/hogares_censo2024.csv")
OUTPUT_DIR = Path("Actividad14/output_hogares")
SAMPLE_SIZE = 120_000
RANDOM_STATE = 42


ID_COLS = ["id_vivienda", "id_hogar"]
TARGET_COL = "tipologia_hogar"

CATEGORICAL_COLS = [
    "region",
    "provincia",
    "comuna",
    "comuna_bajo_umbral",
    "area",
    "tipo_operativo",
    "p12_tenencia_viv",
    "p13_comb_cocina",
    "p14_comb_calefaccion",
    "p15a_serv_tel_movil",
    "p15b_serv_compu",
    "p15c_serv_tablet",
    "p15d_serv_internet_fija",
    "p15e_serv_internet_movil",
    "p15f_serv_internet_satelital",
]

SERVICE_COLS = [
    "p15a_serv_tel_movil",
    "p15b_serv_compu",
    "p15c_serv_tablet",
    "p15d_serv_internet_fija",
    "p15e_serv_internet_movil",
    "p15f_serv_internet_satelital",
]


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8")


def build_representative_sample(df: pd.DataFrame, n_sample: int, random_state: int) -> pd.DataFrame:
    # Keep rows with target for supervised learning example.
    work = df.dropna(subset=[TARGET_COL]).copy()
    work[TARGET_COL] = work[TARGET_COL].astype(int)

    # Strata by geography and class label to preserve key structure for classes.
    work["_strata"] = work["region"].astype(str) + "_" + work[TARGET_COL].astype(str)

    weights = work["_strata"].value_counts(normalize=True)
    target_counts = (weights * n_sample).round().astype(int)

    samples = []
    for stratum, cnt in target_counts.items():
        if cnt <= 0:
            continue
        chunk = work[work["_strata"] == stratum]
        take = min(cnt, len(chunk))
        samples.append(chunk.sample(n=take, random_state=random_state))

    sampled = pd.concat(samples, ignore_index=True)

    # Adjust size if rounding drift occurs.
    if len(sampled) > n_sample:
        sampled = sampled.sample(n=n_sample, random_state=random_state)
    elif len(sampled) < n_sample:
        residual = n_sample - len(sampled)
        missing_pool = work.drop(sampled.index, errors="ignore")
        if len(missing_pool) >= residual:
            sampled = pd.concat(
                [sampled, missing_pool.sample(n=residual, random_state=random_state)],
                ignore_index=True,
            )

    return sampled.drop(columns=["_strata"], errors="ignore")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info: dict[str, int | float] = {}
    clean = df.copy()

    info["rows_before"] = len(clean)
    dup = int(clean.duplicated().sum())
    info["duplicates_removed"] = dup
    clean = clean.drop_duplicates()

    # Convert coded columns to numeric and handle coded outliers such as -99 / 0 as missing.
    anomaly_count = 0
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        mask_anomaly = clean[col].isin([-99, -88, -77, 0])
        anomaly_count += int(mask_anomaly.sum())
        clean.loc[mask_anomaly, col] = np.nan

    # Impute missing in categorical coded variables with mode.
    for col in CATEGORICAL_COLS:
        if clean[col].isna().any():
            mode = clean[col].mode(dropna=True)
            if not mode.empty:
                clean[col] = clean[col].fillna(mode.iloc[0])

    # Target rows with missing are dropped.
    missing_target = int(clean[TARGET_COL].isna().sum())
    info["target_missing_removed"] = missing_target
    clean = clean.dropna(subset=[TARGET_COL])

    clean[TARGET_COL] = clean[TARGET_COL].astype(int)
    info["coded_anomalies_replaced"] = anomaly_count
    info["rows_after"] = len(clean)
    return clean, info


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Feature engineering from service availability.
    for c in SERVICE_COLS:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    service_yes = [(data[c] == 1).astype(int) for c in SERVICE_COLS]
    data["n_servicios_digitales"] = np.sum(service_yes, axis=0)
    data["internet_any"] = (
        ((data["p15d_serv_internet_fija"] == 1) | (data["p15e_serv_internet_movil"] == 1)).astype(int)
    )

    # Number of hogares by vivienda in the sample.
    hogares_por_vivienda = data.groupby("id_vivienda")["id_hogar"].transform("count")
    data["hogares_por_vivienda"] = hogares_por_vivienda

    # Binary area helper feature.
    data["es_urbano"] = (data["area"] == 1).astype(int)

    return data


def build_transformer(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ]
    )


def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=600, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    metrics = []
    reports = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1w = f1_score(y_test, pred, average="weighted")
        metrics.append({"model": name, "accuracy": round(float(acc), 4), "f1_weighted": round(float(f1w), 4)})
        reports[name] = classification_report(y_test, pred, output_dict=True, zero_division=0)

    return pd.DataFrame(metrics).sort_values("f1_weighted", ascending=False), reports


def evaluate_binary_internet(cleaned: pd.DataFrame) -> pd.DataFrame:
    df = cleaned[cleaned["p15d_serv_internet_fija"].isin([1, 2])].copy()
    df["target_internet_fija"] = (df["p15d_serv_internet_fija"] == 1).astype(int)

    cat_cols = [
        "region",
        "provincia",
        "comuna",
        "comuna_bajo_umbral",
        "area",
        "tipo_operativo",
        "p12_tenencia_viv",
        "p13_comb_cocina",
        "p14_comb_calefaccion",
        "tipologia_hogar",
    ]
    num_cols = ["n_servicios_digitales", "internet_any", "hogares_por_vivienda", "es_urbano"]

    X = df[cat_cols + num_cols]
    y = df["target_internet_fija"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    prep = build_transformer(cat_cols, num_cols)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", prep), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append(
            {
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "f1_weighted": round(float(f1_score(y_test, pred, average="weighted")), 4),
                "f1_positive": round(float(f1_score(y_test, pred)), 4),
                "positive_ratio": round(float(y.mean()), 4),
            }
        )

    return pd.DataFrame(rows).sort_values("f1_positive", ascending=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1) Carga y exploracion inicial")
    df = load_data(DATA_PATH)
    print(f"Shape original: {df.shape}")

    print("\n2) Muestra representativa educacional")
    sample = build_representative_sample(df, n_sample=SAMPLE_SIZE, random_state=RANDOM_STATE)
    sample_path = OUTPUT_DIR / "hogares_censo2024_muestra_educacional.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8")
    print(f"Muestra guardada: {sample_path} | shape={sample.shape}")

    print("\n3) Limpieza: faltantes, duplicados y outliers codificados")
    cleaned, clean_info = clean_data(sample)
    cleaned = add_features(cleaned)
    clean_path = OUTPUT_DIR / "hogares_censo2024_muestra_limpia.csv"
    cleaned.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"Limpia guardada: {clean_path} | shape={cleaned.shape}")
    print("Resumen limpieza:")
    print(json.dumps(clean_info, indent=2))

    print("\n4) Transformacion avanzada: normalizacion + encoding + feature engineering")
    model_features_cat = CATEGORICAL_COLS
    model_features_num = ["n_servicios_digitales", "internet_any", "hogares_por_vivienda", "es_urbano"]
    X = cleaned[model_features_cat + model_features_num]
    y = cleaned[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_transformer(model_features_cat, model_features_num)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print(f"Matriz transformada train: {X_train_t.shape}")
    print(f"Matriz transformada test: {X_test_t.shape}")

    print("\n5) Benchmark de modelos para recomendacion didactica")
    metrics_df, reports = evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    print(metrics_df)

    metrics_path = OUTPUT_DIR / "metricas_modelos_tipologia_hogar.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    # Save short class-wise F1 report for the best model.
    best_model = metrics_df.iloc[0]["model"]
    report_df = pd.DataFrame(reports[best_model]).T
    report_path = OUTPUT_DIR / f"reporte_clasificacion_{best_model.lower()}.csv"
    report_df.to_csv(report_path, index=True, encoding="utf-8")

    print("\n6) Objetivo alternativo: predecir internet fija (binario)")
    internet_metrics = evaluate_binary_internet(cleaned)
    print(internet_metrics)
    internet_metrics_path = OUTPUT_DIR / "metricas_modelos_internet_fija.csv"
    internet_metrics.to_csv(internet_metrics_path, index=False, encoding="utf-8")

    print("\nArchivos generados:")
    print(f"- {sample_path}")
    print(f"- {clean_path}")
    print(f"- {metrics_path}")
    print(f"- {report_path}")
    print(f"- {internet_metrics_path}")


if __name__ == "__main__":
    main()

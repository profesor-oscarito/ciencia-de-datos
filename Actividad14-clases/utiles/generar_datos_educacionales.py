from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
ROWS = 30000
OUTPUT_PATH = Path(__file__).resolve().parent / "datos.csv"


rng = np.random.default_rng(SEED)

regiones = {
    "Metropolitana": ["Santiago Centro", "Providencia", "Maipu"],
    "Valparaiso": ["Valparaiso", "Vina del Mar", "Quilpue"],
    "Biobio": ["Concepcion", "Talcahuano", "Los Angeles"],
    "La Araucania": ["Temuco", "Villarrica", "Padre Las Casas"],
}

sucursales = {
    "SCL-01": ("Sucursal Alameda", "Metropolitana", "Santiago Centro"),
    "SCL-02": ("Sucursal Oriente", "Metropolitana", "Providencia"),
    "SCL-03": ("Sucursal Poniente", "Metropolitana", "Maipu"),
    "VAP-01": ("Sucursal Puerto", "Valparaiso", "Valparaiso"),
    "VAP-02": ("Sucursal Costa", "Valparaiso", "Vina del Mar"),
    "BIO-01": ("Sucursal Centro Sur", "Biobio", "Concepcion"),
    "BIO-02": ("Sucursal Industrial", "Biobio", "Talcahuano"),
    "ARA-01": ("Sucursal Frontera", "La Araucania", "Temuco"),
}

productos = {
    "P001": ("Notebook 14\"", "Tecnologia"),
    "P002": ("Mouse Inalambrico", "Tecnologia"),
    "P003": ("Silla Ergonomica", "Hogar"),
    "P004": ("Escritorio Ajustable", "Hogar"),
    "P005": ("Botella Termica", "Deportes"),
    "P006": ("Mancuernas 5kg", "Deportes"),
    "P007": ("Mochila Urbana", "Accesorios"),
    "P008": ("Lampara LED", "Hogar"),
    "P009": ("Audifonos Bluetooth", "Tecnologia"),
    "P010": ("Cuaderno Ejecutivo", "Libreria"),
}

segmentos = ["Estudiante", "Profesional", "Pyme", "Familiar"]
generos = ["F", "M", "No informa"]
canales = ["Tienda", "Online", "App"]
metodos_pago = ["Debito", "Credito", "Transferencia", "Efectivo"]

fechas = pd.date_range("2024-03-01", "2024-09-30", freq="D")

registros = []
for index in range(ROWS):
    sucursal_id = rng.choice(list(sucursales.keys()))
    sucursal, region, comuna = sucursales[sucursal_id]
    producto_id = rng.choice(list(productos.keys()))
    producto, categoria = productos[producto_id]
    cliente_id = f"C{rng.integers(1000, 1125):04d}"
    fecha = pd.Timestamp(rng.choice(fechas)).strftime("%Y-%m-%d")
    edad = int(np.clip(rng.normal(38, 12), 18, 75))
    ingreso = int(np.clip(rng.normal(950000, 350000), 250000, 2600000))
    unidades = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6]))
    precio_unitario = round(float(rng.uniform(2500, 180000)), 2)
    descuento_pct = round(float(rng.choice([0, 0.05, 0.1, 0.15, 0.2])), 2)
    costo_envio = round(float(rng.choice([0, 1990, 2990, 3990, 4990])), 2)
    satisfaccion = int(rng.integers(1, 6))
    devolucion = int(rng.choice([0, 0, 0, 0, 1]))

    registros.append(
        {
            "venta_id": f"V{index + 1:04d}",
            "fecha": fecha,
            "cliente_id": cliente_id,
            "edad": edad,
            "genero": rng.choice(generos, p=[0.47, 0.47, 0.06]),
            "segmento_cliente": rng.choice(segmentos, p=[0.3, 0.35, 0.15, 0.2]),
            "ingreso_mensual": ingreso,
            "sucursal_id": sucursal_id,
            "sucursal": sucursal,
            "region": region,
            "comuna": comuna,
            "canal": rng.choice(canales, p=[0.45, 0.4, 0.15]),
            "metodo_pago": rng.choice(metodos_pago, p=[0.35, 0.4, 0.15, 0.1]),
            "producto_id": producto_id,
            "producto": producto,
            "categoria": categoria,
            "unidades": unidades,
            "precio_unitario": precio_unitario,
            "descuento_pct": descuento_pct,
            "costo_envio": costo_envio,
            "satisfaccion": satisfaccion,
            "devolucion": devolucion,
        }
    )


df = pd.DataFrame(registros)
df["monto_bruto"] = (df["unidades"] * df["precio_unitario"]).round(2)
df["monto_final"] = (df["monto_bruto"] * (1 - df["descuento_pct"]) + df["costo_envio"]).round(2)
df["antiguedad_cliente_meses"] = rng.integers(1, 84, size=len(df))

def set_missing(frame: pd.DataFrame, column: str, count: int) -> None:
    indexes = rng.choice(frame.index.to_numpy(), size=count, replace=False)
    frame.loc[indexes, column] = np.nan


set_missing(df, "edad", 8)
set_missing(df, "genero", 7)
set_missing(df, "segmento_cliente", 6)
set_missing(df, "ingreso_mensual", 10)
set_missing(df, "descuento_pct", 6)
set_missing(df, "satisfaccion", 5)
set_missing(df, "costo_envio", 4)

# Valores anómalos para practicar outliers y reglas de negocio.
outlier_rows = rng.choice(df.index.to_numpy(), size=6, replace=False)
df.loc[outlier_rows[0], ["edad", "ingreso_mensual"]] = [97, 9800000]
df.loc[outlier_rows[1], ["unidades", "precio_unitario"]] = [120, 899990]
df.loc[outlier_rows[2], ["descuento_pct", "monto_final"]] = [1.35, -500]
df.loc[outlier_rows[3], ["costo_envio", "monto_final"]] = [-3500, 15000]
df.loc[outlier_rows[4], ["ingreso_mensual", "satisfaccion"]] = [15000000, 7]
df.loc[outlier_rows[5], ["edad", "unidades"]] = [12, 80]

# Duplicados intencionales.
duplicados = df.sample(8, random_state=SEED).copy()
duplicados["venta_id"] = duplicados["venta_id"].values

df_final = pd.concat([df, duplicados], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=SEED).reset_index(drop=True)

df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"Archivo generado: {OUTPUT_PATH}")
print(f"Filas: {len(df_final)}")
print(f"Duplicados exactos: {df_final.duplicated().sum()}")
print("Nulos por columna:")
print(df_final.isna().sum().sort_values(ascending=False).head(8).to_string())

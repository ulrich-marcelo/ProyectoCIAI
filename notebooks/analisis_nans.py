from pathlib import Path
import pandas as pd

PARQUET_FILE = Path("data/eph_hogares_consolidada_2016_2026.parquet")
df = pd.read_parquet(PARQUET_FILE)

# 1. Eliminar columnas espurias (100% NaN)
cols_a_borrar = [c for c in df.columns if c.endswith("_esp") or c == "mas_500"]
df = df.drop(columns=cols_a_borrar)

# 2. Descartar los 16 registros residuales con respuestas incompletas de vivienda
df = df.dropna().reset_index(drop=True)

# 3. Sobrescribir el parquet limpio
df.to_parquet(PARQUET_FILE, index=False, compression="snappy")

print(f"Dataset corregido con éxito.")
print(f"Filas resultantes: {len(df):,}")
print(f"Columnas resultantes: {df.shape[1]}")
print(f"Total de NaNs remanentes: {df.isna().sum().sum()}")
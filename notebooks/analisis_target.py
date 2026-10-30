"""
===============================================================================
PROYECTO: Predicción de Pobreza mediante Machine Learning (EPH 2016-2026)
AUTOR: Marcelo Gabriel Ulrich | Licenciatura en Ciencia de Datos - UNSAM
AUDITORÍA METODOLÓGICA: Validación de Pobreza en Hogares (Semestral)
===============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# 1. SERIE HISTÓRICA OFICIAL DE POBREZA EN HOGARES (INDEC - 31 AGLOMERADOS)
# =============================================================================
# Valores oficiales extraídos de los Informes Técnicos Semestrales del INDEC
INDEC_HOGARES_OFICIAL = {
    "2016-S2": 21.5,
    "2017-S1": 20.4,
    "2017-S2": 17.9,
    "2018-S1": 19.6,
    "2018-S2": 23.4,
    "2019-S1": 25.4,
    "2019-S2": 25.9,
    "2020-S1": 30.4,
    "2020-S2": 31.6,
    "2021-S1": 31.2,
    "2021-S2": 27.9,
    "2022-S1": 27.7,
    "2022-S2": 29.6,
    "2023-S1": 29.6,
    "2023-S2": 31.8,
    "2024-S1": 42.5,
    "2024-S2": 27.8,
    "2025-S1": 24.1,
    "2025-S2": 21.0,
    "2026-S1": 24.4
}

# =============================================================================
# 2. CÁLCULO PONDERADO DE POBREZA EN HOGARES
# =============================================================================
def auditar_pobreza_hogares(parquet_path: str = "data/eph_hogares_consolidada_2016_2026.parquet") -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")

    # Cargar únicamente las columnas necesarias para el cálculo de tasas
    cols = ['ano4', 'trimestre', 'pondih', 'pobre_oficial', 'pobre_tipo2']
    df = pd.read_parquet(path, columns=cols)

    # 1. Filtrar hogares respondentes (etiqueta no nula) y con factor de expansión válido
    validos = df.dropna(subset=['pobre_oficial', 'pobre_tipo2']).copy()
    validos = validos[validos['pondih'] > 0]

    # 2. Construir la agrupación semestral (T1-T2 = S1; T3-T4 = S2)
    validos['semestre'] = np.where(validos['trimestre'].isin([1, 2]), 1, 2)
    validos['periodo'] = validos['ano4'].astype(str) + "-S" + validos['semestre'].astype(str)

    registros = []

    # 3. Iterar cronológicamente por semestre
    for periodo, subdf in validos.groupby('periodo'):
        w = subdf['pondih'].values

        # Promedio ponderado por PONDIH sobre hogares
        tasa_aduequi = np.average(subdf['pobre_oficial'], weights=w) * 100
        tasa_tipo2   = np.average(subdf['pobre_tipo2'], weights=w) * 100
        indec_val    = INDEC_HOGARES_OFICIAL.get(periodo, np.nan)

        registros.append({
            "periodo": periodo,
            "calculo_hogar_tipo2": round(tasa_tipo2, 1),
            "calculo_aduequi": round(tasa_aduequi, 1),
            "medicion_indec": indec_val
        })

    # DataFrame con las 4 columnas exactas solicitadas
    df_resultado = pd.DataFrame(registros)[
        ["periodo", "calculo_hogar_tipo2", "calculo_aduequi", "medicion_indec"]
    ]
    return df_resultado

# =============================================================================
# 3. EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    df_pobreza_hogares = auditar_pobreza_hogares()
    print("\n--- COMPARACIÓN DE POBREZA EN HOGARES (%) ---")
    print(df_pobreza_hogares.to_string(index=False))
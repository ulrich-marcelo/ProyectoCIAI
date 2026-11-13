from pathlib import Path
import numpy as np
import pandas as pd

INDEC_HOGARES_OFICIAL = {
    "2016-S1": 23.1,
    "2016-S2": 21.5,
    "2017-S1": 20.4, "2017-S2": 17.9,
    "2018-S1": 19.6, "2018-S2": 23.4,
    "2019-S1": 25.4, "2019-S2": 25.9,
    "2020-S1": 30.4, "2020-S2": 31.6,
    "2021-S1": 31.2, "2021-S2": 27.9,
    "2022-S1": 27.7, "2022-S2": 29.6,
    "2023-S1": 29.6, "2023-S2": 31.8,
    "2024-S1": 42.5, "2024-S2": 27.8,
    "2025-S1": 24.1, "2025-S2": 21.0,
    "2026-S1": 24.4
}

def auditar_pobreza_hogares(parquet_path: str = "data/eph_hogares_consolidada_2016_2026.parquet"):
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")

    df = pd.read_parquet(path, columns=['ano4', 'trimestre', 'pobre', 'pondih'])
    validos = df.dropna(subset=['pobre', 'pondih']).copy()
    validos['semestre'] = np.where(validos['trimestre'].isin([1, 2]), 1, 2)

    registros = []
    for (ano, sem), subdf in validos.groupby(['ano4', 'semestre']):
        periodo = f"{ano}-S{sem}"
        tasa_calculada = np.average(subdf['pobre'], weights=subdf['pondih']) * 100
        indec_val = INDEC_HOGARES_OFICIAL.get(periodo, np.nan)
        dif = tasa_calculada - indec_val if not np.isnan(indec_val) else np.nan

        registros.append({
            "periodo": periodo,
            "calculo_pobreza": round(tasa_calculada, 2),
            "medicion_indec": indec_val,
            "diferencia": round(dif, 2) if not np.isnan(dif) else np.nan
        })

    df_res = pd.DataFrame(registros)

    def calcular_metricas(sub_df: pd.DataFrame, nombre: str):
        diffs = sub_df['diferencia'].dropna()
        pred = sub_df['calculo_pobreza']
        return {
            "Escenario": nombre,
            "N_Semestres": len(pred),
            "Std_Predicciones": round(pred.std(), 3),
            "Media_Diferencia (Sesgo)": round(diffs.mean(), 3),
            "Std_Diferencias": round(diffs.std(), 3),
            "MAE_Diferencia": round(diffs.abs().mean(), 3),
            "Max_Diferencia_Abs": round(diffs.abs().max(), 3)
        }

    # Escenario 1: Todos los semestres
    res_todos = calcular_metricas(df_res, "Todos los semestres")

    # Escenario 2: Excluyendo 2016-S1 y 2020-S1
    periodos_excluir = ['2016-S1', '2020-S1']
    df_filtrado = df_res[~df_res['periodo'].isin(periodos_excluir)]
    res_filtrado = calcular_metricas(df_filtrado, "Excluyendo 2016-S1 y 2020-S1")

    df_metricas = pd.DataFrame([res_todos, res_filtrado])
    return df_res, df_metricas

if __name__ == "__main__":
    tabla_semestres, tabla_resumen = auditar_pobreza_hogares()
    
    print("\n--- SERIE TEMPORAL DE POBREZA EN HOGARES (%) ---")
    print(tabla_semestres.to_string(index=False))
    
    print("\n--- COMPARACIÓN DE DESVIACIONES Y DISPERSIÓN ---")
    print(tabla_resumen.to_string(index=False))

    # Guardar las tablas en archivos CSV
    output_dir = Path("data") # O la carpeta que prefieras
    output_dir.mkdir(parents=True, exist_ok=True)

    tabla_semestres.to_csv(output_dir / "serie_pobreza_hogares.csv", index=False, encoding="utf-8-sig")
    tabla_resumen.to_csv(output_dir / "resumen_metricas_pobreza.csv", index=False, encoding="utf-8-sig")
    
    print(f"\nArchivos CSV guardados exitosamente en: {output_dir.resolve()}")
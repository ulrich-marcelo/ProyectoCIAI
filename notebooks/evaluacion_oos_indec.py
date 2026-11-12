"""
===============================================================================
PROYECTO: Modelado de Pobreza con EPH (Machine Learning)
SCRIPT: Evaluación OOS (2022+) y Comparación Semestral contra INDEC Oficial
===============================================================================
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. SERIE HISTÓRICA OFICIAL DE POBREZA EN HOGARES (INDEC)
# -----------------------------------------------------------------------------
INDEC_HOGARES_OFICIAL = {
    "2016-S1": 23.1, "2016-S2": 21.5,
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

PRED_DIR = Path("notebooks/predicciones")
OUT_DIR = Path("notebooks/resultados")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. FUNCIONES DE CÁLCULO DE MÉTRICAS MICRO (NIVEL HOGAR)
# -----------------------------------------------------------------------------
def evaluar_clasificacion_micro(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    acc = (tp + tn) / (tp + tn + fp + fn)

    metricas = {
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Especificidad": round(especificidad, 4),
        "F1-Score": round(f1, 4),
        "F2-Score": round(f2, 4)
    }

    if y_prob is not None:
        p_prec, p_rec, _ = precision_recall_curve(y_true, y_prob)
        metricas["PR-AUC"] = round(float(auc(p_rec, p_prec)), 4)
        metricas["ROC-AUC"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        metricas["Brier-Score"] = round(float(brier_score_loss(y_true, y_prob)), 4)

    return metricas


# -----------------------------------------------------------------------------
# 3. FUNCIONES DE CÁLCULO DE MÉTRICAS MACRO (NIVEL SEMESTRAL)
# -----------------------------------------------------------------------------
def calcular_metricas_macro(serie_estimada: pd.Series, serie_oficial: pd.Series) -> Dict[str, float]:
    df_comp = pd.DataFrame({"est": serie_estimada, "oficial": serie_oficial}).dropna()
    if len(df_comp) == 0:
        return {}

    diff = df_comp["est"] - df_comp["oficial"]
    mae = diff.abs().mean()
    rmse = np.sqrt((diff ** 2).mean())
    mbe = diff.mean()  # Sesgo medio (+ sobreestima, - subestima)
    max_err = diff.abs().max()

    # Precisión Direccional (% acierto en el signo del cambio semestral)
    d_oficial = df_comp["oficial"].diff().dropna()
    d_est = df_comp["est"].diff().dropna()
    
    if len(d_oficial) > 0:
        dir_acc = np.mean(np.sign(d_oficial) == np.sign(d_est)) * 100.0
    else:
        dir_acc = np.nan

    return {
        "N_Semestres": len(df_comp),
        "MAE (pp)": round(mae, 2),
        "RMSE (pp)": round(rmse, 2),
        "MBE_Sesgo (pp)": round(mbe, 2),
        "Max_Abs_Error (pp)": round(max_err, 2),
        "Precision_Direccional (%)": round(dir_acc, 1) if not np.isnan(dir_acc) else np.nan
    }


# -----------------------------------------------------------------------------
# 4. PROCESAMIENTO PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    archivos_pred = list(PRED_DIR.glob("predicciones_oos_*.parquet"))
    if not archivos_pred:
        raise FileNotFoundError(f"No se encontraron archivos en la carpeta '{PRED_DIR.resolve()}'.")

    print("=" * 80)
    print(f"EVALUACIÓN OUT-OF-SAMPLE (OOS) - COMPARACIÓN CON INDEC")
    print(f"Archivos de predicción detectados: {[f.name for f in archivos_pred]}")
    print("=" * 80)

    # 1. Cargar y consolidar predicciones de todas las familias
    dfs_familia = {}
    for p_file in archivos_pred:
        familia = p_file.stem.replace("predicciones_oos_", "")
        df_f = pd.read_parquet(p_file)
        df_f["semestre"] = np.where(df_f["trimestre"].isin([1, 2]), 1, 2)
        df_f["periodo"] = df_f["ano4"].astype(str) + "-S" + df_f["semestre"].astype(str)
        dfs_familia[familia] = df_f

    # Tomar dataset base para columnas comunes
    familia_base = list(dfs_familia.keys())[0]
    df_base = dfs_familia[familia_base]
    tiene_pobre_real = "pobre" in df_base.columns and df_base["pobre"].notna().any()
    tiene_pondih = "pondih" in df_base.columns and df_base["pondih"].notna().all()

    # 2. Resumen Micro a nivel hogar
    filas_micro = []
    if tiene_pobre_real:
        y_true = df_base["pobre"].values.astype(int)
        for fam, df_f in dfs_familia.items():
            for cfg in ["cfg1", "cfg2", "cfg3"]:
                col_pred = f"pred_{fam}_{cfg}"
                col_prob = f"prob_{fam}_{cfg}"
                if col_pred in df_f.columns:
                    m = evaluar_clasificacion_micro(y_true, df_f[col_pred].values, df_f.get(col_prob, None))
                    m["Modelo"] = f"{fam}_{cfg}"
                    m["Familia"] = fam
                    m["Tipo"] = "Individual"
                    filas_micro.append(m)

            col_ens_vot = f"pred_{fam}_ensamble_votacion"
            col_ens_prob = f"prob_{fam}_ensamble_promedio"
            if col_ens_vot in df_f.columns:
                m = evaluar_clasificacion_micro(y_true, df_f[col_ens_vot].values, df_f.get(col_ens_prob, None))
                m["Modelo"] = f"{fam}_ensamble_votacion"
                m["Familia"] = fam
                m["Tipo"] = "Ensamble Votación"
                filas_micro.append(m)

        df_micro = pd.DataFrame(filas_micro)
        cols_orden_micro = ["Familia", "Modelo", "Tipo", "F2-Score", "F1-Score", "PR-AUC", "ROC-AUC", "Precision", "Recall", "Especificidad", "Brier-Score"]
        df_micro = df_micro[[c for c in cols_orden_micro if c in df_micro.columns]]
        df_micro.to_csv(OUT_DIR / "metricas_micro_hogar_oos.csv", index=False)

        print("\n" + "-" * 80)
        print("MÉTRICAS MICRO A NIVEL HOGAR EN OOS (Clasificación Individual y Ensamble):")
        print("-" * 80)
        print(df_micro.to_string(index=False))

    # 3. Agregación Semestral Ponderada con PONDIH
    periodos_oos = sorted(df_base["periodo"].unique())
    print("\n" + "-" * 80)
    print(f"SERIE TEMPORAL DE POBREZA EN HOGARES (%) PONDERADA CON PONDIH (Período: {periodos_oos[0]} a {periodos_oos[-1]}):")
    print("-" * 80)

    tasas_semestrales = []
    for periodo in periodos_oos:
        fila = {"Periodo": periodo}
        sub_base = df_base[df_base["periodo"] == periodo]
        w_base = sub_base["pondih"].values if tiene_pondih else None

        # Tasa Real EPH en la muestra (ponderada)
        if tiene_pobre_real:
            tasa_real_eph = np.average(sub_base["pobre"], weights=w_base) * 100.0
            fila["EPH_Muestra_Real"] = round(tasa_real_eph, 2)

        # Dato oficial INDEC
        fila["INDEC_Oficial"] = INDEC_HOGARES_OFICIAL.get(periodo, np.nan)

        # Tasas predichas para cada familia
        for fam, df_f in dfs_familia.items():
            sub_f = df_f[df_f["periodo"] == periodo]
            w_f = sub_f["pondih"].values if tiene_pondih else None

            # Modelos individuales
            for cfg in ["cfg1", "cfg2", "cfg3"]:
                col_p = f"pred_{fam}_{cfg}"
                if col_p in sub_f.columns:
                    t_cfg = np.average(sub_f[col_p], weights=w_f) * 100.0
                    fila[f"{fam}_{cfg}"] = round(t_cfg, 2)

            # Ensamble Votación
            col_v = f"pred_{fam}_ensamble_votacion"
            if col_v in sub_f.columns:
                t_vot = np.average(sub_f[col_v], weights=w_f) * 100.0
                fila[f"{fam}_votacion"] = round(t_vot, 2)

            # Ensamble Continuo (Soft)
            col_s = f"prob_{fam}_ensamble_promedio"
            if col_s in sub_f.columns:
                t_soft = np.average(sub_f[col_s], weights=w_f) * 100.0
                fila[f"{fam}_soft"] = round(t_soft, 2)

        tasas_semestrales.append(fila)

    df_tasas = pd.DataFrame(tasas_semestrales)
    df_tasas.to_csv(OUT_DIR / "tasas_semestrales_pobreza_hogares.csv", index=False)
    print(df_tasas.to_string(index=False))

    # 4. Cálculo de Métricas Macro frente al dato oficial de INDEC
    serie_indec = df_tasas.set_index("Periodo")["INDEC_Oficial"]
    filas_macro = []

    # Evaluar la propia muestra EPH si está disponible
    if "EPH_Muestra_Real" in df_tasas.columns:
        m_eph = calcular_metricas_macro(df_tasas.set_index("Periodo")["EPH_Muestra_Real"], serie_indec)
        m_eph["Modelo"] = "EPH_Muestra_Real"
        m_eph["Familia"] = "Benchmark_EPH"
        filas_macro.append(m_eph)

    # Evaluar cada columna predicha
    cols_predichas = [c for c in df_tasas.columns if c not in ["Periodo", "INDEC_Oficial", "EPH_Muestra_Real"]]
    for col in cols_predichas:
        m = calcular_metricas_macro(df_tasas.set_index("Periodo")[col], serie_indec)
        m["Modelo"] = col
        m["Familia"] = col.split("_")[0]
        filas_macro.append(m)

    df_macro = pd.DataFrame(filas_macro)
    cols_orden_macro = ["Familia", "Modelo", "MAE (pp)", "RMSE (pp)", "MBE_Sesgo (pp)", "Max_Abs_Error (pp)", "Precision_Direccional (%)", "N_Semestres"]
    df_macro = df_macro[[c for c in cols_orden_macro if c in df_macro.columns]].sort_values(by="MAE (pp)")
    df_macro.to_csv(OUT_DIR / "metricas_macro_vs_indec.csv", index=False)

    print("\n" + "-" * 80)
    print("MÉTRICAS MACRO-TEMPORALES VS INDEC OFICIAL (Ordenadas por menor MAE):")
    print("-" * 80)
    print(df_macro.to_string(index=False))

    # 5. Visualización: Gráfico de Series de Tiempo
    plt.figure(figsize=(12, 6), dpi=300)
    
    # Línea INDEC Oficial
    plt.plot(df_tasas["Periodo"], df_tasas["INDEC_Oficial"], marker="o", color="black", linewidth=2.5, linestyle="--", label="INDEC Oficial (Hogares)")

    # EPH Muestra Real
    if "EPH_Muestra_Real" in df_tasas.columns:
        plt.plot(df_tasas["Periodo"], df_tasas["EPH_Muestra_Real"], marker="s", color="gray", linewidth=1.8, alpha=0.7, label="EPH Real (Ponderada)")

    colores = {"xgboost": "#1f77b4", "random_forest": "#2ca02c", "logistic": "#d62728", "mlp": "#9467bd"}
    
    # Graficar el ensamble por votación de cada familia disponible
    for fam in dfs_familia.keys():
        col_vot = f"{fam}_votacion"
        if col_vot in df_tasas.columns:
            plt.plot(df_tasas["Periodo"], df_tasas[col_vot], marker="^", linewidth=2.0, color=colores.get(fam, None), label=f"Ensamble Votación ({fam.upper()})")

    plt.title("Evolución de la Tasa de Pobreza en Hogares: INDEC Oficial vs. Modelos OOS", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Semestre", fontsize=11, fontweight="bold")
    plt.ylabel("Tasa de Pobreza en Hogares (%)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()

    fig_path = OUT_DIR / "evolucion_pobreza_hogares.png"
    plt.savefig(fig_path)
    plt.close()

    print("\n" + "=" * 80)
    print(f"✓ Evaluación finalizada con éxito.")
    print(f"  - Reportes exportados a: {OUT_DIR.resolve()}/")
    print(f"  - Gráfico guardado en:   {fig_path.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
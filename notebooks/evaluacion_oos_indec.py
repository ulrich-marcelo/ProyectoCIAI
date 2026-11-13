from pathlib import Path
from typing import Dict, List, Optional
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

SCALE_POS_FACTORS = {
    "xgboost": {"cfg1": 1.2, "cfg2": 0.8, "cfg3": 1.0},
    "random_forest": {"cfg1": 1.0, "cfg2": 1.0, "cfg3": 1.0},
    "logistic": {"cfg1": 1.0, "cfg2": 1.0, "cfg3": 1.0},
    "mlp": {"cfg1": 1.0, "cfg2": 1.0, "cfg3": 1.0},
}

PRED_DIR = Path("predicciones") if Path("predicciones").exists() else Path("notebooks/predicciones")
OUT_DIR = Path("resultados") if Path("predicciones").exists() else Path("notebooks/resultados")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_CANDIDATES = [
    Path("data/eph_hogares_consolidada_2016_2026.parquet"),
    Path("../data/eph_hogares_consolidada_2016_2026.parquet"),
]

def obtener_scale_pos_train() -> float:
    for p_path in PARQUET_CANDIDATES:
        if p_path.exists():
            df = pd.read_parquet(p_path, columns=["codusu", "nro_hogar", "ano4", "pobre"])
            df["id_hogar"] = df["codusu"].astype(str) + "_" + df["nro_hogar"].astype(str)
            df_tr_raw = df[df["ano4"] <= 2021]
            hogares_te = set(df[df["ano4"] > 2021]["id_hogar"].unique())
            df_tr = df_tr_raw[~df_tr_raw["id_hogar"].isin(hogares_te)]
            y_tr = df_tr["pobre"].values
            scale_pos = float((len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1))
            return scale_pos
    return 2.70

def recalibrar_probabilidad_bayes(probs_pond: np.ndarray, peso_efectivo: float) -> np.ndarray:
    p = np.clip(probs_pond.astype(np.float64), 1e-7, 1.0 - 1e-7)
    p_cal = p / (p + peso_efectivo * (1.0 - p))
    return p_cal.astype(np.float32)

def promediar_en_espacio_logit(lista_probs: List[np.ndarray]) -> np.ndarray:
    logits = []
    for p in lista_probs:
        p_clip = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
        logits.append(np.log(p_clip / (1.0 - p_clip)))
    logit_medio = np.mean(logits, axis=0)
    probs_ens = 1.0 / (1.0 + np.exp(-logit_medio))
    return probs_ens.astype(np.float32)

def evaluar_clasificacion_micro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = fbeta_score(y_true, y_pred, beta=2, sample_weight=sample_weight, zero_division=0)
    acc = (tp + tn) / (tp + tn + fp + fn)

    metricas = {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "Especificidad": round(float(especificidad), 4),
        "F1-Score": round(float(f1), 4),
        "F2-Score": round(float(f2), 4)
    }

    if y_prob is not None:
        p_prec, p_rec, _ = precision_recall_curve(y_true, y_prob, sample_weight=sample_weight)
        metricas["PR-AUC"] = round(float(auc(p_rec, p_prec)), 4)
        metricas["ROC-AUC"] = round(float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight)), 4)
        metricas["Brier-Score"] = round(float(brier_score_loss(y_true, y_prob, sample_weight=sample_weight)), 4)

    return metricas

def graficar_matrices_confusion_pre_calibracion(
    dfs_familia: Dict[str, pd.DataFrame],
    y_true: np.ndarray,
    sample_weight: Optional[np.ndarray],
    out_dir: Path
) -> List[Path]:
    rutas_generadas = []
    etiquetas = ["No Pobre (0)", "Pobre (1)"]

    for fam, df_f in dfs_familia.items():
        col_pred = f"pred_{fam}_ensamble_votacion"
        if col_pred not in df_f.columns:
            continue
            
        y_pred_raw = df_f[col_pred].values.astype(int)
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        cm = confusion_matrix(y_true, y_pred_raw, sample_weight=sample_weight)
        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0) * 100.0

        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f2 = fbeta_score(y_true, y_pred_raw, beta=2, sample_weight=sample_weight, zero_division=0)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("% por clase real", fontsize=8)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_yticklabels(etiquetas, fontsize=9)
        ax.set_xlabel("Predicción (Antes de Calibrar - Votación F2)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Clase Real (EPH)", fontsize=9, fontweight="bold")
        
        nota_peso = "Ponderada por PONDIH" if sample_weight is not None else "Sin ponderar"
        ax.set_title(
            f"{fam.upper()} - Matriz OOS ({nota_peso})\nAcc: {acc:.1%} | Prec: {prec:.1%} | Rec: {rec:.1%} | F2: {f2:.3f}",
            fontsize=10,
            fontweight="bold",
            pad=10
        )

        for i in range(2):
            for j in range(2):
                valor_abs = cm[i, j]
                valor_pct = cm_norm[i, j]
                color_txt = "white" if valor_pct > 55.0 else "black"
                texto_celda = f"{valor_pct:.1f}%\n({valor_abs:,.0f})"
                ax.text(
                    j, i, texto_celda,
                    ha="center", va="center",
                    color=color_txt, fontsize=9.5, fontweight="bold"
                )

        plt.tight_layout()
        fpath = out_dir / f"matriz_confusion_pre_cal_{fam}.png"
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        rutas_generadas.append(fpath)
        
    return rutas_generadas

def calcular_metricas_macro(serie_estimada: pd.Series, serie_referencia: pd.Series) -> Dict[str, float]:
    df_comp = pd.DataFrame({"est": serie_estimada, "ref": serie_referencia}).dropna()
    if len(df_comp) == 0:
        return {}

    diff = df_comp["est"] - df_comp["ref"]
    mae = diff.abs().mean()
    rmse = np.sqrt((diff ** 2).mean())
    mbe = diff.mean() 
    max_err = diff.abs().max()

    d_ref = df_comp["ref"].diff().dropna()
    d_est = df_comp["est"].diff().dropna()

    if len(d_ref) > 0:
        dir_acc = np.mean(np.sign(d_ref) == np.sign(d_est)) * 100.0
    else:
        dir_acc = np.nan

    return {
        "N_Semestres": int(len(df_comp)),
        "MAE (pp)": round(float(mae), 2),
        "RMSE (pp)": round(float(rmse), 2),
        "MBE_Sesgo (pp)": round(float(mbe), 2),
        "Max_Abs_Error (pp)": round(float(max_err), 2),
        "Precision_Direccional (%)": round(float(dir_acc), 1) if not np.isnan(dir_acc) else np.nan
    }

def main():
    archivos_pred = sorted(list(PRED_DIR.glob("predicciones_oos_*.parquet")))
    if not archivos_pred:
        raise FileNotFoundError(f"No se encontraron archivos en '{PRED_DIR.resolve()}'.")

    scale_pos_base = obtener_scale_pos_train()

    print("=" * 88)
    print("EVALUACIÓN OUT-OF-SAMPLE (OOS) - GRÁFICOS INDIVIDUALES")
    print(f"Carpeta de predicciones: {PRED_DIR.resolve()}")
    print("=" * 88)

    dfs_familia: Dict[str, pd.DataFrame] = {}
    probs_cal_lideres: Dict[str, np.ndarray] = {}

    for p_file in archivos_pred:
        familia = p_file.stem.replace("predicciones_oos_", "")
        df_f = pd.read_parquet(p_file)
        df_f["semestre"] = np.where(df_f["trimestre"].isin([1, 2]), 1, 2)
        df_f["periodo"] = df_f["ano4"].astype(str) + "-S" + df_f["semestre"].astype(str)

        probs_cal_cfgs = []
        hard_sym_cfgs = []

        for cfg in ["cfg1", "cfg2", "cfg3"]:
            col_prob = f"prob_{familia}_{cfg}"
            if col_prob in df_f.columns:
                factor = SCALE_POS_FACTORS.get(familia, {}).get(cfg, 1.0)
                w_eff = scale_pos_base * factor
                p_cal = recalibrar_probabilidad_bayes(df_f[col_prob].values, w_eff)

                df_f[f"prob_{familia}_{cfg}_cal"] = p_cal
                pred_sym = (p_cal >= 0.50).astype(np.int8)
                df_f[f"pred_{familia}_{cfg}_sym50"] = pred_sym

                probs_cal_cfgs.append(p_cal)
                hard_sym_cfgs.append(pred_sym)

        if probs_cal_cfgs:
            df_f[f"prob_{familia}_ensamble_cal"] = np.mean(probs_cal_cfgs, axis=0).astype(np.float32)
            df_f[f"prob_{familia}_ensamble_logit_cal"] = promediar_en_espacio_logit(probs_cal_cfgs)
            votos_sym = np.sum(hard_sym_cfgs, axis=0)
            df_f[f"pred_{familia}_votacion_sym50"] = (votos_sym >= 2).astype(np.int8)

            if f"prob_{familia}_cfg1_cal" in df_f.columns:
                probs_cal_lideres[familia] = df_f[f"prob_{familia}_cfg1_cal"].values

        dfs_familia[familia] = df_f

    familia_base = list(dfs_familia.keys())[0]
    df_base = dfs_familia[familia_base]
    tiene_pobre_real = "pobre" in df_base.columns and df_base["pobre"].notna().any()
    tiene_pondih = "pondih" in df_base.columns and df_base["pondih"].notna().all()
    w_micro = df_base["pondih"].values if tiene_pondih else None

    tiene_super_ensamble = len(probs_cal_lideres) >= 2
    if tiene_super_ensamble:
        lista_lideres = list(probs_cal_lideres.values())
        p_super_prob = np.mean(lista_lideres, axis=0).astype(np.float32)
        p_super_logit = promediar_en_espacio_logit(lista_lideres)
        pred_super_sym50 = (p_super_logit >= 0.50).astype(np.int8)

        df_base["prob_super_ensamble_cal"] = p_super_prob
        df_base["prob_super_ensamble_logit_cal"] = p_super_logit
        df_base["pred_super_ensamble_sym50"] = pred_super_sym50

    filas_micro = []
    rutas_cm = []
    if tiene_pobre_real:
        y_true = df_base["pobre"].values.astype(int)

        # Generar matrices de confusión gráficas INDIVIDUALES
        rutas_cm = graficar_matrices_confusion_pre_calibracion(
            dfs_familia=dfs_familia,
            y_true=y_true,
            sample_weight=w_micro,
            out_dir=OUT_DIR
        )

        for fam, df_f in dfs_familia.items():
            col_vot_f2 = f"pred_{fam}_ensamble_votacion"
            col_prob_raw = f"prob_{fam}_ensamble_promedio"
            if col_vot_f2 in df_f.columns:
                m = evaluar_clasificacion_micro(y_true, df_f[col_vot_f2].values, df_f.get(col_prob_raw), w_micro)
                m.update({"Familia": fam, "Modelo": f"{fam}_votacion_F2", "Regimen": "A: Focalización (F2)"})
                filas_micro.append(m)

            col_vot_sym = f"pred_{fam}_votacion_sym50"
            col_prob_cal = f"prob_{fam}_ensamble_logit_cal"
            if col_vot_sym in df_f.columns:
                m = evaluar_clasificacion_micro(y_true, df_f[col_vot_sym].values, df_f.get(col_prob_cal), w_micro)
                m.update({"Familia": fam, "Modelo": f"{fam}_votacion_sym50", "Regimen": "B: Medición (p_cal>=0.50)"})
                filas_micro.append(m)

        if tiene_super_ensamble:
            m_sup = evaluar_clasificacion_micro(
                y_true,
                df_base["pred_super_ensamble_sym50"].values,
                df_base["prob_super_ensamble_logit_cal"].values,
                w_micro
            )
            m_sup.update({
                "Familia": "INTER-FAMILIA",
                "Modelo": "super_ensamble_logit_sym50",
                "Regimen": "B: Medición (p_cal>=0.50)"
            })
            filas_micro.append(m_sup)

        df_micro = pd.DataFrame(filas_micro)
        df_micro.to_csv(OUT_DIR / "metricas_micro_hogar_oos.csv", index=False)

    periodos_oos = sorted(df_base["periodo"].unique())
    tasas_semestrales = []

    for periodo in periodos_oos:
        fila = {"Periodo": periodo}
        sub_base = df_base[df_base["periodo"] == periodo]
        w_base = sub_base["pondih"].values if tiene_pondih else None

        if tiene_pobre_real:
            fila["EPH_Muestra_Real"] = round(float(np.average(sub_base["pobre"], weights=w_base) * 100.0), 2)

        fila["INDEC_Oficial"] = INDEC_HOGARES_OFICIAL.get(periodo, np.nan)

        for fam, df_f in dfs_familia.items():
            sub_f = df_f[df_f["periodo"] == periodo]
            w_f = sub_f["pondih"].values if tiene_pondih else None

            if f"pred_{fam}_ensamble_votacion" in sub_f.columns:
                fila[f"{fam}_hard_F2"] = round(float(np.average(sub_f[f"pred_{fam}_ensamble_votacion"], weights=w_f) * 100.0), 2)

            if f"pred_{fam}_votacion_sym50" in sub_f.columns:
                fila[f"{fam}_hard_sym50"] = round(float(np.average(sub_f[f"pred_{fam}_votacion_sym50"], weights=w_f) * 100.0), 2)

            if f"prob_{fam}_ensamble_cal" in sub_f.columns:
                fila[f"{fam}_soft_cal"] = round(float(np.average(sub_f[f"prob_{fam}_ensamble_cal"], weights=w_f) * 100.0), 2)

        if tiene_super_ensamble:
            fila["super_hard_sym50"] = round(float(np.average(sub_base["pred_super_ensamble_sym50"], weights=w_base) * 100.0), 2)
            fila["super_soft_cal"] = round(float(np.average(sub_base["prob_super_ensamble_cal"], weights=w_base) * 100.0), 2)
            
        tasas_semestrales.append(fila)

    df_tasas = pd.DataFrame(tasas_semestrales)
    df_tasas.to_csv(OUT_DIR / "tasas_semestrales_pobreza_hogares.csv", index=False)

    colores = {"xgboost": "#1f77b4", "random_forest": "#2ca02c", "logistic": "#d62728", "mlp": "#9467bd"}

    # GRÁFICO INDIVIDUAL 1: Régimen A (Hard F2 original)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(df_tasas["Periodo"], df_tasas["INDEC_Oficial"], "o--", color="black", lw=2.5, label="INDEC Oficial")
    if "EPH_Muestra_Real" in df_tasas.columns:
        plt.plot(df_tasas["Periodo"], df_tasas["EPH_Muestra_Real"], "s-", color="gray", lw=1.6, alpha=0.75, label="EPH Real")
    for fam in dfs_familia.keys():
        if f"{fam}_hard_F2" in df_tasas.columns:
            plt.plot(df_tasas["Periodo"], df_tasas[f"{fam}_hard_F2"], "^-", lw=1.8, color=colores.get(fam), label=f"{fam.upper()} (Hard F2)")
    plt.title("Evolución Pobreza - Régimen A: Focalización (Umbral F2)", fontsize=11, fontweight="bold")
    plt.ylabel("Tasa de Pobreza en Hogares (%)", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend(fontsize=8.5, frameon=True)
    plt.tight_layout()
    fig_path_a = OUT_DIR / "evolucion_pobreza_regimen_A.png"
    plt.savefig(fig_path_a, bbox_inches="tight")
    plt.close()

    # GRÁFICO INDIVIDUAL 2: Régimen B/C (Soft Calibrado)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(df_tasas["Periodo"], df_tasas["INDEC_Oficial"], "o--", color="black", lw=2.5, label="INDEC Oficial")
    if "EPH_Muestra_Real" in df_tasas.columns:
        plt.plot(df_tasas["Periodo"], df_tasas["EPH_Muestra_Real"], "s-", color="gray", lw=1.6, alpha=0.75, label="EPH Real")
    for fam in dfs_familia.keys():
        if f"{fam}_soft_cal" in df_tasas.columns:
            plt.plot(df_tasas["Periodo"], df_tasas[f"{fam}_soft_cal"], "^-", lw=1.8, color=colores.get(fam), label=f"{fam.upper()} (Soft Cal.)")
    if "super_soft_cal" in df_tasas.columns:
        plt.plot(df_tasas["Periodo"], df_tasas["super_soft_cal"], "D-", color="#ff7f0e", lw=2.4, label="SUPER-ENSAMBLE (Soft Cal.)")
    plt.title("Evolución Pobreza - Régimen B/C: Medición Macro Recalibrada", fontsize=11, fontweight="bold")
    plt.ylabel("Tasa de Pobreza en Hogares (%)", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend(fontsize=8.5, frameon=True)
    plt.tight_layout()
    fig_path_bc = OUT_DIR / "evolucion_pobreza_regimen_BC.png"
    plt.savefig(fig_path_bc, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
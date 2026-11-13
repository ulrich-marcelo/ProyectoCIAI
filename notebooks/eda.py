"""
===============================================================================
PROYECTO: Predicción de Pobreza mediante Machine Learning (EPH 2016-2026)
AUTOR: Marcelo Gabriel Ulrich | EEyN - UNSAM
DIRECTORES: César Germán Santamaría | Agustín Moro
===============================================================================
Módulo de Análisis Exploratorio de Datos (EDA) sobre Corpus Post-Limpieza:
1. Anatomía del corpus (53 features por Dimensión Socioeconómica y Tipo) +
   Contraste semestral Target Reconstruido (EPH) vs. INDEC Oficial (2016-2026).
2. Ranking univariado de poder discriminativo: Information Value (IV) ponderado.
3. Dimensión Demográfica: Tamaño del hogar, composición etaria y dependencia.
4. Dimensiones Laboral y Capital Humano: Educación x Informalidad y Salud Pública.
5. Dimensiones Hábitat y Regional: Brechas en 7 regiones y carencias habitacionales.
6. Estructura de multicolinealidad (Spearman) agrupada por Dimensión Conceptual.
===============================================================================
"""

from pathlib import Path
from typing import Dict, List
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURACIÓN ESTÉTICA GENERAL (PUBLICACIÓN / LATEX)
# -----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.titlesize"] = 11.5
plt.rcParams["axes.labelsize"] = 10.5

PARQUET_CANDIDATES = [
    Path("data/eph_hogares_consolidada_2016_2026.parquet"),
    Path("../data/eph_hogares_consolidada_2016_2026.parquet"),
]
PARQUET_PATH = next((p for p in PARQUET_CANDIDATES if p.exists()), PARQUET_CANDIDATES[0])

BASE_OUT = Path("reports") if Path("data").exists() else Path("../reports")
OUTPUT_FIGS = BASE_OUT / "figures"
OUTPUT_TABLES = BASE_OUT / "tables"
OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

# Serie histórica oficial de pobreza en hogares (INDEC)
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

# -----------------------------------------------------------------------------
# DEFINICIÓN DE DIMENSIONES SOCIOECONÓMICAS Y TIPOS ESTADÍSTICOS
# -----------------------------------------------------------------------------
COLS_EXCLUIR = {
    "codusu", "ano4", "trimestre", "nro_hogar", "pondih", "linea_pobreza", "pobre"
}

DIMENSIONES_SOCIOECONOMICAS: Dict[str, List[str]] = {
    "Estructura Demográfica": [
        "cant_miembros", "cant_menores_14", "cant_adultos_mayores",
        "cant_edad_activa", "adequi_hogar",
        "ratio_dependencia_demog", "ratio_dependencia_econom",
        "jefe_mujer", "jefe_edad"
    ],
    "Inserción Laboral": [
        "cant_ocupados", "cant_desocupados", "cant_activos",
        "cant_asalariados", "cant_informales",
        "tasa_informalidad_hogar", "tasa_desocupacion_hogar",
        "jefe_ocupado", "jefe_desocupado", "jefe_informal"
    ],
    "Capital Humano y Salud": [
        "max_educ_hogar", "jefe_nivel_ed", "prop_salud_publica"
    ],
    "Hábitat y Calidad Habitacional": [
        "iv1", "iv2", "iv3", "iv4", "iv5", "iv6", "iv7", "iv8", "iv9",
        "iv10", "iv11", "iv12_1", "iv12_2", "iv12_3",
        "ii1", "ii2", "ii3", "ii3_1", "ii4_1", "ii4_2", "ii4_3",
        "ii5", "ii5_1", "ii6", "ii6_1", "ii7", "ii8", "ii9",
        "personas_por_cuarto", "hacinamiento_critico"
    ],
    "Contexto Regional": [
        "region_estudio"
    ]
}

VAR_TO_DIM = {v: dim for dim, vlist in DIMENSIONES_SOCIOECONOMICAS.items() for v in vlist}

PALETA_DIMENSIONES = {
    "Estructura Demográfica": "#1f77b4",
    "Inserción Laboral": "#ff7f0e",
    "Capital Humano y Salud": "#2ca02c",
    "Hábitat y Calidad Habitacional": "#8c564b",
    "Contexto Regional": "#9467bd"
}

NOMINALES_Y_ENCUESTA = {
    "region_estudio", "iv1", "iv3", "iv4", "iv5", "iv6", "iv8", "iv9",
    "iv10", "iv11", "iv12_1", "iv12_2", "iv12_3", "ii3", "ii3_1",
    "ii4_1", "ii4_2", "ii4_3", "ii5", "ii5_1", "ii6", "ii6_1", "ii7", "ii8"
}
ORDINALES = {"max_educ_hogar", "jefe_nivel_ed", "iv7", "ii9"}
BINARIAS_PURAS = {"jefe_mujer", "jefe_ocupado", "jefe_desocupado", "jefe_informal", "hacinamiento_critico"}

NOMBRES_LEGIBLES = {
    "region_estudio": "Región Estadística (7 zonas)",
    "max_educ_hogar": "Máximo Nivel Educativo del Hogar",
    "jefe_nivel_ed": "Nivel Educativo del Jefe/a",
    "prop_salud_publica": "Prop. Salud Pública Exclusiva",
    "tasa_informalidad_hogar": "Tasa de Informalidad Laboral",
    "tasa_desocupacion_hogar": "Tasa de Desocupación del Hogar",
    "jefe_ocupado": "Jefe/a Ocupado/a",
    "jefe_desocupado": "Jefe/a Desocupado/a",
    "jefe_informal": "Jefe/a Asalariado/a Informal",
    "cant_ocupados": "Cantidad de Ocupados",
    "cant_desocupados": "Cantidad de Desocupados",
    "cant_activos": "Cantidad de Activos",
    "cant_asalariados": "Cantidad de Asalariados",
    "cant_informales": "Cantidad de Informales",
    "cant_miembros": "Tamaño del Hogar (Miembros)",
    "cant_menores_14": "Cantidad de Menores (<14 años)",
    "cant_adultos_mayores": "Cantidad de Adultos Mayores (>=65)",
    "cant_edad_activa": "Miembros en Edad Activa (15-64)",
    "adequi_hogar": "Adultos Equivalentes del Hogar",
    "ratio_dependencia_demog": "Ratio de Dependencia Demográfica",
    "ratio_dependencia_econom": "Ratio de Dependencia Económica",
    "jefe_mujer": "Jefatura de Hogar Femenina",
    "jefe_edad": "Edad del Jefe/a de Hogar",
    "personas_por_cuarto": "Personas por Cuarto",
    "hacinamiento_critico": "Hacinamiento Crítico (>3 pers/cuarto)",
    "iv1": "Tipo de Vivienda",
    "iv2": "Cantidad de Ambientes",
    "iv3": "Material de los Pisos",
    "iv4": "Material de la Cubierta/Techo",
    "iv5": "Presencia de Cielorraso",
    "iv6": "Procedencia del Agua",
    "iv7": "Acceso al Agua (Cañería)",
    "iv8": "Tiene Baño / Letrina",
    "iv9": "Ubicación del Baño",
    "iv10": "Tipo de Desagüe Cloacal",
    "iv11": "Desagüe del Baño",
    "iv12_1": "Cerca de Basural",
    "iv12_2": "Zona Inundable",
    "iv12_3": "Villa de Emergencia",
    "ii1": "Cuartos de Uso Exclusivo",
    "ii2": "Cuartos para Dormir",
    "ii3": "Usa Cuartos para Trabajar",
    "ii3_1": "Cant. Cuartos para Trabajar",
    "ii4_1": "Tiene Cuarto de Cocina",
    "ii4_2": "Tiene Lavadero",
    "ii4_3": "Tiene Garaje",
    "ii5": "Combustible para Cocinar",
    "ii5_1": "Otro Combustible Cocinar",
    "ii6": "Equipamiento Cocina/Heladera",
    "ii6_1": "Detalle Equipamiento",
    "ii7": "Régimen de Tenencia",
    "ii8": "Combustible Utilizado (General)",
    "ii9": "Uso del Baño (Exclusivo/Comp.)"
}


def clasificar_tipo_estadistico(col: str) -> str:
    if col in NOMINALES_Y_ENCUESTA:
        return "Nominal / Encuesta (OHE)"
    if col in ORDINALES:
        return "Ordinal Jerárquica"
    if col in BINARIAS_PURAS:
        return "Binaria Pura (0/1)"
    return "Continua / Conteo / Ratio"


# -----------------------------------------------------------------------------
# FUNCIONES ESTADÍSTICAS PONDERADAS
# -----------------------------------------------------------------------------
def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / max(np.sum(w), 1e-9))


def weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    mu = weighted_mean(x, w)
    var = np.sum(w * (x - mu) ** 2) / max(np.sum(w), 1e-9)
    return float(np.sqrt(max(var, 0.0)))


def calcular_iv_ponderado(serie: pd.Series, y: np.ndarray, w: np.ndarray, es_categorica: bool = False, bins: int = 10) -> float:
    """
    Calcula el Information Value (IV) ponderado por PONDIH.
    Permite medir en una escala unificada el poder discriminativo univariado de
    variables continuas, ordinales, binarias y nominales frente a 'pobre'.
    """
    mask = (w > 0) & (~pd.isna(serie))
    s = serie[mask]
    y_m = y[mask]
    w_m = w[mask]

    if es_categorica or s.nunique() <= 10:
        grupos = s.astype(str)
    else:
        try:
            grupos = pd.qcut(s, q=bins, duplicates="drop").astype(str)
        except Exception:
            grupos = pd.cut(s, bins=min(5, s.nunique()), duplicates="drop").astype(str)

    df_tmp = pd.DataFrame({
        "g": grupos,
        "w_pos": np.where(y_m == 1, w_m, 0.0),
        "w_neg": np.where(y_m == 0, w_m, 0.0),
    })
    agg = df_tmp.groupby("g", observed=False)[["w_pos", "w_neg"]].sum()
    tot_pos = max(agg["w_pos"].sum(), 1e-9)
    tot_neg = max(agg["w_neg"].sum(), 1e-9)

    dist_pos = (agg["w_pos"] + 0.5) / (tot_pos + 0.5 * len(agg))
    dist_neg = (agg["w_neg"] + 0.5) / (tot_neg + 0.5 * len(agg))

    woe = np.log(dist_pos / dist_neg)
    return float(np.sum((dist_pos - dist_neg) * woe))


# -----------------------------------------------------------------------------
# CARGA DEL DATASET CONSOLIDADO
# -----------------------------------------------------------------------------
print(f"[1/7] Cargando dataset consolidado desde: {PARQUET_PATH}")
if not PARQUET_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo Parquet en: {PARQUET_PATH.resolve()}")

df = pd.read_parquet(PARQUET_PATH)
df = df[df["pondih"] > 0].copy().reset_index(drop=True)
df["pobre"] = df["pobre"].astype(int)
df["semestre"] = np.where(df["trimestre"].isin([1, 2]), 1, 2)
df["periodo_sem"] = df["ano4"].astype(str) + "-S" + df["semestre"].astype(str)
df["bloque_temporal"] = np.where(df["ano4"] <= 2021, "Desarrollo (2016–2021)", "Out-of-Sample (2022–2026)")

features = [c for c in df.columns if c not in COLS_EXCLUIR and c in VAR_TO_DIM]
print(f"      ✓ Registros válidos: {len(df):,} hogares | Features analizadas: {len(features)} | Tasa pobreza global (PONDIH): {weighted_mean(df['pobre'].values, df['pondih'].values)*100:.2f}%")


# -----------------------------------------------------------------------------
# FIGURA 1: ANATOMÍA DEL CORPUS (53 FEATURES) + TARGET APROXIMADO VS INDEC
# -----------------------------------------------------------------------------
def generar_fig1_anatomia_y_serie(df_all: pd.DataFrame, feat_list: List[str]):
    print("[2/7] Generando Figura 1: Anatomía del corpus y contraste semestral Target vs. INDEC...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1.0, 1.4]})

    # Panel A: Composición de las 53 features por Dimensión Socioeconómica y Tipo
    ax0 = axes[0]
    rows_comp = [{"Dimension": VAR_TO_DIM[f], "Tipo": clasificar_tipo_estadistico(f)} for f in feat_list]
    df_comp = pd.DataFrame(rows_comp)
    ct = pd.crosstab(df_comp["Dimension"], df_comp["Tipo"])
    orden_dims = [
        "Hábitat y Calidad Habitacional",
        "Inserción Laboral",
        "Estructura Demográfica",
        "Capital Humano y Salud",
        "Contexto Regional"
    ]
    orden_tipos = [
        "Nominal / Encuesta (OHE)",
        "Continua / Conteo / Ratio",
        "Binaria Pura (0/1)",
        "Ordinal Jerárquica"
    ]
    ct = ct.reindex(index=orden_dims, columns=[c for c in orden_tipos if c in ct.columns], fill_value=0)

    colores_tipos = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    ct.plot(kind="barh", stacked=True, ax=ax0, color=colores_tipos, edgecolor="white", width=0.68)
    ax0.invert_yaxis()
    ax0.set_title("A. Composición de las 53 Features por Dimensión y Tipo", fontweight="bold", pad=10)
    ax0.set_xlabel("Cantidad de Variables")
    ax0.set_ylabel("")
    ax0.set_xlim(0, 36)
    ax0.legend(title="Tipo Estadístico", loc="lower right", fontsize=8.5, title_fontsize=9)

    for idx, total in enumerate(ct.sum(axis=1)):
        ax0.text(total + 0.6, idx, f"{int(total)} vars ({total/len(feat_list)*100:.1f}%)", va="center", fontsize=9, fontweight="bold")

    # Panel B: Serie semestral Target Aproximado (EPH PONDIH) vs. INDEC Oficial
    ax1 = axes[1]
    res_sem = []
    for (ano, sem), sub in df_all.groupby(["ano4", "semestre"]):
        per = f"{ano}-S{sem}"
        res_sem.append({
            "ano4": ano,
            "semestre": sem,
            "periodo": per,
            "target_eph_pct": weighted_mean(sub["pobre"].values, sub["pondih"].values) * 100.0,
            "indec_oficial_pct": INDEC_HOGARES_OFICIAL.get(per, np.nan)
        })
    df_s = pd.DataFrame(res_sem).sort_values(["ano4", "semestre"]).reset_index(drop=True)

    idx_corte = df_s[df_s["ano4"] <= 2021].index.max() + 0.5
    ax1.axvspan(-0.5, idx_corte, color="#ecf0f1", alpha=0.75, label="Bloque Desarrollo (2016–2021)")
    ax1.axvspan(idx_corte, len(df_s) - 0.5, color="#fef9e7", alpha=0.85, label="Bloque Out-of-Sample (2022–2026)")
    ax1.axvline(idx_corte, color="#2c3e50", linestyle="--", lw=1.6)

    ax1.plot(
        df_s.index, df_s["indec_oficial_pct"],
        marker="s", markersize=5.5, lw=2.2, linestyle="--", color="black",
        label="INDEC Oficial (Hogares)"
    )
    ax1.plot(
        df_s.index, df_s["target_eph_pct"],
        marker="o", markersize=5.5, lw=2.2, color="#c0392b",
        label="Target Reconstruido (EPH PONDIH)"
    )

    ax1.set_xticks(range(0, len(df_s), 2))
    ax1.set_xticklabels(df_s["periodo"].iloc[::2], rotation=40, ha="right", fontsize=8.5)
    ax1.set_ylabel("Tasa de Pobreza en Hogares (%)", fontweight="bold")
    ax1.set_title("B. Validación del Target Reconstruido frente a INDEC Oficial (Semestral)", fontweight="bold", pad=10)
    ax1.set_ylim(14, 47)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "eda_01_composicion_corpus_dimensiones.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# FIGURA 2: RANKING UNIVARIADO DE PODER DISCRIMINATIVO (IV PONDERADO) + TABLA MAESTRA
# -----------------------------------------------------------------------------
def generar_fig2_ranking_iv_y_tabla(df_all: pd.DataFrame, feat_list: List[str]) -> pd.DataFrame:
    print("[3/7] Generando Figura 2: Poder discriminativo univariado (Information Value)...")
    y = df_all["pobre"].values
    w = df_all["pondih"].values
    mask_pob = y == 1
    mask_nopob = y == 0

    registros = []
    for f in feat_list:
        dim = VAR_TO_DIM[f]
        tipo = clasificar_tipo_estadistico(f)
        es_cat = f in NOMINALES_Y_ENCUESTA
        iv = calcular_iv_ponderado(df_all[f], y, w, es_categorica=es_cat)

        if not es_cat:
            vals = df_all[f].values.astype(float)
            mu_pob = weighted_mean(vals[mask_pob], w[mask_pob])
            mu_nopob = weighted_mean(vals[mask_nopob], w[mask_nopob])
            sd_pool = np.sqrt(0.5 * (weighted_std(vals[mask_pob], w[mask_pob]) ** 2 + weighted_std(vals[mask_nopob], w[mask_nopob]) ** 2))
            cohen_d = (mu_pob - mu_nopob) / max(sd_pool, 1e-9)
            signo = "(+)" if cohen_d > 0.03 else ("(-)" if cohen_d < -0.03 else "(=)")
        else:
            mu_pob, mu_nopob, cohen_d = np.nan, np.nan, np.nan
            signo = "(cat)"

        registros.append({
            "variable": f,
            "etiqueta": NOMBRES_LEGIBLES.get(f, f),
            "dimension": dim,
            "tipo_estadistico": tipo,
            "IV_ponderado": round(iv, 4),
            "media_pobre": round(mu_pob, 3) if not np.isnan(mu_pob) else np.nan,
            "media_no_pobre": round(mu_nopob, 3) if not np.isnan(mu_nopob) else np.nan,
            "cohen_d": round(cohen_d, 3) if not np.isnan(cohen_d) else np.nan,
            "direccion": signo
        })

    df_resumen = pd.DataFrame(registros).sort_values("IV_ponderado", ascending=False).reset_index(drop=True)
    df_resumen.to_csv(OUTPUT_TABLES / "eda_resumen_53_features.csv", index=False)

    top20 = df_resumen.head(20).iloc[::-1]
    colores = [PALETA_DIMENSIONES[d] for d in top20["dimension"]]
    labels = [f"{row['etiqueta']} {row['direccion']}" for _, row in top20.iterrows()]

    fig, ax = plt.subplots(figsize=(12, 7.5))
    bars = ax.barh(labels, top20["IV_ponderado"], color=colores, edgecolor="white", height=0.72)

    ax.axvline(0.10, color="#7f8c8d", linestyle=":", lw=1.2)
    ax.axvline(0.30, color="#2c3e50", linestyle="--", lw=1.3)
    ax.text(0.105, 0.4, "Poder Medio\n(IV = 0.10)", color="#7f8c8d", fontsize=8.5)
    ax.text(0.305, 0.4, "Poder Fuerte\n(IV = 0.30)", color="#2c3e50", fontsize=8.5, fontweight="bold")

    for bar, val in zip(bars, top20["IV_ponderado"]):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8.5)

    handles = [patches.Patch(color=c, label=d) for d, c in PALETA_DIMENSIONES.items()]
    ax.legend(handles=handles, title="Dimensión Socioeconómica", loc="lower right", frameon=True, fontsize=9)
    ax.set_xlabel("Information Value (IV) Ponderado por PONDIH", fontweight="bold")
    ax.set_title("Top 20 Variables con Mayor Poder Discriminativo Univariado frente a Pobreza", fontweight="bold", pad=12)
    ax.set_xlim(0, top20["IV_ponderado"].max() * 1.14)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "eda_02_poder_discriminativo_univariado.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df_resumen


# -----------------------------------------------------------------------------
# FIGURA 3: DIMENSIÓN 1 — ESTRUCTURA DEMOGRÁFICA Y CARGA FAMILIAR
# -----------------------------------------------------------------------------
def generar_fig3_demografia(df_all: pd.DataFrame):
    print("[4/7] Generando Figura 3: Estructura demográfica por tamaño y composición etaria...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel A: Barras agrupadas por Tramo de Tamaño x Composición Etaria clara
    ax0 = axes[0]
    df_d = df_all.copy()

    df_d["tramo_tamano"] = pd.cut(
        df_d["cant_miembros"],
        bins=[0, 2, 4, 99],
        labels=["1–2 miembros\n(Hogares pequeños)", "3–4 miembros\n(Hogares medianos)", "5+ miembros\n(Hogares numerosos)"]
    )

    conds_comp = [
        (df_d["cant_menores_14"] > 0) & (df_d["cant_adultos_mayores"] == 0),
        (df_d["cant_menores_14"] == 0) & (df_d["cant_adultos_mayores"] == 0),
        (df_d["cant_menores_14"] == 0) & (df_d["cant_adultos_mayores"] > 0),
        (df_d["cant_menores_14"] > 0) & (df_d["cant_adultos_mayores"] > 0) & (df_d["cant_miembros"] >= 3),
    ]
    choices_comp = [
        "Con menores (<14) y sin adultos mayores",
        "Solo adultos en edad activa (15–64 años)",
        "Con adultos mayores (>=65) y sin menores",
        "Extendidos: menores y mayores (3+ miemb.)"
    ]
    df_d["perfil_demog"] = np.select(conds_comp, choices_comp, default="Excluir")
    df_d_valid = df_d[df_d["perfil_demog"] != "Excluir"]

    orden_tramos = ["1–2 miembros\n(Hogares pequeños)", "3–4 miembros\n(Hogares medianos)", "5+ miembros\n(Hogares numerosos)"]
    orden_perfiles = [
        "Con menores (<14) y sin adultos mayores",
        "Extendidos: menores y mayores (3+ miemb.)",
        "Solo adultos en edad activa (15–64 años)",
        "Con adultos mayores (>=65) y sin menores"
    ]

    rows_dem = []
    for tr in orden_tramos:
        for pf in orden_perfiles:
            sub = df_d_valid[(df_d_valid["tramo_tamano"] == tr) & (df_d_valid["perfil_demog"] == pf)]
            if len(sub) >= 100:
                rows_dem.append({
                    "Tamaño": tr,
                    "Composición": pf,
                    "Pobreza (%)": weighted_mean(sub["pobre"].values, sub["pondih"].values) * 100.0
                })
    df_dem_plot = pd.DataFrame(rows_dem)

    paleta_perfil = {
        "Con menores (<14) y sin adultos mayores": "#e74c3c",
        "Extendidos: menores y mayores (3+ miemb.)": "#e67e22",
        "Solo adultos en edad activa (15–64 años)": "#3498db",
        "Con adultos mayores (>=65) y sin menores": "#27ae60"
    }

    sns.barplot(
        data=df_dem_plot, x="Tamaño", y="Pobreza (%)", hue="Composición",
        order=orden_tramos, hue_order=orden_perfiles, palette=paleta_perfil, ax=ax0
    )
    ax0.set_xlabel("")
    ax0.set_ylabel("Tasa de Pobreza (% PONDIH)", fontweight="bold")
    ax0.set_title("A. Pobreza según Tamaño y Composición Etaria del Hogar", fontweight="bold", pad=10)
    ax0.legend(title="Composición Etaria del Hogar", loc="upper left", fontsize=8.3, frameon=True)
    ax0.set_ylim(0, 68)

    for container in ax0.containers:
        ax0.bar_label(container, fmt="%.1f%%", fontsize=7.5, padding=2)

    # Panel B: Ratio de Dependencia Económica (Miembros / Ocupados) por Macro-Período
    ax1 = axes[1]
    bins_dep = [0.99, 1.0, 1.5, 2.0, 3.0, 4.0, 999.0]
    labels_dep = ["1.0\n(Todos ocup.)", "1.01–1.5", "1.51–2.0", "2.01–3.0", "3.01–4.0", ">4.0 / Sin\nocupados"]
    df_d["tramo_dep_econ"] = pd.cut(df_d["ratio_dependencia_econom"], bins=bins_dep, labels=labels_dep)

    for bloque, color, marker in [
        ("Desarrollo (2016–2021)", "#34495e", "s"),
        ("Out-of-Sample (2022–2026)", "#c0392b", "o")
    ]:
        sub_b = df_d[df_d["bloque_temporal"] == bloque]
        vals_y = []
        for lbl in labels_dep:
            s_l = sub_b[sub_b["tramo_dep_econ"] == lbl]
            vals_y.append(weighted_mean(s_l["pobre"].values, s_l["pondih"].values) * 100.0)
        ax1.plot(labels_dep, vals_y, marker=marker, markersize=6, lw=2.3, color=color, label=bloque)

    ax1.set_xlabel("Ratio de Dependencia Económica (Miembros / Ocupados)", fontweight="bold")
    ax1.set_ylabel("Tasa de Pobreza (% PONDIH)", fontweight="bold")
    ax1.set_title("B. Gradiente por Carga Económica y Estabilidad Inter-Período", fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=9, frameon=True)
    ax1.set_ylim(0, 72)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "eda_03_demografia_y_carga_familiar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# FIGURA 4: DIMENSIONES 2 Y 3 — INTERACCIÓN LABORAL, EDUCACIÓN Y SALUD PÚBLICA
# -----------------------------------------------------------------------------
def generar_fig4_laboral_y_capital_humano(df_all: pd.DataFrame):
    print("[5/7] Generando Figura 4: Interacción Educación x Informalidad y Salud Pública...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.15, 1.0]})

    df_lc = df_all.copy()
    educ_map = {
        1: "1. Primaria Inc./S.Inst.",
        7: "1. Primaria Inc./S.Inst.",
        2: "2. Primaria Comp.",
        3: "3. Secundaria Inc.",
        4: "4. Secundaria Comp.",
        5: "5. Superior Inc.",
        6: "6. Superior Comp."
    }
    df_lc["educ_grp"] = df_lc["max_educ_hogar"].astype(int).map(educ_map)
    orden_educ = [
        "1. Primaria Inc./S.Inst.", "2. Primaria Comp.", "3. Secundaria Inc.",
        "4. Secundaria Comp.", "5. Superior Inc.", "6. Superior Comp."
    ]

    df_lc["inf_grp"] = pd.cut(
        df_lc["tasa_informalidad_hogar"],
        bins=[-0.01, 0.0, 0.50, 0.99, 1.0],
        labels=["0% (Todo Formal)", "1%–50% (Mixto)", "51%–99% (Mayoría Inf.)", "100% (Todo Informal)"]
    )

    # Panel A: Heatmap Educación x Informalidad
    pivot_ht = pd.DataFrame(
        index=orden_educ,
        columns=["0% (Todo Formal)", "1%–50% (Mixto)", "51%–99% (Mayoría Inf.)", "100% (Todo Informal)"],
        dtype=float
    )
    for ed in orden_educ:
        for inf in pivot_ht.columns:
            sub = df_lc[(df_lc["educ_grp"] == ed) & (df_lc["inf_grp"] == inf)]
            if len(sub) > 0:
                pivot_ht.loc[ed, inf] = weighted_mean(sub["pobre"].values, sub["pondih"].values) * 100.0

    sns.heatmap(
        pivot_ht, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Pobreza (% PONDIH)"},
        ax=axes[0], linewidths=0.8, linecolor="white"
    )
    axes[0].set_title("A. Interacción Capital Humano × Informalidad Laboral (% Pobreza)", fontweight="bold", pad=10)
    axes[0].set_xlabel("Tasa de Informalidad de Asalariados en el Hogar", fontweight="bold")
    axes[0].set_ylabel("Máximo Nivel Educativo de Adultos del Hogar", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=18)

    # Panel B: Cobertura de Salud Pública x Condición Ocupacional del Jefe/a
    cond_jefe = [
        (df_lc["jefe_desocupado"] == 1),
        (df_lc["jefe_ocupado"] == 1) & (df_lc["jefe_informal"] == 1),
        (df_lc["jefe_ocupado"] == 1) & (df_lc["jefe_informal"] == 0),
        (df_lc["jefe_ocupado"] == 0) & (df_lc["jefe_desocupado"] == 0),
    ]
    choices_jefe = [
        "Jefe Desocupado",
        "Jefe Ocupado Informal",
        "Jefe Ocupado Formal / Cta. Propia",
        "Jefe Inactivo (Jubilado/Otros)"
    ]
    df_lc["condicion_jefe"] = np.select(cond_jefe, choices_jefe, default="Otro")
    df_lc["salud_grp"] = pd.cut(
        df_lc["prop_salud_publica"],
        bins=[-0.01, 0.0, 0.99, 1.0],
        labels=["0% Salud Pública\n(Obra Social / Prepaga)", "Cobertura Mixta\n(1%–99% Pública)", "100% Salud Pública\nExclusiva"]
    )

    records_salud = []
    for sj in choices_jefe:
        for sg in ["0% Salud Pública\n(Obra Social / Prepaga)", "Cobertura Mixta\n(1%–99% Pública)", "100% Salud Pública\nExclusiva"]:
            sub = df_lc[(df_lc["condicion_jefe"] == sj) & (df_lc["salud_grp"] == sg)]
            records_salud.append({
                "Condición Jefe": sj,
                "Cobertura Salud": sg,
                "Pobreza (%)": weighted_mean(sub["pobre"].values, sub["pondih"].values) * 100.0
            })
    df_salud_plot = pd.DataFrame(records_salud)

    sns.barplot(
        data=df_salud_plot, x="Cobertura Salud", y="Pobreza (%)", hue="Condición Jefe",
        palette=["#c0392b", "#e67e22", "#2980b9", "#7f8c8d"], ax=axes[1]
    )
    axes[1].set_title("B. Cobertura de Salud Pública como Proxy de Precariedad", fontweight="bold", pad=10)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Tasa de Pobreza (% PONDIH)", fontweight="bold")
    axes[1].set_ylim(0, 72)
    axes[1].legend(title="Inserción del Jefe/a", loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "eda_04_laboral_y_capital_humano.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# FIGURA 5: DIMENSIONES 4 Y 5 — BRECHAS REGIONALES Y CARENCIAS HABITACIONALES
# -----------------------------------------------------------------------------
def generar_fig5_regional_y_habitat(df_all: pd.DataFrame):
    print("[6/7] Generando Figura 5: Brechas regionales y tasas comparativas de hábitat...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Panel A: Pobreza por Región (7 zonas) comparando Desarrollo vs OOS
    reg_rows = []
    for reg, sub_r in df_all.groupby("region_estudio"):
        for blq in ["Desarrollo (2016–2021)", "Out-of-Sample (2022–2026)"]:
            sub_b = sub_r[sub_r["bloque_temporal"] == blq]
            reg_rows.append({
                "Región": reg,
                "Período": blq,
                "Pobreza (%)": weighted_mean(sub_b["pobre"].values, sub_b["pondih"].values) * 100.0
            })
    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(OUTPUT_TABLES / "eda_resumen_regional_temporal.csv", index=False)

    orden_reg = (
        df_reg[df_reg["Período"] == "Out-of-Sample (2022–2026)"]
        .sort_values("Pobreza (%)", ascending=False)["Región"]
        .tolist()
    )

    sns.barplot(
        data=df_reg, x="Región", y="Pobreza (%)", hue="Período",
        order=orden_reg, palette=["#5d6d7e", "#c0392b"], ax=axes[0]
    )
    axes[0].set_title("A. Incidencia de Pobreza por Región Estadística (7 Zonas)", fontweight="bold", pad=10)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Tasa de Pobreza (% PONDIH)", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylim(0, 45)
    axes[0].legend(title="Bloque Temporal", loc="upper right", fontsize=8.5, frameon=True)

    # Panel B: Tasa de pobreza Con Carencia vs Sin Carencia
    indicadores_habitat = [
        ("Vivienda en Villa de Emergencia (iv12_3=1)", df_all["iv12_3"] == 1),
        ("Hacinamiento Crítico (>3 pers/cuarto)", df_all["hacinamiento_critico"] == 1),
        ("Pisos Precarios (Cemento/Tierra, iv3!=1)", df_all["iv3"] != 1),
        ("Vivienda en Zona Inundable (iv12_2=1)", df_all["iv12_2"] == 1),
        ("Baño sin Red Cloacal (iv10!=1)", df_all["iv10"] != 1),
        ("Vivienda sin Cielorraso (iv5=2)", df_all["iv5"] == 2),
        ("Vivienda cerca de Basural (iv12_1=1)", df_all["iv12_1"] == 1),
        ("Agua fuera de la Vivienda (iv6!=1)", df_all["iv6"] != 1),
        ("Tenencia Precaria / Ocupante (ii7>=6)", df_all["ii7"] >= 6),
    ]

    y = df_all["pobre"].values
    w = df_all["pondih"].values
    hab_rows = []
    for nombre, cond in indicadores_habitat:
        c = cond.values
        p_con = weighted_mean(y[c], w[c]) * 100.0
        p_sin = weighted_mean(y[~c], w[~c]) * 100.0
        hab_rows.append({
            "Indicador": nombre,
            "Pobreza_Con": p_con,
            "Pobreza_Sin": p_sin
        })

    df_hab = pd.DataFrame(hab_rows).sort_values("Pobreza_Con", ascending=True).reset_index(drop=True)
    ax1 = axes[1]
    y_pos = np.arange(len(df_hab))
    height = 0.36

    ax1.barh(y_pos + height / 2, df_hab["Pobreza_Con"], height=height, color="#8c564b", label="Con la carencia")
    ax1.barh(y_pos - height / 2, df_hab["Pobreza_Sin"], height=height, color="#bdc3c7", label="Sin la carencia (resto)")

    for idx, row in df_hab.iterrows():
        ax1.text(
            row["Pobreza_Con"] + 0.8, idx + height / 2,
            f"{row['Pobreza_Con']:.1f}%",
            va="center", fontsize=8.3, fontweight="bold", color="#5d3a33"
        )
        ax1.text(
            row["Pobreza_Sin"] + 0.8, idx - height / 2,
            f"{row['Pobreza_Sin']:.1f}%",
            va="center", fontsize=8.0, color="#555555"
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_hab["Indicador"], fontsize=8.8)
    ax1.set_xlabel("Tasa de Pobreza (% PONDIH)", fontweight="bold")
    ax1.set_title("B. Tasa de Pobreza según Privación Habitacional", fontweight="bold", pad=10)
    ax1.set_xlim(0, df_hab["Pobreza_Con"].max() * 1.18)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "eda_05_habitat_y_contexto_regional.png", dpi=300, bbox_inches="tight")
    plt.close(fig)





# -----------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generar_fig1_anatomia_y_serie(df, features)
    df_iv = generar_fig2_ranking_iv_y_tabla(df, features)
    generar_fig3_demografia(df)
    generar_fig4_laboral_y_capital_humano(df)
    generar_fig5_regional_y_habitat(df)
    
    print("\n" + "=" * 75)
    print("✓ EDA COMPLETADO CON ÉXITO.")
    print(f"  - Figuras guardadas en: {OUTPUT_FIGS.resolve()}")
    print(f"  - Tablas guardadas en:  {OUTPUT_TABLES.resolve()}")
    print("\nTop 10 Variables por Information Value (IV Ponderado):")
    print(df_iv[["variable", "dimension", "tipo_estadistico", "IV_ponderado", "direccion"]].head(10).to_string(index=False))
    print("=" * 75)
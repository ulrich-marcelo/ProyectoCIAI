"""
===============================================================================
PROYECTO: Modelado de Pobreza con EPH (Machine Learning)
SCRIPT: Interpretabilidad Unificada con SHAP Recalibrado y Ponderado (PONDIH)
FAMILIAS SOPORTADAS: logistic, random_forest, xgboost, mlp
===============================================================================
"""

import os
import gc
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns",
    category=UserWarning,
    module="sklearn.preprocessing._encoders"
)

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS, DISPOSITIVO Y FACTORES DE ESCALA
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PARQUET_CANDIDATES = [
    Path("data/eph_hogares_consolidada_2016_2026.parquet"),
    Path("../data/eph_hogares_consolidada_2016_2026.parquet"),
]
PARQUET_FILE = next((p for p in PARQUET_CANDIDATES if p.exists()), PARQUET_CANDIDATES[0])

MODEL_DIR = Path("modelos") if Path("modelos").exists() else Path("notebooks/modelos")
OUT_SHAP_DIR = Path("resultados/shap") if Path("modelos").exists() else Path("notebooks/resultados/shap")
OUT_SHAP_DIR.mkdir(parents=True, exist_ok=True)

# Factores de escala de clase positiva aplicados en reentrenamiento_oos.py
SCALE_POS_FACTORS = {
    "xgboost": {"cfg1": 1.2, "cfg2": 0.8, "cfg3": 1.0},
    "random_forest": {"cfg1": 1.0, "cfg2": 1.0, "cfg3": 1.0},
    "logistic": {"cfg1": 1.0, "cfg2": 1.0, "cfg3": 1.0},
    "mlp": {"cfg1": 1.0, "cfg2": 1.0, "cfg3": 1.0},
}

# Hiperparámetros MLP de respaldo (para checkpoints antiguos sin diccionario)
MLP_PARAMS_FALLBACK = {
    "cfg1": {"hidden_layers": [512, 256, 128], "dropout": 0.4},
    "cfg2": {"hidden_layers": [512, 256, 128], "dropout": 0.4},
    "cfg3": {"hidden_layers": [512, 256, 128], "dropout": 0.4},
}

# Listas exactas de variables categóricas/ordinales usadas en reentrenamiento_oos.py
COLS_NOMINALES_Y_BINARIAS_ENCUESTA = [
    "region_estudio", "iv1", "iv3", "iv4", "iv5", "iv6", "iv8", "iv9",
    "iv10", "iv11", "iv12_1", "iv12_2", "iv12_3", "ii3", "ii3_1",
    "ii4_1", "ii4_2", "ii4_3", "ii5", "ii5_1", "ii6", "ii6_1", "ii7", "ii8"
]
COLS_ORDINALES = ["max_educ_hogar", "jefe_nivel_ed", "iv7", "ii9"]

# -----------------------------------------------------------------------------
# MAPEO DE VARIABLES ORIGINALES A DIMENSIONES CONCEPTUALES (FASE 6)
# -----------------------------------------------------------------------------
DIMENSIONES_CONCEPTUALES = {
    "Hábitat y Calidad Habitacional": {
        "iv1", "iv2", "iv3", "iv4", "iv5", "iv6", "iv7", "iv8", "iv9",
        "iv10", "iv11", "iv12_1", "iv12_2", "iv12_3",
        "ii1", "ii2", "ii3", "ii3_1", "ii4_1", "ii4_2", "ii4_3",
        "ii5", "ii5_1", "ii6", "ii6_1", "ii7", "ii8", "ii9",
        "personas_por_cuarto", "hacinamiento_critico"
    },
    "Capital Humano y Salud": {
        "max_educ_hogar", "jefe_nivel_ed", "prop_salud_publica"
    },
    "Inserción Laboral": {
        "cant_ocupados", "cant_desocupados", "cant_activos",
        "cant_asalariados", "cant_informales",
        "tasa_informalidad_hogar", "tasa_desocupacion_hogar",
        "tasa_ocupacion_hogar", "horas_trabajadas_tot", "presencia_subocupados",
        "jefe_ocupado", "jefe_desocupado", "jefe_informal"
    },
    "Estructura Demográfica": {
        "cant_miembros", "cant_menores_14", "cant_adultos_mayores",
        "cant_edad_activa", "adequi_hogar",
        "ratio_dependencia_demog", "ratio_dependencia_econom",
        "jefe_mujer", "jefe_edad"
    },
    "Contexto Regional": {
        "region_estudio"
    }
}

NOMBRES_LEGIBLES = {
    "region_estudio": "Región Estadística",
    "max_educ_hogar": "Máximo Nivel Educativo del Hogar",
    "jefe_nivel_ed": "Nivel Educativo del Jefe/a",
    "prop_salud_publica": "Prop. Cobertura Salud Pública Exclusiva",
    "tasa_informalidad_hogar": "Tasa de Informalidad Laboral del Hogar",
    "tasa_desocupacion_hogar": "Tasa de Desocupación del Hogar",
    "tasa_ocupacion_hogar": "Tasa de Ocupación del Hogar",
    "horas_trabajadas_tot": "Horas Trabajadas Totales del Hogar",
    "presencia_subocupados": "Presencia de Subocupados",
    "jefe_ocupado": "Jefe/a Ocupado/a",
    "jefe_desocupado": "Jefe/a Desocupado/a",
    "jefe_informal": "Jefe/a Asalariado/a Informal",
    "cant_ocupados": "Cantidad de Ocupados",
    "cant_desocupados": "Cantidad de Desocupados",
    "cant_activos": "Cantidad de Económicamente Activos",
    "cant_asalariados": "Cantidad de Asalariados",
    "cant_informales": "Cantidad de Trabajadores Informales",
    "cant_miembros": "Tamaño del Hogar (Miembros)",
    "cant_menores_14": "Cantidad de Menores de 14 años",
    "cant_adultos_mayores": "Cantidad de Adultos Mayores",
    "cant_edad_activa": "Miembros en Edad Activa",
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
    "iv11": "Desagüe del Baño (Cloaca/Pozo)",
    "iv12_1": "Vivienda cerca de Basural",
    "iv12_2": "Vivienda en Zona Inundable",
    "iv12_3": "Vivienda en Villa de Emergencia",
    "ii1": "Cantidad de Cuartos de Uso Exclusivo",
    "ii2": "Cuartos para Dormir",
    "ii3": "Usa Cuartos para Trabajar",
    "ii4_1": "Tiene Cuarto de Cocina",
    "ii4_2": "Tiene Lavadero",
    "ii4_3": "Tiene Garaje",
    "ii5": "Combustible para Cocinar",
    "ii7": "Régimen de Tenencia de la Vivienda",
    "ii8": "Combustible Utilizado (General)",
    "ii9": "Uso del Baño (Exclusivo/Compartido)"
}


# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE RECALIBRACIÓN BAYESIANA Y HOMOGENEIZACIÓN
# -----------------------------------------------------------------------------
def homogeneizar_categorica(serie: pd.Series) -> pd.Series:
    """
    Idéntica a reentrenamiento_oos.py: rellena nulos y asegura tipo string
    unificado antes de pasar por OneHotEncoder.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return serie.fillna(-1).astype(int).astype(str)
    return serie.fillna("missing").astype(str)


def recalibrar_probabilidad_bayes(probs_pond: np.ndarray, peso_efectivo: float) -> np.ndarray:
    """
    Invierte la reponderación de clase positiva (class_weight='balanced' / scale_pos_weight):
        p_cal = p_pond / (p_pond + w * (1 - p_pond))
    """
    p = np.clip(probs_pond.astype(np.float64), 1e-7, 1.0 - 1e-7)
    p_cal = p / (p + peso_efectivo * (1.0 - p))
    return p_cal.astype(np.float32)


def escalar_shap_a_prob_calibrada(
    sv_pond: np.ndarray,
    probs_pond_eval: np.ndarray,
    probs_pond_bg: np.ndarray,
    peso_efectivo: float
) -> np.ndarray:
    """
    Transforma los valores SHAP de TreeExplainer desde el espacio de probabilidad
    reponderada (p_pond) al espacio de probabilidad real recalibrada (p_cal),
    preservando aditividad exacta por observación:
        sum_j(phi_cal[i, j]) == p_cal(x_i) - E[p_cal(X_bg)]
    """
    p_cal_eval = recalibrar_probabilidad_bayes(probs_pond_eval, peso_efectivo).astype(np.float64)
    p_cal_bg = recalibrar_probabilidad_bayes(probs_pond_bg, peso_efectivo).astype(np.float64)

    base_pond = float(np.mean(probs_pond_bg))
    base_cal = float(np.mean(p_cal_bg))

    diff_pond = probs_pond_eval.astype(np.float64) - base_pond
    diff_cal = p_cal_eval - base_cal

    p_clip = np.clip(probs_pond_eval.astype(np.float64), 1e-6, 1.0 - 1e-6)
    deriv = peso_efectivo / ((p_clip + peso_efectivo * (1.0 - p_clip)) ** 2)

    alpha = np.where(np.abs(diff_pond) > 1e-6, diff_cal / diff_pond, deriv)
    return (sv_pond.astype(np.float64) * alpha[:, np.newaxis]).astype(np.float32)


# -----------------------------------------------------------------------------
# ARQUITECTURA TABULAR MLP (IDÉNTICA A REENTRENAMIENTO_OOS.PY)
# -----------------------------------------------------------------------------
class TabularMLP(nn.Module):
    def __init__(self, in_features: int, hidden_layers: List[int], dropout: float = 0.4):
        super().__init__()
        layers = []
        last_dim = in_features
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def cargar_mlp(pt_path: Path, in_features: int, cfg_id: str) -> TabularMLP:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    checkpoint = torch.load(pt_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        params = checkpoint.get("params", MLP_PARAMS_FALLBACK[cfg_id])
        dim = checkpoint.get("in_features", in_features)
        state_dict = checkpoint["state_dict"]
    else:
        params = MLP_PARAMS_FALLBACK[cfg_id]
        dim = in_features
        state_dict = checkpoint

    model = TabularMLP(
        in_features=dim,
        hidden_layers=params["hidden_layers"],
        dropout=params.get("dropout", 0.4)
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# RECONSTRUCCIÓN DE NOMBRES DE COLUMNAS POST-COLUMNTRANSFORMER
# -----------------------------------------------------------------------------
def mapear_columnas_transformadas(preprocessor, raw_cols: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Agrupa los índices de salida del ColumnTransformer correspondientes a cada
    variable original de la EPH (colapsando dummies One-Hot por aditividad SHAP).
    """
    transformed_names = list(preprocessor.get_feature_names_out())
    raw_cols_sorted = sorted(raw_cols, key=len, reverse=True)
    mapa_var_a_indices: Dict[str, List[int]] = {c: [] for c in raw_cols}

    for idx, tname in enumerate(transformed_names):
        sin_prefijo = tname.split("__", 1)[1] if "__" in tname else tname

        encontrada = None
        for rcol in raw_cols_sorted:
            if sin_prefijo == rcol or sin_prefijo.startswith(rcol + "_"):
                encontrada = rcol
                break

        if encontrada is None:
            raise ValueError(f"No se pudo mapear la columna transformada '{tname}' a ninguna feature original.")
        mapa_var_a_indices[encontrada].append(idx)

    mapa_var_a_indices = {k: v for k, v in mapa_var_a_indices.items() if len(v) > 0}
    return transformed_names, mapa_var_a_indices


# -----------------------------------------------------------------------------
# CARGA, PREPROCESAMIENTO ESTRICTO Y MUESTREO ESTRATIFICADO PARA SHAP
# -----------------------------------------------------------------------------
def preparar_muestras_shap(sample_per_sem: int = 500, bg_size: int = 250, seed: int = 42):
    print(f"\n[1/4] Cargando preprocesador y dataset desde: {PARQUET_FILE}")
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo Parquet en: {PARQUET_FILE.resolve()}")

    preprocessor = joblib.load(MODEL_DIR / "preprocessor_fitted.joblib")
    raw_cols = joblib.load(MODEL_DIR / "feature_names_raw.joblib")

    df = pd.read_parquet(PARQUET_FILE)
    df["id_hogar"] = df["codusu"].astype(str) + "_" + df["nro_hogar"].astype(str)

    df_train_raw = df[df["ano4"] <= 2021].copy()
    df_test = df[df["ano4"] > 2021].copy().reset_index(drop=True)

    # Purga inter-panel idéntica a reentrenamiento_oos.py
    hogares_test = set(df_test["id_hogar"].unique())
    df_train = df_train_raw[~df_train_raw["id_hogar"].isin(hogares_test)].copy().reset_index(drop=True)

    # Factor exacto de desbalanceo del set de entrenamiento purgado (para recalibración Bayesiana)
    y_tr = df_train["pobre"].values
    scale_pos_base = float((len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1))

    # Etiqueta semestral en OOS
    df_test["semestre"] = np.where(df_test["trimestre"].isin([1, 2]), 1, 2)
    df_test["periodo"] = df_test["ano4"].astype(str) + "-S" + df_test["semestre"].astype(str)

    # Muestra background de Train (referencia para esperanza condicional E[f(X)])
    df_bg = df_train.sample(n=min(bg_size, len(df_train)), random_state=seed).copy().reset_index(drop=True)

    # Muestra estratificada por semestre en OOS (para análisis global y longitudinal balanceado)
    if sample_per_sem > 0:
        muestras = []
        for _, grp in df_test.groupby("periodo"):
            n_take = min(sample_per_sem, len(grp))
            muestras.append(grp.sample(n=n_take, random_state=seed))
        df_eval = pd.concat(muestras, axis=0).reset_index(drop=True)
    else:
        df_eval = df_test.copy().reset_index(drop=True)

    # Conservar copia numérica cruda antes de homogeneizar categorías (para coloración Beeswarm)
    df_eval_display = df_eval[raw_cols].copy()

    # Homogeneización de categóricas y ordinales idéntica a reentrenamiento_oos.py
    for c in [col for col in COLS_NOMINALES_Y_BINARIAS_ENCUESTA if col in raw_cols]:
        df_bg[c] = homogeneizar_categorica(df_bg[c])
        df_eval[c] = homogeneizar_categorica(df_eval[c])

    for c in [col for col in COLS_ORDINALES if col in raw_cols]:
        df_bg[c] = df_bg[c].fillna(-1)
        df_eval[c] = df_eval[c].fillna(-1)

    X_bg = preprocessor.transform(df_bg[raw_cols]).astype(np.float32)
    X_eval = preprocessor.transform(df_eval[raw_cols]).astype(np.float32)

    if np.isnan(X_bg).any() or np.isnan(X_eval).any():
        raise ValueError("Se detectaron valores NaN tras el preprocesamiento en SHAP. Revisar limpieza.py.")

    print(f"      ✓ Factor desbalanceo Train (scale_pos_base): {scale_pos_base:.4f}")
    print(f"      ✓ Muestra Background (Train): {X_bg.shape[0]:,} hogares | {X_bg.shape[1]} features transformadas")
    print(f"      ✓ Muestra Evaluación (OOS):   {X_eval.shape[0]:,} hogares en {df_eval['periodo'].nunique()} semestres")

    del df, df_train_raw, df_train
    gc.collect()

    return preprocessor, raw_cols, X_bg, X_eval, df_eval, df_eval_display, scale_pos_base


# -----------------------------------------------------------------------------
# CÓMPUTO DE SHAP EN ESPACIO DE PROBABILIDAD RECALIBRADA (p_cal)
# -----------------------------------------------------------------------------
def calcular_shap_modelo(
    familia: str,
    cfg_id: str,
    X_bg: np.ndarray,
    X_eval: np.ndarray,
    scale_pos_base: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      - sv_cal: matriz SHAP (N_eval, N_features_transformadas) en unidades homogéneas
                de probabilidad recalibrada [0, 1].
      - probs_cal_eval: vector de probabilidades recalibradas p_cal(x_i) para segmentar
                        estratos de riesgo (Núcleo Duro vs. Franja Vulnerable).
    """
    factor = SCALE_POS_FACTORS.get(familia, {}).get(cfg_id, 1.0)
    w_eff = scale_pos_base * factor

    if familia == "logistic":
        model = joblib.load(MODEL_DIR / f"logistic_{cfg_id}.joblib")

        def predict_fn_log(x_np: np.ndarray) -> np.ndarray:
            p_pond = model.predict_proba(x_np)[:, 1]
            return recalibrar_probabilidad_bayes(p_pond, w_eff)

        probs_cal_eval = predict_fn_log(X_eval)
        bg_summary = shap.kmeans(X_bg, min(50, len(X_bg)))
        explainer = shap.KernelExplainer(predict_fn_log, bg_summary)
        sv = explainer.shap_values(X_eval, nsamples=200, silent=True)
        return np.asarray(sv, dtype=np.float32), probs_cal_eval

    elif familia == "random_forest":
        model = joblib.load(MODEL_DIR / f"random_forest_{cfg_id}.joblib")
        probs_pond_eval = model.predict_proba(X_eval)[:, 1]
        probs_pond_bg = model.predict_proba(X_bg)[:, 1]
        probs_cal_eval = recalibrar_probabilidad_bayes(probs_pond_eval, w_eff)

        # Submuestra representativa de background (25 centroides) para mantener barra de progreso y velocidad
        bg_rf = shap.kmeans(X_bg, min(25, len(X_bg))).data
        masker = shap.maskers.Independent(bg_rf, max_samples=len(bg_rf))
        explainer = shap.TreeExplainer(model, data=masker, feature_perturbation="interventional")
        sv_pond = explainer.shap_values(X_eval, check_additivity=False)
        if isinstance(sv_pond, list):
            sv_pond = sv_pond[1]
        elif sv_pond.ndim == 3:
            sv_pond = sv_pond[:, :, 1]

        sv_cal = escalar_shap_a_prob_calibrada(sv_pond, probs_pond_eval, probs_pond_bg, w_eff)
        return sv_cal, probs_cal_eval

    elif familia == "xgboost":
        model = joblib.load(MODEL_DIR / f"xgboost_{cfg_id}.joblib")
        probs_pond_eval = model.predict_proba(X_eval)[:, 1]
        probs_pond_bg = model.predict_proba(X_bg)[:, 1]
        probs_cal_eval = recalibrar_probabilidad_bayes(probs_pond_eval, w_eff)

        explainer = shap.TreeExplainer(
            model,
            data=X_bg,
            feature_perturbation="interventional",
            model_output="probability"
        )
        sv_pond = explainer.shap_values(X_eval, check_additivity=False)
        if isinstance(sv_pond, list):
            sv_pond = sv_pond[1]
        elif sv_pond.ndim == 3:
            sv_pond = sv_pond[:, :, 1]

        sv_cal = escalar_shap_a_prob_calibrada(sv_pond, probs_pond_eval, probs_pond_bg, w_eff)
        return sv_cal, probs_cal_eval

    elif familia == "mlp":
        base_model = cargar_mlp(MODEL_DIR / f"mlp_{cfg_id}.pt", X_bg.shape[1], cfg_id)

        # Wrapper diferenciable que emite directamente la probabilidad real recalibrada p_cal:
        # En espacio logit, dividir odds por w_eff equivale exactamente a restar log(w_eff)
        log_w = float(np.log(w_eff))

        class CalibratedMLPWrapper(nn.Module):
            def __init__(self, net: nn.Module, shift: float):
                super().__init__()
                self.net = net
                self.shift = shift

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                logits = self.net(x) - self.shift
                return torch.sigmoid(logits)

        wrapped_model = CalibratedMLPWrapper(base_model, log_w).to(DEVICE)
        wrapped_model.eval()

        # Background de 50 muestras representativas en tensor
        bg_mlp_np = shap.kmeans(X_bg, min(50, len(X_bg))).data.astype(np.float32)
        bg_tensor = torch.tensor(bg_mlp_np, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=DEVICE)
            probs_cal_eval = wrapped_model(X_eval_t).squeeze(1).cpu().numpy()

        # GradientExplainer aprovecha backpropagation vectorizada en CPU/GPU (10x-30x más rápido que KernelExplainer)
        explainer = shap.GradientExplainer(wrapped_model, bg_tensor)

        # Procesar en mini-batches de 256 filas para mantener bajo consumo de RAM en CPU
        batch_size = 256
        sv_list = []
        for start in range(0, len(X_eval), batch_size):
            batch_t = torch.tensor(X_eval[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            sv_batch = explainer.shap_values(batch_t, nsamples=100)
            if isinstance(sv_batch, list):
                sv_batch = sv_batch[0]
            sv_batch = np.asarray(sv_batch, dtype=np.float32)
            if sv_batch.ndim == 3:
                sv_batch = sv_batch.squeeze(-1)
            sv_list.append(sv_batch)

        sv_cal = np.vstack(sv_list)
        return sv_cal, probs_cal_eval

    else:
        raise ValueError(f"Familia no soportada: {familia}")


# -----------------------------------------------------------------------------
# AGREGACIÓN PONDERADA POR VARIABLE, DIMENSIÓN, ESTRATO Y EVOLUCIÓN TEMPORAL
# -----------------------------------------------------------------------------
def procesar_y_guardar_shap(
    familia: str,
    sv_transformed: np.ndarray,
    probs_cal_eval: np.ndarray,
    X_eval: np.ndarray,
    df_eval: pd.DataFrame,
    df_eval_display: pd.DataFrame,
    mapa_var_a_indices: Dict[str, List[int]]
):
    print(f"\n[3/4] Agregando valores SHAP recalibrados y ponderados por PONDIH ({familia.upper()})...")
    vars_ordenadas = list(mapa_var_a_indices.keys())
    n_obs = sv_transformed.shape[0]
    n_vars = len(vars_ordenadas)

    w_eval = df_eval["pondih"].values.astype(np.float64) if "pondih" in df_eval.columns else np.ones(n_obs)

    # 1. Matriz SHAP colapsada a variables originales de la EPH (aditividad exacta)
    sv_vars = np.zeros((n_obs, n_vars), dtype=np.float32)
    X_vars_display = np.zeros((n_obs, n_vars), dtype=np.float32)

    for j, var_name in enumerate(vars_ordenadas):
        idxs = mapa_var_a_indices[var_name]
        sv_vars[:, j] = sv_transformed[:, idxs].sum(axis=1)
        if var_name in df_eval_display.columns and pd.api.types.is_numeric_dtype(df_eval_display[var_name]):
            X_vars_display[:, j] = df_eval_display[var_name].fillna(-1).values.astype(np.float32)
        else:
            X_vars_display[:, j] = X_eval[:, idxs[0]]

    # Importancia global ponderada por PONDIH: Mean_w(|SHAP|)
    mean_abs_vars = np.average(np.abs(sv_vars), axis=0, weights=w_eval)
    mean_signed_vars = np.average(sv_vars, axis=0, weights=w_eval)

    df_imp_vars = pd.DataFrame({
        "familia": familia,
        "variable": vars_ordenadas,
        "etiqueta": [NOMBRES_LEGIBLES.get(v, v) for v in vars_ordenadas],
        "mean_abs_shap_cal": mean_abs_vars,
        "mean_signed_shap_cal": mean_signed_vars,
        "peso_relativo_pct": (mean_abs_vars / mean_abs_vars.sum()) * 100.0
    }).sort_values("mean_abs_shap_cal", ascending=False).reset_index(drop=True)

    df_imp_vars.to_csv(OUT_SHAP_DIR / f"importancia_variables_{familia}.csv", index=False)

    # 2. Agregación por Dimensión Conceptual (evita dilución por colinealidad)
    dim_neta_hogar = {}
    dim_bruta_hogar = {}

    vars_asignadas = set()
    for dim_name, var_set in DIMENSIONES_CONCEPTUALES.items():
        idxs_dim = [j for j, v in enumerate(vars_ordenadas) if v in var_set]
        for j in idxs_dim:
            vars_asignadas.add(vars_ordenadas[j])
        if idxs_dim:
            # Empuje neto de la dimensión en cada hogar i: sum_{j in G_k} phi_{i,j}
            dim_neta_hogar[dim_name] = sv_vars[:, idxs_dim].sum(axis=1)
            # Masa bruta en cada hogar i: sum_{j in G_k} |phi_{i,j}|
            dim_bruta_hogar[dim_name] = np.abs(sv_vars[:, idxs_dim]).sum(axis=1)

    no_asignadas = [j for j, v in enumerate(vars_ordenadas) if v not in vars_asignadas]
    if no_asignadas:
        dim_neta_hogar["Otras Variables"] = sv_vars[:, no_asignadas].sum(axis=1)
        dim_bruta_hogar["Otras Variables"] = np.abs(sv_vars[:, no_asignadas]).sum(axis=1)

    df_dim_neta = pd.DataFrame(dim_neta_hogar)
    df_dim_bruta = pd.DataFrame(dim_bruta_hogar)

    imp_dim_neta = pd.Series(
        np.average(df_dim_neta.abs().values, axis=0, weights=w_eval),
        index=df_dim_neta.columns
    )
    imp_dim_bruta = pd.Series(
        np.average(df_dim_bruta.values, axis=0, weights=w_eval),
        index=df_dim_bruta.columns
    )

    df_imp_dim = pd.DataFrame({
        "familia": familia,
        "dimension": imp_dim_neta.index,
        "importancia_neta_media": imp_dim_neta.values,
        "peso_neto_pct": (imp_dim_neta.values / imp_dim_neta.values.sum()) * 100.0,
        "importancia_bruta_media": imp_dim_bruta.values,
        "peso_bruto_pct": (imp_dim_bruta.values / imp_dim_bruta.values.sum()) * 100.0
    }).sort_values("importancia_neta_media", ascending=False).reset_index(drop=True)

    df_imp_dim.to_csv(OUT_SHAP_DIR / f"importancia_dimensiones_{familia}.csv", index=False)

    # 3. Descomposición por Estrato de Riesgo (Núcleo Duro vs. Franja Vulnerable vs. Integrados)
    estratos = np.where(
        probs_cal_eval >= 0.50,
        "1_Nucleo_Duro_Structural (p_cal>=0.50)",
        np.where(
            probs_cal_eval >= 0.12,
            "2_Franja_Vulnerable (0.12<=p_cal<0.50)",
            "3_Hogares_Integrados (p_cal<0.12)"
        )
    )
    filas_estrato = []
    for est in sorted(np.unique(estratos)):
        mask = estratos == est
        w_est = w_eval[mask]
        vals_abs_est = np.average(df_dim_neta.abs().values[mask], axis=0, weights=w_est)
        vals_signed_est = np.average(df_dim_neta.values[mask], axis=0, weights=w_est)
        pct_est = (vals_abs_est / vals_abs_est.sum()) * 100.0
        for d_idx, dim_name in enumerate(df_dim_neta.columns):
            filas_estrato.append({
                "familia": familia,
                "estrato": est,
                "n_hogares_muestra": int(mask.sum()),
                "dimension": dim_name,
                "empuje_neto_prob_cal": round(float(vals_signed_est[d_idx]), 4),
                "magnitud_abs_prob_cal": round(float(vals_abs_est[d_idx]), 4),
                "peso_relativo_pct": round(float(pct_est[d_idx]), 2)
            })
    df_estratos = pd.DataFrame(filas_estrato)
    df_estratos.to_csv(OUT_SHAP_DIR / f"shap_estratos_riesgo_{familia}.csv", index=False)

    # 4. SHAP Longitudinal Ponderado por Semestre (2022+)
    filas_long = []
    for per, idx_grp in df_eval.groupby("periodo").groups.items():
        idx_arr = np.asarray(list(idx_grp))
        w_per = w_eval[idx_arr]
        vals_per = np.average(df_dim_neta.abs().values[idx_arr], axis=0, weights=w_per)
        pct_per = (vals_per / vals_per.sum()) * 100.0
        for d_idx, dim_name in enumerate(df_dim_neta.columns):
            filas_long.append({
                "familia": familia,
                "periodo": per,
                "dimension": dim_name,
                "magnitud_abs_prob_cal": float(vals_per[d_idx]),
                "peso_relativo_pct": float(pct_per[d_idx])
            })
    df_long_melt = pd.DataFrame(filas_long)
    df_long_melt.to_csv(OUT_SHAP_DIR / f"shap_longitudinal_{familia}.csv", index=False)

    # 5. Generar gráfico Beeswarm (Top 15 variables originales en espacio p_cal)
    print(f"[4/4] Generando gráficos SHAP para {familia.upper()}...")
    etiquetas_vars = [NOMBRES_LEGIBLES.get(v, v) for v in vars_ordenadas]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        sv_vars,
        features=X_vars_display,
        feature_names=etiquetas_vars,
        max_display=15,
        show=False
    )
    plt.title(f"Impacto SHAP en Probabilidad Real de Pobreza — {familia.upper()} (OOS 2022+)", fontsize=12, pad=12)
    plt.xlabel("Valor SHAP (Impacto en Probabilidad Recalibrada p_cal)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_SHAP_DIR / f"beeswarm_{familia}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n--- TOP 10 VARIABLES MÁS DETERMINANTES (PONDERADO POR PONDIH) ---")
    print(df_imp_vars[["variable", "etiqueta", "mean_abs_shap_cal", "peso_relativo_pct"]].head(10).to_string(index=False))

    print("\n--- IMPORTANCIA POR DIMENSIÓN CONCEPTUAL ---")
    print(df_imp_dim[["dimension", "importancia_neta_media", "peso_neto_pct"]].to_string(index=False))

    print("\n--- EMPUJE NETO POR ESTRATO DE RIESGO (p_cal) ---")
    piv_est = df_estratos.pivot(index="dimension", columns="estrato", values="empuje_neto_prob_cal")
    print(piv_est.to_string())


# -----------------------------------------------------------------------------
# CONSOLIDACIÓN COMPARATIVA ENTRE LAS 4 FAMILIAS
# -----------------------------------------------------------------------------
def consolidar_comparacion_familias():
    familias = ["logistic", "random_forest", "xgboost", "mlp"]
    disponibles = [f for f in familias if (OUT_SHAP_DIR / f"importancia_dimensiones_{f}.csv").exists()]

    if not disponibles:
        print("No se encontraron resultados SHAP previos para consolidar.")
        return

    print(f"\nConsolidando comparación entre familias disponibles: {disponibles}")

    # 1. Comparación de dimensiones conceptuales entre modelos
    dfs_dim = [pd.read_csv(OUT_SHAP_DIR / f"importancia_dimensiones_{f}.csv") for f in disponibles]
    df_all_dim = pd.concat(dfs_dim, axis=0)
    pivot_dim = df_all_dim.pivot(index="dimension", columns="familia", values="peso_neto_pct").round(2)
    pivot_dim.to_csv(OUT_SHAP_DIR / "comparacion_dimensiones_familias.csv")

    print("\n--- PESO RELATIVO DE DIMENSIONES POR FAMILIA (%) ---")
    print(pivot_dim.to_string())

    pivot_dim.plot(kind="bar", figsize=(11, 6), width=0.75)
    plt.title("Importancia Relativa por Dimensión Conceptual según Familia de Modelos (%)", fontsize=12)
    plt.ylabel("Participación en el Impacto SHAP Neto (%)")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Familia")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_SHAP_DIR / "comparacion_dimensiones_familias.png", dpi=300)
    plt.close()

    # 2. Comparación de ranking Top 10 variables lado a lado
    ranking_vars = {}
    for f in disponibles:
        df_v = pd.read_csv(OUT_SHAP_DIR / f"importancia_variables_{f}.csv").head(10)
        ranking_vars[f.upper()] = [
            f"{row['etiqueta']} ({row['peso_relativo_pct']:.1f}%)" for _, row in df_v.iterrows()
        ]
    df_rank = pd.DataFrame(ranking_vars)
    df_rank.index = [f"#{i+1}" for i in range(len(df_rank))]
    df_rank.to_csv(OUT_SHAP_DIR / "comparacion_top10_variables_familias.csv")
    print("\n--- COMPARACIÓN TOP 10 VARIABLES ENTRE FAMILIAS ---")
    print(df_rank.to_string())

    # 3. Consolidación por Estratos de Riesgo (Núcleo Duro vs. Franja Vulnerable)
    dfs_est = [
        pd.read_csv(OUT_SHAP_DIR / f"shap_estratos_riesgo_{f}.csv")
        for f in disponibles
        if (OUT_SHAP_DIR / f"shap_estratos_riesgo_{f}.csv").exists()
    ]
    if dfs_est:
        df_all_est = pd.concat(dfs_est, axis=0)
        piv_est_mean = (
            df_all_est.groupby(["estrato", "dimension"])["empuje_neto_prob_cal"]
            .mean()
            .unstack(level=0)
            .round(4)
        )
        piv_est_mean.to_csv(OUT_SHAP_DIR / "comparacion_estratos_riesgo_media.csv")
        print("\n--- EMPUJE NETO PROMEDIO POR ESTRATO DE RIESGO (p_cal) ---")
        print(piv_est_mean.to_string())

    # 4. Evolución longitudinal del promedio de familias disponibles
    dfs_long = [pd.read_csv(OUT_SHAP_DIR / f"shap_longitudinal_{f}.csv") for f in disponibles]
    df_all_long = pd.concat(dfs_long, axis=0)
    df_long_mean = df_all_long.groupby(["periodo", "dimension"])["peso_relativo_pct"].mean().unstack()
    df_long_mean.to_csv(OUT_SHAP_DIR / "evolucion_longitudinal_dimensiones.csv")

    df_long_mean.plot(kind="line", marker="o", linewidth=2, figsize=(11, 5.5))
    plt.title("Evolución Longitudinal del Peso de cada Dimensión en la Pobreza (2022–2026)", fontsize=12)
    plt.ylabel("Peso Relativo SHAP (%)")
    plt.xlabel("Semestre")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Dimensión Conceptual", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_SHAP_DIR / "evolucion_longitudinal_dimensiones.png", dpi=300)
    plt.close()

    print(f"\n✓ Archivos y gráficos consolidados guardados en: {OUT_SHAP_DIR.resolve()}")


# -----------------------------------------------------------------------------
# CLI PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis SHAP Recalibrado, Ponderado y Longitudinal")
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest", "xgboost", "mlp", "all"],
        help="Familia de modelos a explicar."
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Promedia los valores SHAP recalibrados de las 3 configs (cfg1, cfg2, cfg3) en lugar de usar solo cfg1."
    )
    parser.add_argument(
        "--sample-per-sem",
        type=int,
        default=500,
        help="Hogares muestreados por semestre OOS (default: 500 -> ~4.500 hogares totales). Usar 0 para todo OOS."
    )
    parser.add_argument(
        "--consolidar",
        action="store_true",
        help="Genera las tablas y gráficos comparativos cruzando las familias ya calculadas."
    )
    args = parser.parse_args()

    if not args.model and not args.consolidar:
        parser.error("Debes especificar --model <familia> y/o --consolidar.")

    if args.model:
        preprocessor, raw_cols, X_bg, X_eval, df_eval, df_eval_display, scale_pos_base = preparar_muestras_shap(
            sample_per_sem=args.sample_per_sem
        )
        _, mapa_var_a_indices = mapear_columnas_transformadas(preprocessor, raw_cols)

        familias_run = ["logistic", "random_forest", "xgboost", "mlp"] if args.model == "all" else [args.model]
        cfgs_run = ["cfg1", "cfg2", "cfg3"] if args.all_configs else ["cfg1"]

        for fam in familias_run:
            print(f"\n[2/4] Calculando SHAP en espacio recalibrado (p_cal) para: {fam.upper()} ({cfgs_run})...")
            sv_acum = None
            probs_acum = None
            for cid in cfgs_run:
                print(f"      -> Procesando {fam}_{cid}...")
                sv_c, probs_c = calcular_shap_modelo(fam, cid, X_bg, X_eval, scale_pos_base)
                sv_acum = sv_c if sv_acum is None else (sv_acum + sv_c)
                probs_acum = probs_c if probs_acum is None else (probs_acum + probs_c)

            sv_final = sv_acum / len(cfgs_run)
            probs_final = probs_acum / len(cfgs_run)
            procesar_y_guardar_shap(
                fam, sv_final, probs_final, X_eval, df_eval, df_eval_display, mapa_var_a_indices
            )

    if args.consolidar:
        consolidar_comparacion_familias()
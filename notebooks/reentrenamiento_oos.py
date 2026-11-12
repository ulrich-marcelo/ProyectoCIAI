"""
===============================================================================
PROYECTO: Modelado de Pobreza con EPH (Machine Learning)
SCRIPT: Reentrenamiento Final (2016-2021 Purgado) e Inferencia OOS (2022+)
FAMILIAS DISPONIBLES: logistic, random_forest, xgboost, mlp
===============================================================================
"""

import os
import sys
import gc
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import parallel_backend
from joblib.externals.loky import get_reusable_executor
import xgboost as xgb

# -----------------------------------------------------------------------------
# SUPRESIÓN DE ADVERTENCIAS ESPERADAS
# -----------------------------------------------------------------------------
# Silenciar advertencia de OneHotEncoder cuando rellena con ceros categorías desconocidas
warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns",
    category=UserWarning,
    module="sklearn.preprocessing._encoders"
)

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DEL ENTORNO Y DISPOSITIVO
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PARQUET_FILE = Path("data/eph_hogares_consolidada_2016_2026.parquet")
if not PARQUET_FILE.exists():
    PARQUET_FILE = Path("../data/eph_hogares_consolidada_2016_2026.parquet")

OUTPUT_PRED_DIR = Path("predicciones")
OUTPUT_MODEL_DIR = Path("modelos")
OUTPUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DICCIONARIO DE HIPERPARÁMETROS SELECCIONADOS Y UMBRALES ÓPTIMOS (TAU)
# -----------------------------------------------------------------------------
CONFIGS_TOP3 = {
    "xgboost": [
        {
            "id": "cfg1",
            "name": "rosy-sweep-17",
            "tau": 0.285164,
            "params": {
                "max_depth": 4,
                "learning_rate": 0.078048,
                "subsample": 0.641177,
                "colsample_bytree": 0.723352,
                "scale_pos_factor": 1.2,
                "n_estimators": 400
            }
        },
        {
            "id": "cfg2",
            "name": "desert-sweep-20",
            "tau": 0.226621,
            "params": {
                "max_depth": 3,
                "learning_rate": 0.147129,
                "subsample": 0.956033,
                "colsample_bytree": 0.981564,
                "scale_pos_factor": 0.8,
                "n_estimators": 400
            }
        },
        {
            "id": "cfg3",
            "name": "smooth-sweep-1",
            "tau": 0.255106,
            "params": {
                "max_depth": 4,
                "learning_rate": 0.102233,
                "subsample": 0.754142,
                "colsample_bytree": 0.729457,
                "scale_pos_factor": 1.0,
                "n_estimators": 400
            }
        }
    ],
    "random_forest": [
        {
            "id": "cfg1",
            "name": "earthy-sweep-18",
            "tau": 0.286567,
            "params": {
                "n_estimators": 200,
                "max_depth": 19,
                "min_samples_leaf": 10,
                "max_features": 0.25,
                "class_weight": "balanced",
                "random_state": 42
            }
        },
        {
            "id": "cfg2",
            "name": "solar-sweep-20",
            "tau": 0.278046,
            "params": {
                "n_estimators": 200,
                "max_depth": 21,
                "min_samples_leaf": 10,
                "max_features": 0.25,
                "class_weight": "balanced",
                "random_state": 42
            }
        },
        {
            "id": "cfg3",
            "name": "peach-sweep-11",
            "tau": 0.287114,
            "params": {
                "n_estimators": 200,
                "max_depth": 19,
                "min_samples_leaf": 12,
                "max_features": 0.25,
                "class_weight": "balanced",
                "random_state": 42
            }
        }
    ],
    "logistic": [
        {
            "id": "cfg1",
            "name": "giddy-sweep-26",
            "tau": 0.264293,
            "params": {
                "C": 30.723073,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 500,
                "random_state": 42
            }
        },
        {
            "id": "cfg2",
            "name": "polished-sweep-4",
            "tau": 0.264293,
            "params": {
                "C": 11.556903,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 500,
                "random_state": 42
            }
        },
        {
            "id": "cfg3",
            "name": "lemon-sweep-1",
            "tau": 0.264227,
            "params": {
                "C": 2.965034,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 500,
                "random_state": 42
            }
        }
    ],
    "mlp": [
        {
            "id": "cfg1",
            "name": "jumping-sweep-17",
            "tau": 0.265700,
            "params": {
                "hidden_layers": [512, 256, 128],
                "dropout": 0.4,
                "lr": 0.000584,
                "weight_decay": 0.000091,
                "batch_size": 2048,
                "epochs": 15
            }
        },
        {
            "id": "cfg2",
            "name": "brisk-sweep-25",
            "tau": 0.274520,
            "params": {
                "hidden_layers": [512, 256, 128],
                "dropout": 0.4,
                "lr": 0.000392,
                "weight_decay": 0.000068,
                "batch_size": 2048,
                "epochs": 15
            }
        },
        {
            "id": "cfg3",
            "name": "silver-sweep-27",
            "tau": 0.261401,
            "params": {
                "hidden_layers": [512, 256, 128],
                "dropout": 0.4,
                "lr": 0.000550,
                "weight_decay": 0.000115,
                "batch_size": 2048,
                "epochs": 15
            }
        }
    ]
}

# -----------------------------------------------------------------------------
# ARQUITECTURA TABULAR MLP (PYTORCH)
# -----------------------------------------------------------------------------
class TabularMLP(nn.Module):
    def __init__(self, in_features: int, hidden_layers: List[int], dropout: float = 0.2):
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


def entrenar_modelo_mlp(X_tr: np.ndarray, y_tr: np.ndarray, params: Dict[str, Any]) -> TabularMLP:
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    model = TabularMLP(
        in_features=X_tr.shape[1],
        hidden_layers=params["hidden_layers"],
        dropout=params.get("dropout", 0.4)
    ).to(DEVICE)

    pos_w = (len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4)
    )

    batch_size = params.get("batch_size", 2048)
    n_samples = X_tr_t.shape[0]
    epochs = params.get("epochs", 15)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_samples, device=DEVICE)
        for j in range(0, n_samples, batch_size):
            idx = perm[j : j + batch_size]
            optimizer.zero_grad(set_to_none=True)
            out = model(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            optimizer.step()

    del X_tr_t, y_tr_t, optimizer, criterion
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return model


def predecir_mlp(model: TabularMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).squeeze(1).cpu().numpy()
    del X_t
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return probs


# -----------------------------------------------------------------------------
# FUNCIÓN AUXILIAR DE HOMOGENEIZACIÓN DE CATEGORÍAS
# -----------------------------------------------------------------------------
def homogeneizar_categorica(serie: pd.Series) -> pd.Series:
    """
    Rellena valores nulos y asegura tipo string unificado.
    Evita desajustes como comparar 1 con 1.0 o NaN en OneHotEncoder.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return serie.fillna(-1).astype(int).astype(str)
    return serie.fillna("missing").astype(str)


# -----------------------------------------------------------------------------
# CARGA DE DATOS, PURGA INTER-PANEL Y PIPELINE DE PREPROCESAMIENTO
# -----------------------------------------------------------------------------
def preparar_datos():
    print(f"\n[1/4] Leyendo dataset desde: {PARQUET_FILE}")
    df = pd.read_parquet(PARQUET_FILE)

    # Identificación estricta del hogar (codusu + nro_hogar)
    df["id_hogar"] = df["codusu"].astype(str) + "_" + df["nro_hogar"].astype(str)

    df_train_raw = df[df["ano4"] <= 2021].copy()
    df_test = df[df["ano4"] > 2021].copy().reset_index(drop=True)

    print(f"      Observaciones Train inicial (<=2021): {len(df_train_raw):,}")
    print(f"      Observaciones Test OOS (>2021):       {len(df_test):,}")

    # PURGA DE HOGARES: Identificar hogares de test y excluirlos de train
    hogares_en_test = set(df_test["id_hogar"].unique())
    mascara_purga = ~df_train_raw["id_hogar"].isin(hogares_en_test)
    df_train = df_train_raw[mascara_purga].copy().reset_index(drop=True)

    hogares_removidos = len(df_train_raw) - len(df_train)
    print(f"      ✓ Purga aplicada: {hogares_removidos:,} registros eliminados de train por solapamiento de panel.")
    print(f"      Observaciones Train final depurado:   {len(df_train):,}")

    del df_train_raw
    gc.collect()

    cols_excluir = {
        "id_hogar", "codusu", "nro_hogar", "ano4", "trimestre", "pobre", "linea_pobreza",
        "pondera", "pondih", "pondiio", "pondii",
        "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
        "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19_a", "v19_b", "v19_am", "v21", "v22",
        "vii1_1", "vii1_2", "vii2_1", "vii2_2", "vii2_3", "vii2_4",
        "ix_tot", "ix_men10", "ix_mayeq10"
    }

    todas_cols = [c for c in df_train.columns if c not in cols_excluir]

    nominales_y_binarias_encuesta = [c for c in [
        "region_estudio", "iv1", "iv3", "iv4", "iv5", "iv6", "iv8", "iv9",
        "iv10", "iv11", "iv12_1", "iv12_2", "iv12_3", "ii3", "ii3_1",
        "ii4_1", "ii4_2", "ii4_3", "ii5", "ii5_1", "ii6", "ii6_1", "ii7", "ii8"
    ] if c in todas_cols]

    ordinales = [c for c in [
        "max_educ_hogar", "jefe_nivel_ed", "iv7", "ii9"
    ] if c in todas_cols]

    binarias_puras = [c for c in [
        "jefe_mujer", "jefe_ocupado", "jefe_desocupado", "jefe_informal", "hacinamiento_critico"
    ] if c in todas_cols]

    numericas = [
        c for c in todas_cols
        if c not in nominales_y_binarias_encuesta and c not in ordinales and c not in binarias_puras
    ]

    # Pre-limpieza y homogenización de tipos categóricos
    for c in nominales_y_binarias_encuesta:
        df_train[c] = homogeneizar_categorica(df_train[c])
        df_test[c] = homogeneizar_categorica(df_test[c])

    for c in ordinales:
        df_train[c] = df_train[c].fillna(-1)
        df_test[c] = df_test[c].fillna(-1)

    print(f"\n[2/4] Ajustando ColumnTransformer en Train (2016-2021 purgado)...")
    preprocessor = ColumnTransformer(transformers=[
        ("nom", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), nominales_y_binarias_encuesta),
        ("ord", Pipeline([
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler())
        ]), ordinales),
        ("bin", "passthrough", binarias_puras),
        ("num", StandardScaler(), numericas)
    ], remainder="drop")

    X_train = preprocessor.fit_transform(df_train[todas_cols]).astype(np.float32)
    y_train = df_train["pobre"].values.astype(np.int32)

    X_test = preprocessor.transform(df_test[todas_cols]).astype(np.float32)
    y_test = df_test["pobre"].values.astype(np.int32) if "pobre" in df_test.columns else None

    # Guardar artefactos
    joblib.dump(preprocessor, OUTPUT_MODEL_DIR / "preprocessor_fitted.joblib")
    joblib.dump(todas_cols, OUTPUT_MODEL_DIR / "feature_names_raw.joblib")

    return df_train, df_test, X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------------
# ENTRENAMIENTO E INFERENCIA POR FAMILIA
# -----------------------------------------------------------------------------
def ejecutar_familia(familia: str, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, df_test: pd.DataFrame):
    print(f"\n[3/4] Reentrenando Top 3 de la familia: {familia.upper()}...")
    configs = CONFIGS_TOP3[familia]

    cols_meta = ["codusu", "nro_hogar", "ano4", "trimestre"]
    for c in ["pobre", "pondih", "pondera", "linea_pobreza"]:
        if c in df_test.columns:
            cols_meta.append(c)

    res_df = df_test[cols_meta].copy()

    hard_preds = []
    prob_preds = []

    scale_pos = (len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1)

    for cfg_info in configs:
        cfg_id = cfg_info["id"]
        cfg_name = cfg_info["name"]
        tau = cfg_info["tau"]
        params = cfg_info["params"]

        print(f"   -> Entrenando {cfg_id} ({cfg_name}) con tau = {tau:.4f}...")

        if familia == "logistic":
            # Sin 'penalty' ni 'n_jobs' (obsoletos/deprecados en sklearn >= 1.8)
            model = LogisticRegression(
                C=params["C"],
                solver=params.get("solver", "lbfgs"),
                class_weight=params.get("class_weight", "balanced"),
                max_iter=params.get("max_iter", 500),
                random_state=42
            )
            model.fit(X_tr, y_tr)
            probs_te = model.predict_proba(X_te)[:, 1]
            joblib.dump(model, OUTPUT_MODEL_DIR / f"{familia}_{cfg_id}.joblib")

        elif familia == "random_forest":
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                class_weight=params["class_weight"],
                random_state=params["random_state"],
                n_jobs=-1
            )
            with parallel_backend("loky"):
                model.fit(X_tr, y_tr)
                probs_te = model.predict_proba(X_te)[:, 1]
            get_reusable_executor().shutdown(wait=True)
            joblib.dump(model, OUTPUT_MODEL_DIR / f"{familia}_{cfg_id}.joblib")

        elif familia == "xgboost":
            xgb_major_ver = int(xgb.__version__.split(".")[0])
            xgb_params = {
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "scale_pos_weight": scale_pos * params["scale_pos_factor"],
                "random_state": 42
            }
            if DEVICE == "cuda":
                if xgb_major_ver >= 2:
                    xgb_params["tree_method"] = "hist"
                    xgb_params["device"] = "cuda"
                else:
                    xgb_params["tree_method"] = "gpu_hist"
            else:
                xgb_params["tree_method"] = "hist"
                xgb_params["n_jobs"] = -1

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, verbose=False)
            probs_te = model.predict_proba(X_te)[:, 1]
            joblib.dump(model, OUTPUT_MODEL_DIR / f"{familia}_{cfg_id}.joblib")

        elif familia == "mlp":
            model = entrenar_modelo_mlp(X_tr, y_tr, params)
            probs_te = predecir_mlp(model, X_te)
            torch.save(model.state_dict(), OUTPUT_MODEL_DIR / f"{familia}_{cfg_id}.pt")

        # Inferencia binaria según el tau óptimo
        preds_te = (probs_te >= tau).astype(int)

        res_df[f"prob_{familia}_{cfg_id}"] = probs_te.astype(np.float32)
        res_df[f"pred_{familia}_{cfg_id}"] = preds_te.astype(np.int8)

        prob_preds.append(probs_te)
        hard_preds.append(preds_te)

    # -------------------------------------------------------------------------
    # ENSAMBLES: VOTACIÓN MAYORITARIA Y PROMEDIO CONTINUO
    # -------------------------------------------------------------------------
    print(f"   -> Construyendo ensambles para {familia.upper()}...")
    
    # 1. Votación Mayoritaria (Hard): >= 2 votos de 3
    suma_votos = hard_preds[0] + hard_preds[1] + hard_preds[2]
    voto_mayoria = (suma_votos >= 2).astype(np.int8)
    res_df[f"pred_{familia}_ensamble_votacion"] = voto_mayoria

    # 2. Promedio continuo de probabilidades (Soft)
    prob_media = (prob_preds[0] + prob_preds[1] + prob_preds[2]) / 3.0
    res_df[f"prob_{familia}_ensamble_promedio"] = prob_media.astype(np.float32)

    # Exportar archivo Parquet
    out_file = OUTPUT_PRED_DIR / f"predicciones_oos_{familia}.parquet"
    print(f"\n[4/4] Guardando archivo Parquet: {out_file.resolve()}")
    res_df.to_parquet(out_file, index=False)
    print(f"      ✓ Predicciones OOS guardadas exitosamente ({len(res_df):,} filas).")


# -----------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL CON CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reentrenamiento y generación de predicciones OOS")
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest", "xgboost", "mlp", "all"],
        required=True,
        help="Familia de modelos a reentrenar. Usar 'all' para correr las 4 en una máquina con GPU."
    )
    args = parser.parse_args()

    df_train, df_test, X_train, y_train, X_test, y_test = preparar_datos()

    modelos_a_ejecutar = [args.model] if args.model != "all" else ["logistic", "random_forest", "xgboost", "mlp"]

    for mod in modelos_a_ejecutar:
        ejecutar_familia(mod, X_train, y_train, X_test, df_test)

    print("\n" + "=" * 70)
    print("PROCESO FINALIZADO CON ÉXITO.")
    print("=" * 70)
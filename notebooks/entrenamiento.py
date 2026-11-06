"""
===============================================================================
PROYECTO: Modelado de Pobreza con EPH (Machine Learning)
SCRIPT: Sintonización Bayesiana Unificada con W&B y Backtesting Walk-Forward
ARQUITECTURAS: Logistic Regression, Random Forest, XGBoost y Tabular MLP
CORRECCIONES:
- Clasificación rigurosa de features (Nominales, Ordinales, Binarias 0/1 y Continuas)
- Escalado de variables ordinales para MLP y Logistic
- Función constructora robusta para XGBoost
- Control de hilos para permitir 4 ejecuciones en paralelo
===============================================================================
"""

# 1. IMPORTACIÓN PRIORITARIA DE PYTORCH (Evita colisiones de DLLs en Windows)
import torch
import torch.nn as nn

import os
import sys

print("=" * 70)
print(f"Intérprete Python en uso: {sys.executable}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de cómputo para PyTorch: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"GPU activa: {torch.cuda.get_device_name(0)}")
else:
    print("ADVERTENCIA: PyTorch en CPU.")
print("=" * 70 + "\n")

# Configuración de hilos base
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import wandb

import joblib
from joblib import parallel_backend
from joblib.externals.loky import get_reusable_executor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss
import xgboost as xgb

warnings.filterwarnings("ignore")

PARQUET_FILE = Path("data/eph_hogares_consolidada_2016_2026.parquet")
if not PARQUET_FILE.exists():
    PARQUET_FILE = Path("../data/eph_hogares_consolidada_2016_2026.parquet")


random_forest_n_jobs = -1


# =============================================================================
# 1. EVALUACIÓN Y CALIBRACIÓN VECTORIZADA DE MÉTRICAS (F2-SCORE Y PR-AUC)
# =============================================================================
def calcular_metricas(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_probs)

    denominador = (4 * precision + recall)
    f2_scores = np.divide(
        5 * (precision * recall),
        denominador,
        out=np.zeros_like(precision),
        where=denominador > 0
    )

    best_idx = np.argmax(f2_scores)
    best_f2 = float(f2_scores[best_idx])
    best_tau = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred = (y_probs >= best_tau).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    recall_opt = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision_opt = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    especificidad_opt = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_opt = float(2 * (precision_opt * recall_opt) / (precision_opt + recall_opt)) if (precision_opt + recall_opt) > 0 else 0.0

    brier = float(brier_score_loss(y_true, y_probs))
    eps = 1e-7
    y_probs_clip = np.clip(y_probs, eps, 1.0 - eps)
    logloss = float(-np.mean(y_true * np.log(y_probs_clip) + (1.0 - y_true) * np.log(1.0 - y_probs_clip)))

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f2_score": best_f2,
        "f1_score": f1_opt,
        "recall": recall_opt,
        "precision": precision_opt,
        "especificidad": especificidad_opt,
        "brier_score": brier,
        "log_loss": logloss,
        "optimal_threshold": best_tau
    }

# =============================================================================
# 2. RED NEURONAL TABULAR (PYTORCH)
# =============================================================================
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

def entrenar_evaluar_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray, 
    X_va: np.ndarray, y_va: np.ndarray, 
    config: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)

    model = TabularMLP(
        in_features=X_tr.shape[1],
        hidden_layers=config["hidden_layers"],
        dropout=config.get("dropout", 0.2)
    ).to(DEVICE)

    pos_w = (len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.get("lr", 1e-3), 
        weight_decay=config.get("weight_decay", 1e-4)
    )

    batch_size = config.get("batch_size", 2048)
    n_samples = X_tr_t.shape[0]
    epochs = config.get("epochs", 15)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_samples, device=DEVICE)
        for j in range(0, n_samples, batch_size):
            indices = perm[j : j + batch_size]
            optimizer.zero_grad(set_to_none=True)
            out = model(X_tr_t[indices])
            loss = criterion(out, y_tr_t[indices])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs_tr = torch.sigmoid(model(X_tr_t)).squeeze(1).cpu().numpy()
        probs_va = torch.sigmoid(model(X_va_t)).squeeze(1).cpu().numpy()

    del model, optimizer, criterion, X_tr_t, y_tr_t, X_va_t
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return calcular_metricas(y_tr, probs_tr), calcular_metricas(y_va, probs_va)

# =============================================================================
# 3. CONSTRUCTOR ROBUSTO PARA XGBOOST
# =============================================================================
def construir_modelo_xgboost(cfg: Any, scale_pos: float) -> xgb.XGBClassifier:
    xgb_major_ver = int(xgb.__version__.split(".")[0])
    
    params = {
        "n_estimators": 400,
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "scale_pos_weight": scale_pos * cfg.scale_pos_factor,
        "early_stopping_rounds": 30,
        "eval_metric": "aucpr",
        "random_state": 42
    }

    if DEVICE == "cuda":
        if xgb_major_ver >= 2:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        else:
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = 0
    else:
        params["tree_method"] = "hist"
        params["n_jobs"] = 4  # Límite controlado para no saturar CPU

    return xgb.XGBClassifier(**params)

# =============================================================================
# 4. PREPROCESAMIENTO RIGUROSO Y CACHÉ DE FOLDS EN MEMORIA
# =============================================================================
PREPARED_FOLDS: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

def cargar_y_preprocesar_folds():
    global PREPARED_FOLDS
    if len(PREPARED_FOLDS) > 0:
        return

    print("[Inicialización] Cargando Parquet y aplicando partición estricta de variables...")
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {PARQUET_FILE.resolve()}")

    df = pd.read_parquet(PARQUET_FILE)
    df_dev = df[df["ano4"] <= 2021].reset_index(drop=True)

    cols_excluir = {
        "codusu", "nro_hogar", "ano4", "trimestre", "pobre", "linea_pobreza",
        "pondera", "pondih", "pondiio", "pondii",
        "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", 
        "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19_a", "v19_b", "v19_am", "v21", "v22",
        "vii1_1", "vii1_2", "vii2_1", "vii2_2", "vii2_3", "vii2_4",
        "ix_tot", "ix_men10", "ix_mayeq10"
    }

    todas_cols = [c for c in df_dev.columns if c not in cols_excluir]

    nans_totales = df_dev[todas_cols].isna().sum()
    cols_con_nan = nans_totales[nans_totales > 0]
    if not cols_con_nan.empty:
        detalle = "\n".join([f"  - {col}: {cant:,} NaNs" for col, cant in cols_con_nan.items()])
        raise ValueError(
            f"\n\n{'!'*70}\n"
            f"¡FRENO DE EMERGENCIA! Se detectaron NaNs y la imputación está prohibida:\n{detalle}\n"
            f"{'!'*70}\n"
        )

    # -------------------------------------------------------------------------
    # PARTICIÓN FORMAL SEGÚN DISEÑO DE REGISTRO EPH
    # -------------------------------------------------------------------------
    # 1. Nominales y Categóricas dicotómicas (1=Sí, 2=No) -> OneHotEncoder
    nominales_y_binarias_encuesta = [c for c in [
        "region_estudio", "iv1", "iv3", "iv4", "iv5", "iv6", "iv8", "iv9", 
        "iv10", "iv11", "iv12_1", "iv12_2", "iv12_3", "ii3", "ii3_1", 
        "ii4_1", "ii4_2", "ii4_3", "ii5", "ii5_1", "ii6", "ii6_1", "ii7", "ii8"
    ] if c in todas_cols]

    # 2. Ordinales con jerarquía natural -> OrdinalEncoder + StandardScaler
    ordinales = [c for c in [
        "max_educ_hogar", "jefe_nivel_ed", "iv7", "ii9"
    ] if c in todas_cols]

    # 3. Binarias nativas construidas en Python (0 ó 1 puro) -> Passthrough directo
    binarias_puras = [c for c in [
        "jefe_mujer", "jefe_ocupado", "jefe_desocupado", "jefe_informal", "hacinamiento_critico"
    ] if c in todas_cols]

    # 4. Continuas, conteos y ratios -> StandardScaler
    numericas = [c for c in todas_cols if c not in nominales_y_binarias_encuesta and c not in ordinales and c not in binarias_puras]

    print(f"Features identificadas -> Categóricas One-Hot: {len(nominales_y_binarias_encuesta)} | Ordinales: {len(ordinales)} | Binarias 0/1: {len(binarias_puras)} | Numéricas: {len(numericas)}")

    periodos = (
        df_dev[["ano4", "trimestre"]]
        .drop_duplicates()
        .sort_values(by=["ano4", "trimestre"])
        .values.tolist()
    )

    train_quarters = 8
    for i in range(train_quarters, len(periodos)):
        val_ano, val_trim = periodos[i]
        train_window = periodos[i - train_quarters : i]

        train_mask = pd.Series(False, index=df_dev.index)
        for a, t in train_window:
            train_mask |= (df_dev["ano4"] == a) & (df_dev["trimestre"] == t)
        val_mask = (df_dev["ano4"] == val_ano) & (df_dev["trimestre"] == val_trim)

        # Purga inter-panel obligatoria
        hogares_val = set(df_dev.loc[val_mask, "codusu"].unique())
        train_mask_purged = train_mask & (~df_dev["codusu"].isin(hogares_val))

        train_sub = df_dev[train_mask_purged]
        val_sub = df_dev[val_mask]

        # Pipeline sin imputadores
        preprocessor = ColumnTransformer(transformers=[
            ("nom", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), nominales_y_binarias_encuesta),
            ("ord", Pipeline([
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ("scaler", StandardScaler())
            ]), ordinales),
            ("bin", "passthrough", binarias_puras),
            ("num", StandardScaler(), numericas)
        ], remainder="drop")

        X_tr = preprocessor.fit_transform(train_sub[todas_cols]).astype(np.float32)
        y_tr = train_sub["pobre"].values.astype(np.int32)
        X_va = preprocessor.transform(val_sub[todas_cols]).astype(np.float32)
        y_va = val_sub["pobre"].values.astype(np.int32)

        if np.isnan(X_tr).any() or np.isnan(X_va).any():
            raise ValueError("¡FRENO DE EMERGENCIA! Se detectaron NaNs post-transformación.")

        PREPARED_FOLDS.append((X_tr, y_tr, X_va, y_va))

    del df, df_dev
    gc.collect()
    print(f"✓ {len(PREPARED_FOLDS)} pliegues procesados correctamente y retenidos en RAM.\n")

# =============================================================================
# 5. WORKER UNIFICADO DE W&B SWEEPS
# =============================================================================
def train_sweep_iteration():
    run = wandb.init()
    cfg = wandb.config

    cargar_y_preprocesar_folds()

    val_metrics_list = []
    train_metrics_list = []

    for X_tr, y_tr, X_va, y_va in PREPARED_FOLDS:
        if cfg.model_family == "logistic":
            model = LogisticRegression(
                C=cfg.C,
                penalty="l2",
                solver="lbfgs",
                tol=1e-3,
                class_weight="balanced",
                max_iter=500,
                random_state=42,
                n_jobs=2  # Balanceado para paralelismo
            )
            model.fit(X_tr, y_tr)
            probs_tr = model.predict_proba(X_tr)[:, 1]
            probs_va = model.predict_proba(X_va)[:, 1]
            m_tr = calcular_metricas(y_tr, probs_tr)
            m_va = calcular_metricas(y_va, probs_va)

        elif cfg.model_family == "random_forest":
            num_cpus = 4 if os.cpu_count() and os.cpu_count() >= 8 else max(1, (os.cpu_count() or 4) - 2)

            max_feat = cfg.max_features
            try:
                max_feat = float(max_feat)
            except (ValueError, TypeError):
                max_feat = str(max_feat)

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=int(cfg.max_depth),
                min_samples_leaf=int(cfg.min_samples_leaf),
                max_features=max_feat,
                class_weight="balanced",
                random_state=42
            )
            with parallel_backend('threading', n_jobs=num_cpus):
                model.fit(X_tr, y_tr)
                probs_tr = model.predict_proba(X_tr)[:, 1]
                probs_va = model.predict_proba(X_va)[:, 1]
            m_tr = calcular_metricas(y_tr, probs_tr)
            m_va = calcular_metricas(y_va, probs_va)

        elif cfg.model_family == "xgboost":
            scale_pos = (len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1)
            model = construir_modelo_xgboost(cfg, scale_pos)
            
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            except Exception as e:
                if "CUDA" in str(e) or "GPU" in str(e) or "device" in str(e):
                    xgb_ver = int(xgb.__version__.split(".")[0])
                    if xgb_ver >= 2:
                        model.set_params(tree_method="hist", device="cpu", n_jobs=4)
                    else:
                        model.set_params(tree_method="hist", n_jobs=4)
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    raise e

            probs_tr = model.predict_proba(X_tr)[:, 1]
            probs_va = model.predict_proba(X_va)[:, 1]
            m_tr = calcular_metricas(y_tr, probs_tr)
            m_va = calcular_metricas(y_va, probs_va)

        elif cfg.model_family == "mlp":
            config_mlp = {
                "hidden_layers": cfg.hidden_layers,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "batch_size": 2048,
                "epochs": 15
            }
            m_tr, m_va = entrenar_evaluar_mlp(X_tr, y_tr, X_va, y_va, config_mlp)

        train_metrics_list.append(m_tr)
        val_metrics_list.append(m_va)

    if cfg.model_family == "random_forest":
        get_reusable_executor().shutdown(wait=True)
    gc.collect()

    wandb.log({
        "val_mean_pr_auc": float(np.mean([m["pr_auc"] for m in val_metrics_list])),
        "train_mean_pr_auc": float(np.mean([m["pr_auc"] for m in train_metrics_list])),
        "val_mean_roc_auc": float(np.mean([m["roc_auc"] for m in val_metrics_list])),
        "train_mean_roc_auc": float(np.mean([m["roc_auc"] for m in train_metrics_list])),
        "val_mean_f2_score": float(np.mean([m["f2_score"] for m in val_metrics_list])),
        "train_mean_f2_score": float(np.mean([m["f2_score"] for m in train_metrics_list])),
        "val_mean_f1_score": float(np.mean([m["f1_score"] for m in val_metrics_list])),
        "train_mean_f1_score": float(np.mean([m["f1_score"] for m in train_metrics_list])),
        "val_mean_recall": float(np.mean([m["recall"] for m in val_metrics_list])),
        "train_mean_recall": float(np.mean([m["recall"] for m in train_metrics_list])),
        "val_mean_precision": float(np.mean([m["precision"] for m in val_metrics_list])),
        "train_mean_precision": float(np.mean([m["precision"] for m in train_metrics_list])),
        "val_mean_especificidad": float(np.mean([m["especificidad"] for m in val_metrics_list])),
        "train_mean_especificidad": float(np.mean([m["especificidad"] for m in train_metrics_list])),
        "val_mean_brier_score": float(np.mean([m["brier_score"] for m in val_metrics_list])),
        "train_mean_brier_score": float(np.mean([m["brier_score"] for m in train_metrics_list])),
        "val_mean_log_loss": float(np.mean([m["log_loss"] for m in val_metrics_list])),
        "train_mean_log_loss": float(np.mean([m["log_loss"] for m in train_metrics_list])),
        "mean_optimal_threshold": float(np.mean([m["optimal_threshold"] for m in val_metrics_list])),
        "gap_pr_auc": float(np.mean([m["pr_auc"] for m in train_metrics_list]) - np.mean([m["pr_auc"] for m in val_metrics_list])),
        "gap_f2_score": float(np.mean([m["f2_score"] for m in train_metrics_list]) - np.mean([m["f2_score"] for m in val_metrics_list]))
    })
    run.finish()

# =============================================================================
# 6. CONFIGURACIONES DE SWEEPS BAYESIANOS
# =============================================================================
SWEEP_CONFIGS = {
    "logistic": {
        "method": "bayes",
        "metric": {"name": "val_mean_pr_auc", "goal": "maximize"},
        "parameters": {
            "model_family": {"value": "logistic"},
            "C": {"distribution": "log_uniform_values", "min": 1e-4, "max": 100.0}
        }
    },
    "random_forest": {
        "method": "bayes",
        "metric": {"name": "val_mean_pr_auc", "goal": "maximize"},
        "parameters": {
            "model_family": {"value": "random_forest"},
            "max_depth": {"distribution": "int_uniform", "min": 6, "max": 22},
            "min_samples_leaf": {"distribution": "int_uniform", "min": 4, "max": 35},
            "max_features": {"values": ["sqrt", 0.25, 0.4, 0.55]}
        }
    },
    "xgboost": {
        "method": "bayes",
        "metric": {"name": "val_mean_pr_auc", "goal": "maximize"},
        "parameters": {
            "model_family": {"value": "xgboost"},
            "learning_rate": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.2},
            "max_depth": {"values": [3, 4, 6, 8]},
            "subsample": {"distribution": "uniform", "min": 0.6, "max": 1.0},
            "colsample_bytree": {"distribution": "uniform", "min": 0.6, "max": 1.0},
            "scale_pos_factor": {"values": [0.8, 1.0, 1.2]}
        }
    },
    "mlp": {
        "method": "bayes",
        "metric": {"name": "val_mean_pr_auc", "goal": "maximize"},
        "parameters": {
            "model_family": {"value": "mlp"},
            "hidden_layers": {"values": [[256, 128], [512, 256, 128], [128, 64]]},
            "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
            "dropout": {"values": [0.1, 0.2, 0.3, 0.4]},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2}
        }
    }
}

# =============================================================================
# 7. ENTRADA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento con control estricto de NaNs")
    parser.add_argument("--model", choices=["logistic", "random_forest", "xgboost", "mlp"], required=True)
    parser.add_argument("--project", default="eph-pobreza-ml", help="Nombre del proyecto en W&B")
    parser.add_argument("--count", type=int, default=30, help="Cantidad de iteraciones bayesianas")
    args = parser.parse_args()

    cargar_y_preprocesar_folds()

    sweep_config = SWEEP_CONFIGS[args.model]
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"Iniciando Sweep Bayesiano en W&B para {args.model.upper()}. Sweep ID: {sweep_id}")
    
    wandb.agent(sweep_id, function=train_sweep_iteration, count=args.count)
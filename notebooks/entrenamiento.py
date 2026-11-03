"""
===============================================================================
PROYECTO: Modelado de Pobreza con EPH (Machine Learning)
SCRIPT: Sintonización Bayesiana Unificada con W&B y Backtesting Walk-Forward
ARQUITECTURAS: Logistic Regression, Random Forest, XGBoost y Tabular MLP
===============================================================================
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import joblib
from joblib import parallel_backend
from joblib.externals.loky import get_reusable_executor

import gc
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import wandb

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc
import xgboost as xgb

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# 1. Detección y diagnóstico de dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Dispositivo de cómputo seleccionado: {DEVICE.upper()} ---")
if DEVICE == "cuda":
    print(f"GPU activa: {torch.cuda.get_device_name(0)}")
else:
    print("ADVERTENCIA: GPU no detectada por PyTorch. Corriendo en CPU.")

# Resolución robusta de la ruta del dataset
PARQUET_FILE = Path("../data/eph_hogares_consolidada_2016_2026.parquet")
if not PARQUET_FILE.exists():
    PARQUET_FILE = Path("data/eph_hogares_consolidada_2016_2026.parquet")

# =============================================================================
# 1. EVALUACIÓN Y CALIBRACIÓN VECTORIZADA DE UMBRAL (F2-SCORE Y PR-AUC)
# =============================================================================
def calcular_metricas(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    # Cálculo vectorizado sin bucles: F2 = 5 * (P * R) / (4 * P + R)
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
    recall_opt = float(recall[best_idx])

    return {
        "pr_auc": float(pr_auc),
        "f2_score": best_f2,
        "optimal_threshold": best_tau,
        "recall_at_optimal": recall_opt
    }

# =============================================================================
# 2. ARQUITECTURA RED NEURONAL TABULAR (PYTORCH)
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
) -> Dict[str, float]:
    
    # Transferencia directa a VRAM (los 110k registros ocupan ~15 MB)
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
        probs = torch.sigmoid(model(X_va_t)).squeeze(1).cpu().numpy()

    # Liberación forzada de memoria de video entre folds
    del model, optimizer, criterion, X_tr_t, y_tr_t, X_va_t
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return calcular_metricas(y_va, probs)

# =============================================================================
# 3. PREPROCESAMIENTO GLOBAL Y CACHÉ DE PLIEGUES EN MEMORIA
# =============================================================================
PREPARED_FOLDS: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

def cargar_y_preprocesar_folds():
    global PREPARED_FOLDS
    if len(PREPARED_FOLDS) > 0:
        return

    print("\n[Inicialización] Cargando Parquet y preparando folds temporales con purga...")
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {PARQUET_FILE.resolve()}")

    df = pd.read_parquet(PARQUET_FILE)
    df_dev = df[df["ano4"] <= 2021].reset_index(drop=True)

    periodos = (
        df_dev[["ano4", "trimestre"]]
        .drop_duplicates()
        .sort_values(by=["ano4", "trimestre"])
        .values.tolist()
    )

    cols_excluir = {"codusu", "nro_hogar", "linea_pobreza", "pobre", "ano4", "trimestre"}
    todas_cols = [c for c in df_dev.columns if c not in cols_excluir]

    nominales = [c for c in ["region_estudio", "iv1", "ii7"] if c in todas_cols]
    ordinales = [c for c in ["max_educ_hogar", "jefe_nivel_ed"] if c in todas_cols]
    numericas = [c for c in todas_cols if c not in nominales and c not in ordinales]

    train_quarters = 8
    for i in range(train_quarters, len(periodos)):
        val_ano, val_trim = periodos[i]
        train_window = periodos[i - train_quarters : i]

        train_mask = pd.Series(False, index=df_dev.index)
        for a, t in train_window:
            train_mask |= (df_dev["ano4"] == a) & (df_dev["trimestre"] == t)
        val_mask = (df_dev["ano4"] == val_ano) & (df_dev["trimestre"] == val_trim)

        # Purga inter-panel (EPH rotativo 2-2-2)
        hogares_val = set(df_dev.loc[val_mask, "codusu"].unique())
        train_mask_purged = train_mask & (~df_dev["codusu"].isin(hogares_val))

        train_sub = df_dev[train_mask_purged]
        val_sub = df_dev[val_mask]

        # Pipeline con imputación robusta contra NaNs
        preprocessor = ColumnTransformer(transformers=[
            ("nom", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
                ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
            ]), nominales),
            ("ord", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), ordinales),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numericas)
        ], remainder="drop")

        X_tr = preprocessor.fit_transform(train_sub[todas_cols]).astype(np.float32)
        y_tr = train_sub["pobre"].values.astype(np.int32)
        X_va = preprocessor.transform(val_sub[todas_cols]).astype(np.float32)
        y_va = val_sub["pobre"].values.astype(np.int32)

        PREPARED_FOLDS.append((X_tr, y_tr, X_va, y_va))

    del df, df_dev
    gc.collect()
    print(f"✓ {len(PREPARED_FOLDS)} pliegues procesados y almacenados en RAM. Listos para sweeps.\n")

# =============================================================================
# 4. WORKER UNIFICADO DE W&B SWEEPS
# =============================================================================
def train_sweep_iteration():
    run = wandb.init()
    cfg = wandb.config

    # Asegurar que los datos estén cargados en memoria
    cargar_y_preprocesar_folds()

    pr_aucs, f2s = [], []

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
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            metrics = calcular_metricas(y_va, probs)

        elif cfg.model_family == "random_forest":
            # Fijamos un número razonable de hilos (ej. 6 u 8) para no saturar el SO
            num_cpus = max(1, os.cpu_count() - 2) if os.cpu_count() else 4

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                max_features=cfg.max_features,
                class_weight="balanced",
                random_state=42
            )
            
            # Usar hilos en lugar de procesos evita los cuelgues de memoria compartida en Windows
            with parallel_backend('threading', n_jobs=num_cpus):
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_va)[:, 1]

            metrics = calcular_metricas(y_va, probs)

        elif cfg.model_family == "xgboost":
            scale_pos = (len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1)[cite: 4]
            model = xgb.XGBClassifier(
                tree_method="hist",
                device=DEVICE,
                n_estimators=400,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                scale_pos_weight=scale_pos * cfg.scale_pos_factor,
                early_stopping_rounds=30,
                eval_metric="aucpr",
                random_state=42
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            probs = model.predict_proba(X_va)[:, 1]
            metrics = calcular_metricas(y_va, probs)

        elif cfg.model_family == "mlp":
            config_mlp = {
                "hidden_layers": cfg.hidden_layers,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "batch_size": 2048,
                "epochs": 15
            }
            metrics = entrenar_evaluar_mlp(X_tr, y_tr, X_va, y_va, config_mlp)

        pr_aucs.append(metrics["pr_auc"])
        f2s.append(metrics["f2_score"])

    # Logueo del promedio de métricas sobre los 14 folds
    wandb.log({
        "mean_pr_auc": float(np.mean(pr_aucs)),
        "mean_f2_score": float(np.mean(f2s))
    })
    # Forzar apagado de procesos huérfanos y limpieza de RAM
    if cfg.model_family == "random_forest":
        get_reusable_executor().shutdown(wait=True)
    
    gc.collect()

    wandb.log({
        "mean_pr_auc": float(np.mean(pr_aucs)),
        "mean_f2_score": float(np.mean(f2s))
    })
    run.finish()

# =============================================================================
# 5. CONFIGURACIONES DE SWEEPS BAYESIANOS POR MODELO
# =============================================================================
SWEEP_CONFIGS = {
    "logistic": {
        "method": "bayes",
        "metric": {"name": "mean_pr_auc", "goal": "maximize"},
        "parameters": {
            "model_family": {"value": "logistic"},
            "C": {"distribution": "log_uniform_values", "min": 1e-4, "max": 100.0}
        }
    },
    "random_forest": {
        "method": "bayes",
        "metric": {"name": "mean_pr_auc", "goal": "maximize"},
        "parameters": {
            "model_family": {"value": "random_forest"},
            "max_depth": {"values": [8, 12, 16, 20]},
            "min_samples_leaf": {"values": [5, 10, 20, 30]},
            "max_features": {"values": ["sqrt", 0.3, 0.5]}
        }
    },
    "xgboost": {
        "method": "bayes",
        "metric": {"name": "mean_pr_auc", "goal": "maximize"},
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
        "metric": {"name": "mean_pr_auc", "goal": "maximize"},
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
# 6. ENTRADA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento con EPH")
    parser.add_argument("--model", choices=["logistic", "random_forest", "xgboost", "mlp"], required=True)
    parser.add_argument("--project", default="eph-pobreza-ml", help="Nombre del proyecto en W&B")
    parser.add_argument("--count", type=int, default=30, help="Cantidad de iteraciones bayesianas")
    args = parser.parse_args()

    # Preprocesar los 14 folds en RAM antes de iniciar el agente
    cargar_y_preprocesar_folds()

    sweep_config = SWEEP_CONFIGS[args.model]
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"Iniciando Sweep Bayesiano en W&B para {args.model.upper()}. Sweep ID: {sweep_id}")
    
    wandb.agent(sweep_id, function=train_sweep_iteration, count=args.count)
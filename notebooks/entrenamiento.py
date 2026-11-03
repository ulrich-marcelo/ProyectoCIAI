"""
===============================================================================
PROYECTO: Modelado de Pobreza con EPH (Machine Learning)
SCRIPT: Sintonización Bayesiana con W&B y Backtesting Walk-Forward con Purga
===============================================================================
"""

import gc
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import wandb

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, fbeta_score
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARQUET_FILE = Path("data/eph_hogares_consolidada_2016_2026.parquet")

# =============================================================================
# 1. SPLITTER TEMPORAL CON PURGA INTER-PANEL (EPH 2-2-2)
# =============================================================================
class PurgedWalkForwardSplitter:
    def __init__(self, df: pd.DataFrame, train_quarters: int = 8):
        self.df = df
        self.train_quarters = train_quarters
        self.periodos = (
            df[["ano4", "trimestre"]]
            .drop_duplicates()
            .sort_values(by=["ano4", "trimestre"])
            .values.tolist()
        )

    def get_folds(self, start_idx: int = 8, end_idx: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if end_idx is None:
            end_idx = len(self.periodos)

        folds = []
        for i in range(start_idx, end_idx):
            val_ano, val_trim = self.periodos[i]
            train_window = self.periodos[i - self.train_quarters : i]

            # Máscara para la ventana de entrenamiento (8 trimestres)
            train_mask = pd.Series(False, index=self.df.index)
            for a, t in train_window:
                train_mask |= (self.df["ano4"] == a) & (self.df["trimestre"] == t)

            # Máscara para la ventana de validación/test (1 trimestre)
            val_mask = (self.df["ano4"] == val_ano) & (self.df["trimestre"] == val_trim)

            # PURGA: Eliminar de train cualquier hogar presente en val
            hogares_val = set(self.df.loc[val_mask, "codusu"].unique())
            hogares_train = self.df["codusu"].isin(hogares_val)
            train_mask_purged = train_mask & (~hogares_train)

            idx_train = np.where(train_mask_purged)[0]
            idx_val = np.where(val_mask)[0]

            folds.append((idx_train, idx_val))
        return folds

# =============================================================================
# 2. EVALUACIÓN Y CALIBRACIÓN DE UMBRAL (F2-SCORE Y PR-AUC)
# =============================================================================
def calcular_metricas(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    # Búsqueda del umbral óptimo empírico que maximiza F2
    f2_scores = []
    thresholds_eval = np.linspace(0.1, 0.9, 81)
    for tau in thresholds_eval:
        preds = (y_probs >= tau).astype(int)
        f2_scores.append(fbeta_score(y_true, preds, beta=2, zero_division=0))

    best_idx = np.argmax(f2_scores)
    best_tau = float(thresholds_eval[best_idx])
    best_f2 = float(f2_scores[best_idx])
    
    preds_best = (y_probs >= best_tau).astype(int)
    recall_opt = float(np.sum((preds_best == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1))

    return {
        "pr_auc": float(pr_auc),
        "f2_score": best_f2,
        "optimal_threshold": best_tau,
        "recall_at_optimal": recall_opt
    }

# =============================================================================
# 3. RED NEURONAL TABULAR (PYTORCH)
# =============================================================================
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    
    train_ds = TabularDataset(X_tr, y_tr)
    val_ds = TabularDataset(X_va, y_va)
    
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 512), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    model = TabularMLP(
        in_features=X_tr.shape[1],
        hidden_layers=config["hidden_layers"],
        dropout=config.get("dropout", 0.2)
    ).to(DEVICE)

    # Balanceo por pérdida logarítmica ponderada
    pos_weight = torch.tensor([(len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.get("lr", 1e-3), 
        weight_decay=config.get("weight_decay", 1e-4)
    )

    epochs = config.get("epochs", 25)
    for _ in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    y_preds_list = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            x_b = x_b.to(DEVICE)
            probs = torch.sigmoid(model(x_b)).cpu().numpy()
            y_preds_list.append(probs)

    y_probs = np.vstack(y_preds_list).ravel()
    return calcular_metricas(y_va, y_probs)

# =============================================================================
# 4. PREPROCESADOR DE VARIABLES
# =============================================================================
def construir_preprocesador(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    cols_excluir = {"codusu", "nro_hogar", "linea_pobreza", "pobre", "ano4", "trimestre"}
    todas_cols = [c for c in df.columns if c not in cols_excluir]

    nominales = [c for c in ["region_estudio", "iv1", "ii7"] if c in todas_cols]
    ordinales = [c for c in ["max_educ_hogar", "jefe_nivel_ed"] if c in todas_cols]
    numericas = [c for c in todas_cols if c not in nominales and c not in ordinales]

    transformers = [
        ("nom", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), nominales),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinales),
        ("num", StandardScaler(), numericas)
    ]
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, todas_cols

# =============================================================================
# 5. WORKER UNIFICADO DE W&B SWEEPS (FASE DE DESARROLLO 2016-2021)
# =============================================================================
def train_sweep_iteration():
    run = wandb.init()
    cfg = wandb.config

    # Carga puntual de datos
    df = pd.read_parquet(PARQUET_FILE)
    
    # Período de optimización bayesiana: 2016-2021
    df_dev = df[df["ano4"] <= 2021].reset_index(drop=True)
    splitter = PurgedWalkForwardSplitter(df_dev, train_quarters=8)
    folds = splitter.get_folds()

    preprocessor, feature_names = construir_preprocesador(df_dev)
    
    pr_aucs, f2s = [], []

    for fold_idx, (idx_tr, idx_va) in enumerate(folds):
        train_sub = df_dev.iloc[idx_tr]
        val_sub = df_dev.iloc[idx_va]

        X_train_raw = train_sub[feature_names]
        y_train = train_sub["pobre"].values
        X_val_raw = val_sub[feature_names]
        y_val = val_sub["pobre"].values

        # Ajuste del preprocesamiento exclusivamente sobre el set de train
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        if cfg.model_family == "logistic":
            model = LogisticRegression(
                C=cfg.C,
                penalty=cfg.penalty,
                solver="saga",
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            metrics = calcular_metricas(y_val, probs)

        elif cfg.model_family == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                max_features=cfg.max_features,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            metrics = calcular_metricas(y_val, probs)

        elif cfg.model_family == "xgboost":
            scale_pos = (len(y_train) - np.sum(y_train)) / max(np.sum(y_train), 1)
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
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            probs = model.predict_proba(X_val)[:, 1]
            metrics = calcular_metricas(y_val, probs)

        elif cfg.model_family == "mlp":
            config_mlp = {
                "hidden_layers": cfg.hidden_layers,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "batch_size": 512,
                "epochs": 25
            }
            metrics = entrenar_evaluar_mlp(X_train, y_train, X_val, y_val, config_mlp)

        pr_aucs.append(metrics["pr_auc"])
        f2s.append(metrics["f2_score"])

    # Logueo del promedio sobre los folds del período de desarrollo
    wandb.log({
        "mean_pr_auc": np.mean(pr_aucs),
        "mean_f2_score": np.mean(f2s)
    })
    run.finish()

# =============================================================================
# 6. CONFIGURACIONES DE SWEEP POR MODELO
# =============================================================================
SWEEP_CONFIGS = {
    "logistic": {
        "method": "bayes",
        "metric": {"name": "mean_pr_auc", "goal": "maximize"},
        "parameters": {
            "model_family": {"value": "logistic"},
            "C": {"distribution": "log_uniform_values", "min": 1e-4, "max": 100.0},
            "penalty": {"values": ["l1", "l2"]}
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
# 7. EJECUCIÓN DEL PIPELINE COMPLETO
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento con EPH")
    parser.add_argument("--model", choices=["logistic", "random_forest", "xgboost", "mlp"], required=True)
    parser.add_argument("--project", default="eph-pobreza-ml", help="Nombre del proyecto en W&B")
    parser.add_argument("--count", type=int, default=30, help="Cantidad de iteraciones bayesianas")
    args = parser.parse_args()

    sweep_config = SWEEP_CONFIGS[args.model]
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"\nIniciando Sweep Bayesiano en W&B para {args.model.upper()}. Sweep ID: {sweep_id}")
    
    wandb.agent(sweep_id, function=train_sweep_iteration, count=args.count)
import xgboost as xgb
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
dtrain = xgb.DMatrix(X, label=y)

params = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0
}

bst = xgb.train(params, dtrain, num_boost_round=5)
print("¡XGBoost está entrenando en la GPU con CUDA 11!")
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import mean_squared_error, r2_score
import copy
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def grid_search_with_metrics(model, param_grid, X, y, model_name="Model", n_folds=5, random_state=42):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    grid = list(ParameterGrid(param_grid))
    results = []

    best_score = np.inf
    best_params = None
    best_model = None

    logger.info(f"\nGridSearch {model_name} — {len(grid)} param sets")

    for idx, params in enumerate(grid, 1):
        start = time.time()
        logger.info(f"\n{model_name} - set {idx}/{len(grid)}: {params}")

        fold_train_mse, fold_val_mse, fold_train_r2, fold_val_r2 = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            m = copy.deepcopy(model).set_params(**params)
            m.fit(X[train_idx], y[train_idx])

            y_train_pred = m.predict(X[train_idx])
            y_val_pred = m.predict(X[val_idx])

            train_mse = mean_squared_error(y[train_idx], y_train_pred)
            val_mse = mean_squared_error(y[val_idx], y_val_pred)
            train_r2 = r2_score(y[train_idx], y_train_pred)
            val_r2 = r2_score(y[val_idx], y_val_pred)

            fold_train_mse.append(train_mse)
            fold_val_mse.append(val_mse)
            fold_train_r2.append(train_r2)
            fold_val_r2.append(val_r2)

            logger.info(
                f"[Fold {fold}/{n_folds}] Train MSE={train_mse:.4f} | Val MSE={val_mse:.4f} "
                f"| Train R²={train_r2:.4f} | Val R²={val_r2:.4f}"
            )

        avg_train_mse, avg_val_mse = np.mean(fold_train_mse), np.mean(fold_val_mse)
        avg_train_r2, avg_val_r2 = np.mean(fold_train_r2), np.mean(fold_val_r2)
        results.append({
            "params": params,
            "train_mse": avg_train_mse,
            "val_mse": avg_val_mse,
            "train_r2": avg_train_r2,
            "val_r2": avg_val_r2,
        })

        if avg_val_mse < best_score:
            best_score = avg_val_mse
            best_params = params
            best_model = copy.deepcopy(m)

        logger.info(
            f"→ Mean CV: Train MSE={avg_train_mse:.4f} | Val MSE={avg_val_mse:.4f} | "
            f"Train R²={avg_train_r2:.4f} | Val R²={avg_val_r2:.4f}"
        )
        logger.info(f"Elapsed: {time.time() - start:.2f}s")

    df_results = pd.DataFrame(results).sort_values("val_mse")
    logger.info(f"Best {model_name} params: {best_params} | CV Val MSE={best_score:.4f}")
    return best_model, best_params, df_results

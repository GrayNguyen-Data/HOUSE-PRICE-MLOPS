import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
import copy
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model=None, n_folds=5, random_state=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_base_models = []
        logger.info(f"Initialized StackingRegressor with {len(base_models)} base models and {n_folds}-fold CV.")

    def fit(self, X, y):
        start_time = time.time()
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((n_samples, len(self.base_models)))

        # --- Train base models ---
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {type(model).__name__}")
            fold_train_mse, fold_val_mse = [], []
            fold_train_r2, fold_val_r2 = [], []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                m = copy.deepcopy(model)
                m.fit(X[train_idx], y[train_idx])

                y_train_pred = m.predict(X[train_idx])
                y_val_pred = m.predict(X[val_idx])
                oof_preds[val_idx, i] = y_val_pred

                # --- Metrics ---
                train_mse = mean_squared_error(y[train_idx], y_train_pred)
                val_mse = mean_squared_error(y[val_idx], y_val_pred)
                train_r2 = r2_score(y[train_idx], y_train_pred)
                val_r2 = r2_score(y[val_idx], y_val_pred)

                fold_train_mse.append(train_mse)
                fold_val_mse.append(val_mse)
                fold_train_r2.append(train_r2)
                fold_val_r2.append(val_r2)

                logger.info(
                    f"[Model {i+1} | Fold {fold}/{self.n_folds}] "
                    f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | "
                    f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}"
                )

            # --- Mean across folds ---
            logger.info(
                f"→ Base model {i+1} mean scores: "
                f"Train MSE={np.mean(fold_train_mse):.4f} | Val MSE={np.mean(fold_val_mse):.4f} | "
                f"Train R²={np.mean(fold_train_r2):.4f} | Val R²={np.mean(fold_val_r2):.4f}"
            )

        logger.info("OOF predictions generated. Fitting meta model...")

        # --- Fit meta model ---
        if self.meta_model is None:
            from .ridge import RidgeRegressor
            meta = RidgeRegressor(alpha=1.0)
        else:
            meta = copy.deepcopy(self.meta_model)
        meta.fit(oof_preds, y)
        self.meta_model_ = meta

        # --- Refit base models on full data ---
        self.fitted_base_models = []
        logger.info("Refitting base models on full training data...")
        for m in self.base_models:
            cloned = copy.deepcopy(m)
            cloned.fit(X, y)
            self.fitted_base_models.append(cloned)

        logger.info(f"StackingRegressor fitting complete. Total time: {time.time() - start_time:.2f}s")
        return self

    def predict(self, X):
        logger.info("Predicting with StackingRegressor...")
        X = np.asarray(X)
        if not self.fitted_base_models:
            raise RuntimeError("Call fit before predict.")
        meta_features = np.column_stack([m.predict(X) for m in self.fitted_base_models])
        preds = self.meta_model_.predict(meta_features)
        return preds

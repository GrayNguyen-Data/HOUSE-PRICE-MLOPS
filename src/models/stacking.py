import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
import copy


class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model=None, n_folds=5, random_state=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_base_models = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        oof_preds = np.zeros((n_samples, len(self.base_models)))

        # OOF predictions
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                m = copy.deepcopy(model)
                m.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx, i] = m.predict(X[val_idx])

        # Fit meta model on OOF predictions
        if self.meta_model is None:
            from .ridge import RidgeRegressor
            meta = RidgeRegressor(alpha=1.0)
        else:
            meta = copy.deepcopy(self.meta_model)

        meta.fit(oof_preds, y)
        self.meta_model_ = meta

        # Fit base models on full data
        self.fitted_base_models = []
        for m in self.base_models:
            cloned = copy.deepcopy(m)
            cloned.fit(X, y)
            self.fitted_base_models.append(cloned)

        return self

    def predict(self, X):
        X = np.asarray(X)
        if not self.fitted_base_models:
            raise RuntimeError("Call fit before predict on StackingRegressor")
        meta_features = np.column_stack([m.predict(X) for m in self.fitted_base_models])
        return self.meta_model_.predict(meta_features)

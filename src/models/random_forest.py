import numpy as np
from .tree import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, n_estimators=10, max_features="sqrt", max_depth=5, min_samples_split=2, bootstrap=True, random_state=None):
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.features_idx = []

    def _max_features_count(self, n_features):
        if isinstance(self.max_features, int):
            return self.max_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        return n_features

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.trees = []
        self.features_idx = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.randint(0, n_samples, size=n_samples)
            else:
                idx = np.arange(n_samples)

            max_feat = self._max_features_count(n_features)
            feat_idx = rng.choice(n_features, size=max_feat, replace=False)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X[idx][:, feat_idx], y[idx])
            self.trees.append(tree)
            self.features_idx.append(feat_idx)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.column_stack([
            tree.predict(X[:, feat_idx]) for tree, feat_idx in zip(self.trees, self.features_idx)
        ])
        return preds.mean(axis=1)

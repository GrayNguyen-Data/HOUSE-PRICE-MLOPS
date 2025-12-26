import numpy as np

class _XGNode:
    __slots__ = ("feature","threshold","left","right","value","gain")
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.gain = None

class XGBoostRegressor:
    """A simplified XGBoost-like regressor using second-order approximation."""
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=2, lam=1.0, gamma=0.0):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.trees = []
        self.init_pred = 0.0

    def set_params(self, **params):
        for k, v in params.items():
            if k in ("n_estimators", "max_depth", "min_samples_split"):
                setattr(self, k, int(v))
            elif k in ("learning_rate", "lam", "gamma"):
                setattr(self, k, float(v))
            else:
                setattr(self, k, v)
        return self

    def _calc_grad_hess(self, y, y_pred):
        g = y_pred - y
        h = np.ones_like(g)
        return g, h

    def _build_tree(self, X, g, h, depth=0):
        node = _XGNode()
        G = g.sum()
        H = h.sum()
        node.value = -G / (H + self.lam)
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            return node

        best_gain = 0.0
        best_feat = None
        best_thr = None
        for feat in range(X.shape[1]):
            values = np.unique(X[:, feat])
            if values.shape[0] == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                if left_mask.sum() == 0 or left_mask.sum() == X.shape[0]:
                    continue
                G_L = g[left_mask].sum(); H_L = h[left_mask].sum()
                G_R = g[~left_mask].sum(); H_R = h[~left_mask].sum()
                gain = 0.5*(G_L*G_L/(H_L + self.lam) + G_R*G_R/(H_R + self.lam) - G*G/(H + self.lam)) - self.gamma
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr

        if best_feat is None:
            return node

        node.feature = best_feat
        node.threshold = best_thr
        left_mask = X[:, best_feat] <= best_thr
        node.left = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth+1)
        node.right = self._build_tree(X[~left_mask], g[~left_mask], h[~left_mask], depth+1)
        node.gain = best_gain
        return node

    def _predict_row(self, x, node):
        if node.feature is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.init_pred = float(y.mean())
        y_pred = np.full(y.shape, self.init_pred)
        self.trees = []
        for m in range(self.n_estimators):
            g, h = self._calc_grad_hess(y, y_pred)
            tree = self._build_tree(X, g, h, depth=0)
            update = np.array([self._predict_row(row, tree) for row in X])
            y_pred = y_pred + self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.full((X.shape[0],), self.init_pred)
        for tree in self.trees:
            y_pred += self.learning_rate * np.array([self._predict_row(row, tree) for row in X])
        return y_pred
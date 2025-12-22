import numpy as np

class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        node = _TreeNode()
        # stopping
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split or np.unique(y).shape[0] == 1:
            node.value = float(np.mean(y))
            return node

        best_feat, best_thr, best_loss = None, None, float('inf')
        for feat in range(X.shape[1]):
            values = np.unique(X[:, feat])
            if values.shape[0] == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                if left_mask.sum() == 0 or left_mask.sum() == X.shape[0]:
                    continue
                y_l, y_r = y[left_mask], y[~left_mask]
                loss = (y_l.size * np.var(y_l) + y_r.size * np.var(y_r))
                if loss < best_loss:
                    best_loss = loss
                    best_feat = feat
                    best_thr = thr

        if best_feat is None:
            node.value = float(np.mean(y))
            return node

        node.feature = best_feat
        node.threshold = best_thr
        left_mask = X[:, best_feat] <= best_thr
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_row(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        preds = [self._predict_row(row, self.root) for row in X]
        return np.array(preds)

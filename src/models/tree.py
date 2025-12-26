import numpy as np

#Node của cây 
class TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self):
        self.feature = None       
        self.threshold = None     
        self.left = None         
        self.right = None        
        self.value = None      

class DecisionTreeRegressor:
    """
    Decision Tree Regression đơn giản (dùng cho Random Forest).
    Chia nhánh dựa trên MSE, threshold = trung bình các giá trị liên tiếp.
    """
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    #Xây dựng cây 
    def build_tree(self, X, y, depth):
        node = TreeNode()
        n_samples, n_features = X.shape

        #Điều kiện dừng
        if n_samples < self.min_samples_split or depth >= self.max_depth or np.unique(y).shape[0] == 1:
            node.value = float(np.mean(y))
            return node

        # Tìm feature và threshold tốt nhất
        best_feature = None
        best_threshold = None
        best_loss = float('inf')

        for feat in range(n_features):
            values = np.unique(X[:, feat])
            if values.shape[0] == 1:
                continue  # không chia được feature này

            thresholds = (values[:-1] + values[1:]) / 2.0  # trung bình các giá trị liên tiếp

            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue  # bỏ split không hợp lệ

                y_left = y[left_mask]
                y_right = y[right_mask]

                loss = y_left.size * np.var(y_left) + y_right.size * np.var(y_right)

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feat
                    best_threshold = thr

        # Nếu không tìm được feature tốt → node là leaf
        if best_feature is None:
            node.value = float(np.mean(y))
            return node

        # Gán node và build các nhánh con
        node.feature = best_feature
        node.threshold = best_threshold

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self.build_tree(X, y, depth=0)
        return self

    def predict_row(self, x, node):
        if node.value is not None:       # gặp leaf → trả giá trị
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_row(x, node.left)
        return self.predict_row(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        preds = [self.predict_row(row, self.root) for row in X]
        return np.array(preds)
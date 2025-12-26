import numpy as np

class RidgeRegressor:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def set_params(self, **params):
        for k, v in params.items():
            if k == "alpha":
                self.alpha = float(v)
            elif k == "fit_intercept":
                self.fit_intercept = bool(v)
            else:
                setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X_design = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_design = X

        I = np.eye(X_design.shape[1])
        if self.fit_intercept:
            I[0, 0] = 0.0

        A = X_design.T.dot(X_design) + self.alpha * I
        b = X_design.T.dot(y)
        w = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_
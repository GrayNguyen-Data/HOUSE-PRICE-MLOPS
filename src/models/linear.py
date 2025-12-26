import numpy as np

class LinearRegressor:
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.coef = None
        self.intercept = 0.0

    def set_params(self, **params):
        for k, v in params.items():
            if k == "fit_intercept":
                self.fit_intercept = bool(v)
            else:
                setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype =float)
        y = np.asarray(y, dtype =float)
        sum_samples = X.shape[0]
        if self.fit_intercept:
            X_new = np.hstack([np.ones((sum_samples, 1)), X])
        else:
            X_new = X

        W = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y

        if self.fit_intercept:
            self.intercept = float(W[0])
            self.coef = W[1:]
        else:
            self.intercept = 0.0
            self.coef = W

        return self

    def predict(self, X):
        X = np.asarray(X, dtype = float)
        return X @ self.coef + self.intercept
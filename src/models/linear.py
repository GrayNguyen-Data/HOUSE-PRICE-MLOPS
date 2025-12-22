import numpy as np

class LinearRegressor:
    """Ordinary Least Squares linear regression (closed-form).

    Minimizes squared error: w = (X^T X)^-1 X^T y
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]
        if self.fit_intercept:
            X_design = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_design = X

        # Solve normal equation with pseudo-inverse for stability
        w, *_ = np.linalg.lstsq(X_design, y, rcond=None)

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

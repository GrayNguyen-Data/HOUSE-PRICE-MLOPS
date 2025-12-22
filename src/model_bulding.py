from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
from models.stacking import StackingRegressor
from models.ridge import RidgeRegressor
from models.random_forest import RandomForestRegressor
from models.xgboost import XGBoostRegressor
from models.lightgbm import LightGBMRegressor
from models.linear import LinearRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuildingStrategy(ABC):
     
    """RegressorMixin: Cho biết đây là mô hình hồi quy -> và nó hỗ trợ các đánh giá score"""
    @abstractmethod
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        pass

class LinearRegressionStratery(ModelBuildingStrategy):
    
    def build_train_model(self, X_train, y_train) -> Pipeline:
        
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train không phải là dataframe")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train không phải là dạng series")
        
        logging.info("Khởi tạo mô hình hồi quy tuyến tính  và chuẩn hóa")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)

        logging.info("Hoàn thành việc training model.")
        return pipeline

class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Chuyển đổi lựa chọn mô hình khác.")
        self._strategy = strategy
    
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        logging.info("Build và training với mô hình đã chọn.")
        return self._strategy.build_train_model(X_train, y_train)


class StackingStrategy(ModelBuildingStrategy):
    def __init__(self, base_models=None, meta_model=None, n_folds=5):
        if base_models is None:
            base_models = [
                RidgeRegressor(alpha=1.0),
                XGBoostRegressor(n_estimators=100, learning_rate=0.05, max_depth=4),
                LightGBMRegressor(n_estimators=100, learning_rate=0.05, max_leaves=31),
                RandomForestRegressor(n_estimators=100, max_depth=8)
            ]
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else LinearRegressor(fit_intercept=True)
        self.n_folds = n_folds

    def build_train_model(self, X_train, y_train):
        logging.info("Khởi tạo Stacking với base models: Ridge, XGBoost, LightGBM, RandomForest")
        X = X_train.values if hasattr(X_train, 'values') else X_train
        y = y_train.values if hasattr(y_train, 'values') else y_train
        stack = StackingRegressor(base_models=self.base_models, meta_model=self.meta_model, n_folds=self.n_folds)
        stack.fit(X, y)
        logging.info("Hoàn tất training Stacking model.")
        return stack

if __name__ == "__main__":
    pass

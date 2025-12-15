from abc import ABC, abstractmethod
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EvaluatorModelStrategy(ABC):
    @abstractmethod
    def evaluator(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        pass

class RegressionEvaluatorModel(EvaluatorModelStrategy):
    def evaluator(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info("Sử dụng mô hình đã chọn làm dự đoán.")
        y_pred = model.predict(X_test)

        logging.info('Tính toán các metrics dự đoán.')
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Mean Squared Error": mse, "R-Squared": r2}
        logging.info(f"Các chỉ số đánh giá: {metrics}")
        return metrics

class EvaluatorModel:
    def __init__(self, strategy: EvaluatorModelStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: EvaluatorModelStrategy):
        logging.info("Đổi sự lựa chọn mô hình.")
        self._strategy = strategy
    
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict: 
        logging.info("Đánh giá mô hình với phương pháp đã chọn.")
        return self._strategy.evaluator(model, X_test, y_test)  # ✅ ĐÃ THÊM RETURN

if __name__ == "__main__":
    pass
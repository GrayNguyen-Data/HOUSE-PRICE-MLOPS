import logging
from typing import Tuple

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from zenml import step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, float, float]:
    """
    Đánh giá mô hình hồi quy đã huấn luyện.

    Trả về:
    dict: Dictionary chứa các metric đánh giá (MSE, R2)
    float: Mean Squared Error
    float: R² score
    """

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test phải là pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test phải là pandas Series.")

    logging.info("Thực hiện dự đoán trên dữ liệu test.")
    y_pred = trained_model.predict(X_test)

    # Tính metric
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evaluation_metrics = {"Mean Squared Error": mse, "R2 Score": r2}

    logging.info(f"Đánh giá hoàn tất. MSE = {mse}, R² = {r2}")

    return evaluation_metrics, mse, r2

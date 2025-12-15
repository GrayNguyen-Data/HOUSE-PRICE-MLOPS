import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Lấy experiment tracker hiện tại từ ZenML
experiment_tracker = Client().active_stack.experiment_tracker

# Định nghĩa model ZenML artifact
model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Mô hình dự đoán giá nhà"
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:

    # Kiểm tra kiểu dữ liệu đầu vào
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train phải là pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train phải là pandas Series.")

    logging.info(f"Huấn luyện mô hình với các cột: {X_train.columns.tolist()}")

    # Xây dựng pipeline: chuẩn hóa numeric (nếu muốn) + LinearRegression
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # Chuẩn hóa dữ liệu numeric
            ("model", LinearRegression())
        ]
    )

    # Bắt đầu MLflow run nếu chưa có run đang active
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Kích hoạt tự động logging cho scikit-learn
        mlflow.sklearn.autolog()

        logging.info("Đang huấn luyện mô hình Hồi quy tuyến tính...")
        pipeline.fit(X_train, y_train)
        logging.info("Hoàn tất huấn luyện mô hình.")

    except Exception as e:
        logging.error(f"Lỗi trong quá trình huấn luyện mô hình: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline

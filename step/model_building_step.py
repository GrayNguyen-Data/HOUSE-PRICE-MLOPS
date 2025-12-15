from typing import Annotated
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zenml import ArtifactConfig, step, Model

# Định nghĩa model ZenML artifact
model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Mô hình dự đoán giá nhà"
)

@step(enable_cache=False, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:

    # Kiểm tra kiểu dữ liệu đầu vào
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train phải là pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train phải là pandas Series.")

    # Xây dựng pipeline: chuẩn hóa numeric + LinearRegression
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # Chuẩn hóa dữ liệu numeric
            ("model", LinearRegression())
        ]
    )

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    return pipeline

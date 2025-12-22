from typing import Annotated
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zenml import step, Model
import logging
from models.stacking import StackingRegressor
from models.ridge import RidgeRegressor
from models.xgboost import XGBoostRegressor
from models.lightgbm import LightGBMRegressor
from models.random_forest import RandomForestRegressor
from models.linear import LinearRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Mô hình dự đoán giá nhà"
)

@step(enable_cache=False, model=model)
def model_building_step(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.DataFrame, "y_train"]
) -> Annotated[Pipeline, "sklearn_pipeline"]:
    """Xây dựng và train mô hình Linear Regression."""
    logging.info("=" * 80)
    logging.info("BẮT ĐẦU MODEL BUILDING STEP")
    logging.info("=" * 80)
    
    logging.info(f"[INPUT] X_train - Shape: {X_train.shape}, Type: {type(X_train).__name__}")
    if hasattr(X_train, 'columns'):
        logging.info(f"        First 5 columns: {X_train.columns.tolist()[:5]}")
    
    logging.info(f"[INPUT] y_train - Shape: {y_train.shape}, Type: {type(y_train).__name__}")
    if hasattr(y_train, 'columns'):
        logging.info(f"        Columns: {y_train.columns.tolist()}")
    
    logging.info("=" * 80)

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train phải là DataFrame, nhận được {type(X_train)}")
    if not isinstance(y_train, pd.DataFrame):
        raise TypeError(f"y_train phải là DataFrame, nhận được {type(y_train)}")

    if y_train.shape[1] != 1:
        raise ValueError(
            f"   y_train phải có đúng 1 cột, nhận được {y_train.shape[1]} cột.\n"
            f"   Columns: {y_train.columns.tolist()}\n"
            f"   Có thể ZenML đã load nhầm artifact!"
        )

    y_train_series = y_train.iloc[:, 0]
    logging.info(f"Converted y_train to Series: {y_train_series.shape}")
    # Inspect features passed to model for debugging
    try:
        cols = X_train.columns.tolist()
        dtypes = X_train.dtypes.to_dict()
        logging.info(f"[DEBUG] X_train columns ({len(cols)}): {cols[:50]}")
        logging.info(f"[DEBUG] X_train dtypes sample: {{k: str(v) for k,v in list(dtypes.items())[:20]}}")
    except Exception:
        logging.info("[DEBUG] X_train is not a DataFrame or has no columns attribute.")

    # Xây dựng pipeline
    # Use custom stacking as final estimator (scale features first)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                StackingRegressor(
                    base_models=[
                        RidgeRegressor(alpha=1.0),
                        XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3),
                        LightGBMRegressor(n_estimators=50, learning_rate=0.1, max_leaves=31),
                        RandomForestRegressor(n_estimators=50, max_depth=6),
                    ],
                    meta_model=LinearRegressor(fit_intercept=True),
                    n_folds=5,
                ),
            ),
        ]
    )

    # Huấn luyện mô hình
    logging.info("Bắt đầu train mô hình...")
    pipeline.fit(X_train, y_train_series)
    logging.info("Hoàn tất train mô hình")
    logging.info("=" * 80)
    
    return pipeline
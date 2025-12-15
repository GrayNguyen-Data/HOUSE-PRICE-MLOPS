
from zenml import step, Model
from zenml.steps import get_step_context
from typing import Annotated, Dict
import logging
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

model_object = Model(
    name="prices_predictor",
    description="House price prediction model for HCM City",
)

@step(
    enable_cache=False,
    model=model_object,
)
def model_evaluator_step(
    trained_model: Annotated[Pipeline, "trained_model"],
    X_test: Annotated[pd.DataFrame, "X_test"],
    y_test: Annotated[pd.DataFrame, "y_test"],
) -> Annotated[Dict[str, float], "evaluation_metrics"]:

    logging.info("BẮT ĐẦU MODEL EVALUATION STEP")

    # 1. Ensure y_test is Series
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
        logging.info("Converted y_test DataFrame -> Series")

    # 2. Predict
    y_pred = trained_model.predict(X_test)

    # 3. Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"✅ Evaluation finished | MSE={mse:.4f} | R²={r2:.4f}")

    # 4. STEP CONTEXT
    step_context = get_step_context()

    # LOG METADATA → OUTPUT ARTIFACT (DASHBOARD HIỆN)
    step_context.add_output_metadata(
        output_name="evaluation_metrics",
        metadata={
            "mse": float(mse),
            "r2": float(r2),
            "num_features": int(X_test.shape[1]),
            "num_test_samples": int(X_test.shape[0]),
        },
    )

    logging.info("Metrics đã được lưu vào OUTPUT artifact metadata")

    # 5. MODEL VERSION METADATA
    model_version = step_context.model

    model_version.log_metadata(
        {
            "mse": float(mse),
            "r2": float(r2),
            "num_features": int(X_test.shape[1]),
            "num_test_samples": int(X_test.shape[0]),
            "r2_threshold": 0.85,
        }
    )

    logging.info("Metrics đã được lưu vào Model Version metadata")

    # 6. Promotion logic
    if r2 >= 0.85:
        model_version.set_stage("production", force=True)
        model_version.log_metadata({"promoted_to_production": True})

        logging.info(
            f"🚀 Model version {model_version.version} PROMOTED to PRODUCTION"
        )
    else:
        model_version.log_metadata({"promoted_to_production": False})
        logging.warning("Model NOT promoted")

    logging.info("KẾT THÚC MODEL EVALUATION STEP")

    # 7. Return artifact
    return {
        "mse": float(mse),
        "r2": float(r2),
    }

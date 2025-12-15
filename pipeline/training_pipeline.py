
from step.data_ingestion_step import data_ingestion_step
from step.data_splitter_step import data_splitter_step
from step.feature_engineering_step import feature_engineering_step
from step.handle_missing_value_step import handle_missing_values_step
from step.model_building_step import model_building_step
from step.evaluator_model_step import model_evaluator_step
from step.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline, step


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="/Users/ayushsingh/Desktop/end-to-end-production-grade-projects/prices-predictor-system/data/archive.zip"
    )

    # Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="SalePrice")

    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()

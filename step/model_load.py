from sklearn.pipeline import Pipeline
from zenml import Model, step
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step
def model_loader(model_name: str) -> Pipeline:
    logging.info(f"Đang load mô hình production: {model_name}")

    # Load model ZenML theo tên và version production
    model = Model(name=model_name, version="production")

    # Load artifact pipeline đã lưu (tên artifact: "sklearn_pipeline")
    model_pipeline: Pipeline = model.load_artifact("sklearn_pipeline")

    logging.info(f"Mô hình {model_name} đã được load thành công.")

    return model_pipeline

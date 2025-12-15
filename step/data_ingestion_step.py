from zenml import step
from typing import Annotated
import pandas as pd
from src.data_ingestion import DataIngestorFactory

@step
def data_ingestion_step(file_path: str) -> Annotated[pd.DataFrame, "raw_data"]:
    """Đọc dữ liệu từ file zip."""
    file_extension = ".zip"
    data_ingestion = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestion.ingest(file_path)
    return df
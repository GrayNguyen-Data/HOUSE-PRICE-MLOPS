import pandas as pd
from src.data_ingestion import DataIngestorFactory

def data_ingestion_step(file_path: str) -> pd.DataFrame:
    file_extension = ".zip"
    data_ingestion = DataIngestorFactory.get_data_ingestor(file_extension)

    df = data_ingestion.ingest(file_path)
    return df
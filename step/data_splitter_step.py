from typing import Tuple, Annotated
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=True)
def data_splitter_step(
    df: Annotated[pd.DataFrame, "transformed_data"],
    target_column: str
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_test"]
]:
    """Chia dữ liệu thành train và test sets.
    
    Returns:
        Tuple theo thứ tự: X_train, y_train, X_test, y_test
    """
    logging.info("=" * 80)
    logging.info("BẮT ĐẦU DATA SPLITTER STEP")
    logging.info("=" * 80)
    
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, y_train, X_test, y_test = splitter.split(df, target_column)
    
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame(name=target_column)
        logging.info("Đổi y_train thành DataFrame")
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame(name=target_column)
        logging.info("Đổi y_test thành DataFrame")
    
    logging.info("-------------------------------------------------------------------")
    logging.info(f"  [0] X_train: {X_train.shape} - {X_train.columns[:3].tolist()}...")
    logging.info(f"  [1] y_train: {y_train.shape} - {y_train.columns.tolist()}")
    logging.info(f"  [2] X_test:  {X_test.shape} - {X_test.columns[:3].tolist()}...")
    logging.info(f"  [3] y_test:  {y_test.shape} - {y_test.columns.tolist()}")

    assert y_train.shape[1] == 1, f"y_train phải có 1 cột, có {y_train.shape[1]}"
    assert y_test.shape[1] == 1, f"y_test phải có 1 cột, có {y_test.shape[1]}"
    assert y_train.columns[0] == target_column, f"y_train column phải là {target_column}"
    assert y_test.columns[0] == target_column, f"y_test column phải là {target_column}"
    
    logging.info("Hoàn thành")
    logging.info("=" * 80)

    return X_train, y_train, X_test, y_test
import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    logging.info(f"Bắt đầu bước phát hiện outlier, DataFrame có shape: {df.shape}")

    if df is None:
        logging.error("Nhận được DataFrame là None.")
        raise ValueError("Input df phải là một pandas DataFrame hợp lệ, không được None.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Loại dữ liệu không hợp lệ: {type(df)}")
        raise ValueError("Input df phải là pandas DataFrame.")

    if column_name not in df.columns:
        logging.error(f"Cột '{column_name}' không tồn tại trong DataFrame.")
        raise ValueError(f"Cột '{column_name}' không tồn tại trong DataFrame.")

    # Chỉ sử dụng các cột numeric
    df_numeric = df.select_dtypes(include=[int, float])

    # Khởi tạo detector với Z-score threshold = 3
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # Phát hiện outlier
    outliers = outlier_detector.detected_outlier(df_numeric)

    # Loại bỏ outlier
    df_cleaned = outlier_detector.handle_outlier(df_numeric, method="remove")

    logging.info(f"Bước phát hiện outlier hoàn tất. Số hàng sau khi loại bỏ: {df_cleaned.shape[0]}")

    return df_cleaned

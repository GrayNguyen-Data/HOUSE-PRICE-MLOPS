# from typing import Annotated
# import logging
# import pandas as pd
# from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
# from zenml import step

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# @step(enable_cache=False)  # ✅ Disable cache
# def outlier_detection_step(
#     df: Annotated[pd.DataFrame, "clean_data"],
#     column_name: str = ""
# ) -> Annotated[pd.DataFrame, "outlier_removed_data"]:
#     """Phát hiện và loại bỏ outliers."""
#     logging.info(f"Bắt đầu bước phát hiện outlier, DataFrame shape: {df.shape}")
#     logging.info(f"Columns: {df.columns.tolist()}")

#     if df is None or not isinstance(df, pd.DataFrame):
#         raise ValueError("Input df phải là pandas DataFrame hợp lệ.")

#     # ✅ Lưu tất cả columns ban đầu
#     original_columns = df.columns.tolist()
    
#     # Lấy numeric columns
#     numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
#     non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    
#     logging.info(f"Numeric columns: {numeric_cols}")
#     logging.info(f"Non-numeric columns: {non_numeric_cols}")

#     if not numeric_cols:
#         logging.warning("Không có cột numeric nào để xử lý outlier.")
#         return df

#     # Chỉ xử lý outlier trên numeric columns
#     df_numeric = df[numeric_cols]
    
#     outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
#     df_cleaned = outlier_detector.handle_outlier(df_numeric, method="remove")
    
#     # ✅ Ghép lại non-numeric columns nếu có
#     if non_numeric_cols:
#         remaining_indices = df_cleaned.index
#         df_non_numeric = df.loc[remaining_indices, non_numeric_cols]
#         df_cleaned = pd.concat([df_cleaned, df_non_numeric], axis=1)
        
#         # ✅ Đảm bảo thứ tự columns giống ban đầu
#         df_cleaned = df_cleaned[original_columns]

#     logging.info(f"✅ Outlier removed. Remaining rows: {df_cleaned.shape[0]}")
#     logging.info(f"Final columns: {df_cleaned.columns.tolist()}")
    
#     return df_cleaned

from typing import Annotated
import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def outlier_detection_step(
    df: Annotated[pd.DataFrame, "clean_data"],
) -> Annotated[pd.DataFrame, "outlier_removed_data"]:
    """Phát hiện và loại bỏ outliers."""
    logging.info(f"Bắt đầu bước phát hiện outlier, DataFrame shape: {df.shape}")
    
    # 1. Xác định các cột liên tục (Continuous Features)
    # Loại bỏ các cột OHE/Binary (chỉ chứa 0 và 1) bằng cách chọn các cột có số giá trị unique > 2
    
    continuous_cols = []
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].nunique() > 2:
            continuous_cols.append(col)
    
    logging.info(f"✅ Auto-selected {len(continuous_cols)} columns for Outlier Detection.")

    if not continuous_cols:
        logging.warning("Không có cột continuous nào để xử lý outlier. Bỏ qua bước loại bỏ outlier.")
        return df

    # 2. Áp dụng Outlier Detection chỉ trên các cột Continuous
    df_continuous = df[continuous_cols]
    
    # Sử dụng ZScoreOutlierDetection (threshold=3)
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    
    # Lấy mask outliers (True nếu là outlier)
    # df_continuous.shape: (2930, X)
    outliers_mask = outlier_detector.detected_outlier(df_continuous) 
    
    # Chỉ loại bỏ hàng nếu nó là outlier TRONG BẤT KỲ cột continuous nào
    outliers_to_remove = outliers_mask.any(axis=1) 
    
    # Loại bỏ outliers khỏi DataFrame GỐC
    df_cleaned = df[~outliers_to_remove]
    
    logging.info(f"✅ Outlier removed. Original rows: {df.shape[0]}, Remaining rows: {df_cleaned.shape[0]}")
    
    return df_cleaned
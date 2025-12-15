# # File: handle_missing_value_step.py

# from typing import Annotated, Optional # <-- THÊM Optional
# import pandas as pd
# from src.handle_missing_values import (
#     DropMissingValueStrategy,
#     FillMissingValuesStrategy,
#     MissingValueHandler,
# )
# from zenml import step

# @step
# def handle_missing_values_step(
#     df: Annotated[pd.DataFrame, "raw_data"],
#     strategy: str = "mean",
#     fill_value: Optional[str] = None # ✅ SỬA DỤNG Optional[str]
# ) -> Annotated[pd.DataFrame, "clean_data"]:
#     """Xử lý các giá trị thiếu."""
#     if strategy == "drop":
#         handler = MissingValueHandler(DropMissingValueStrategy(axis=0))
#     elif strategy in ["mean", "median", "mode"]:
#         handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
#     elif strategy == "constant":
#         # Truyền fill_value
#         handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy, fill_value=fill_value))
#     else:
#         raise ValueError(f"Phương pháp không được hỗ trợ: {strategy}")

#     cleaned_df = handler.handle_missing_value(df)
#     return cleaned_df

from typing import Annotated, Optional # <-- THÊM Optional
import pandas as pd
from src.handle_missing_values import (
    DropMissingValueStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from zenml import step

@step
def handle_missing_values_step(
    df: Annotated[pd.DataFrame, "raw_data"],
    strategy: str = "mean",
    fill_value: Optional[str] = None 
) -> Annotated[pd.DataFrame, "clean_data"]:
    """Xử lý các giá trị thiếu."""
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValueStrategy(axis=0))
    elif strategy in ["mean", "median", "mode"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    elif strategy == "constant":
        # Truyền fill_value
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy, fill_value=fill_value))
    else:
        raise ValueError(f"Phương pháp không được hỗ trợ: {strategy}")

    cleaned_df = handler.handle_missing_value(df)
    return cleaned_df
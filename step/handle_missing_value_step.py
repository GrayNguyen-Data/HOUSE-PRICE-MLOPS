import pandas as pd
from src.handle_missing_values import (
    DropMissingValueStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from zenml import step

@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Xử lý các giá trị thiếu bằng MissingValueHandler và chiến lược đã chỉ định."""
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValueStrategy(axis=0))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Phương pháp không được hỗ trợ: {strategy}")

    cleaned_df = handler.handle_missing_value(df)
    return cleaned_df

from zenml import step
from typing import Annotated, Optional # ✅ THÊM Optional
import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)

@step
def feature_engineering_step(
    df: Annotated[pd.DataFrame, "outlier_removed_data"],
    strategy: str = "log",
    features: Optional[list] = None 
) -> Annotated[pd.DataFrame, "transformed_data"]:
    """Áp dụng feature engineering."""
    features_list = features if features is not None else [] 

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features_list))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaling(features_list))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features_list))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features_list))
    else:
        raise ValueError(f"Phương pháp không được hỗ trợ: {strategy}")

    transformed_df = engineer.apply_Transform(df)
    return transformed_df

from  abc import ABC, abstractmethod

import pandas as pd
import numpy as np 
import logging

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

logging.basicConfig(level = logging.INFO, format ="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        self._features = features
    
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Áp dụng kĩ thuật lấy log cho features")

        df_transformed = df.copy()
        for feature in self._features:
            # Kiểm tra feature có tồn tại và không phải object
            if feature in df_transformed.columns and df_transformed[feature].dtype != object:
                df_transformed[feature] = np.log1p(df[feature])
            else:
                logging.warning(f"Cột '{feature}' không tồn tại hoặc không phải kiểu số để áp dụng Log Transformation.")

        
        logging.info("Đã hoàn thành việc lấy log feature.")

        return df_transformed

class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        self._features = features
        self.scaler = StandardScaler()
    
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Áp dụng kĩ thuật StandardScaling cho features.")
        df_transformed = df.copy()
        features_to_scale = [f for f in self._features if f in df_transformed.columns and df_transformed[f].dtype != object]
        
        if features_to_scale:
            df_transformed[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
        else:
            logging.warning("Không có cột nào hợp lệ để áp dụng Standard Scaling.")
            
        logging.info("Đã hoàn thành việc scale bằng phương pháp StandardScaling cho feature.")
        return df_transformed

class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list, feature_range=(0,1)):
        self._features = features
        self.scaler =MinMaxScaler(feature_range=feature_range)
    
    def transformation(self, df:pd.DataFrame) ->pd.DataFrame:
        logging.info("Áp dụng kĩ thuật MinMaxScaling cho các features.")
        df_transformed = df.copy()
        features_to_scale = [f for f in self._features if f in df_transformed.columns and df_transformed[f].dtype != object]
        
        if features_to_scale:
            df_transformed[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
        else:
            logging.warning("Không có cột nào hợp lệ để áp dụng MinMax Scaling.")

        logging.info("Đã hoàn thành việc sacle bằng phương pháp MinMaxScaling cho feature.")
        return df_transformed

class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        """ 
            - spares = false -> trả về một mảng numpy
            - drop = 'first' -> xóa đi cột đầu tiên, nhằm mục đích tránh overfiting bởi vì cột đầu tiên = 1 -(tất cả các cột còn lại) -> có mối quan hệ mật thiết.
        """
        self._features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop = 'first', handle_unknown='ignore') 
    
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Áp dụng kĩ thuật OneHotEncoding cho các features.")
        df_transformed = df.copy()
        
        # ✅ LOGIC MỚI: Tự động phát hiện các cột object nếu self._features là rỗng
        if not self._features:
            categorical_cols = df_transformed.select_dtypes(include=['object']).columns.tolist()
            self._features = categorical_cols
            logging.info(f"✅ Tự động phát hiện {len(self._features)} cột object để OHE.")
        
        ohe_cols = [col for col in self._features if col in df_transformed.columns and df_transformed[col].dtype == object]

        if not ohe_cols:
             logging.warning("Không tìm thấy cột object/categorical nào hợp lệ để áp dụng OneHotEncoding.")
             return df_transformed
        
        # 2. Thực hiện OHE
        original_index = df_transformed.index
        
        # Fit/Transform chỉ trên các cột chuỗi
        transformed_matrix = self.encoder.fit_transform(df_transformed[ohe_cols])

        # Tạo DataFrame mới từ kết quả OHE với index GỐC
        encoder_df = pd.DataFrame(
            transformed_matrix, 
            columns = self.encoder.get_feature_names_out(ohe_cols),
            index = original_index
        )
        
        # 3. Ghép lại
        df_transformed = df_transformed.drop(columns=ohe_cols)
        df_transformed = pd.concat([df_transformed, encoder_df], axis = 1)
        
        logging.info("Đã hoàn thành việc scale bằng phương pháp OneHotEncoding cho features.")
        logging.info(f"DataFrame mới có {df_transformed.shape[1]} cột.")
        return df_transformed
    
class FeatureEngineer:
    def __init__(self, stratery: FeatureEngineeringStrategy):
        self._stratery = stratery
    
    def set_stratery(self, stratery: FeatureEngineeringStrategy):
        logging.info("Chuyển đổi chiến lược Feature Engineering")
        self._stratery = stratery
    
    def apply_Transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Bắt đầu áp dụng Feature Transformation.")
        return self._stratery.transformation(df)

if __name__ == "__main__":
    pass
from abc import ABC, abstractmethod
import pandas as pd
import logging

"""Thiết lập thông báo lỗi"""
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValueHandlingStrategy(ABC): 
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DropMissingValueStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        """ 
        - axis = 0 -> xóa hàng bị thiếu
        - axis = 1 -> xóa cột bị thiếu
        - thresh: int -> số lượng giá trị không bị NA tối thiểu để row/column được giữ lại.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Đã xóa các giá trị bị thiếu với axis={self.axis} và thresh={self.thresh}")
        df_clean = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Các giá trị bị thiếu đã được xóa.")
        return df_clean
    
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        self.method = method
        self.fill_value = fill_value
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Điền các giá trị thiếu với method={self.method}")
        df_cleaned = df.copy()
        
        # Chỉ xử lý cột số cho mean, median, mode
        numeric_columns = df_cleaned.select_dtypes(include="number").columns
        
        if self.method == "mean":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
        
        elif self.method == "median":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].median())
        
        elif self.method == "mode":  # mode: tần xuất xuất hiện
            for col in numeric_columns:
                if df_cleaned[col].isnull().any():
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
        
        elif self.method == 'constant':
            # Xử lý điền hằng số cho TẤT CẢ các cột còn lại
            if self.fill_value is None:
                logging.warning("Sử dụng strategy='constant' nhưng 'fill_value' là None. Không có gì được điền.")
            else:
                df_cleaned = df_cleaned.fillna(self.fill_value)
        
        else:
            logging.warning(f"Method '{self.method}' không được hỗ trợ.")
        
        logging.info("Giá trị bị thiếu đã được xử lý.")
        return df_cleaned

class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info("Chiến lược chọn phương pháp xử lý dữ liệu thiếu")
        self._strategy = strategy
    
    def handle_missing_value(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Thực thi chiến lược xử lý dữ liệu")
        return self._strategy.handle(df)

if __name__ == "__main__":
    pass
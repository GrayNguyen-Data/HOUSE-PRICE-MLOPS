from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detected_outlier(self, df: pd.DataFrame) -> pd.DataFrame: 
        pass

"""Phát hiện outlier bằng phương pháp ZScore"""
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self._threshold = threshold
    
    def detected_outlier(self, df):
        logging.info("Phát hiện outlier bằng phương pháp Zscore")
        zscore = np.abs((df - df.mean()) / df.std())
        outlier = zscore > self._threshold
        logging.info(f"Hoàn tất việc tìm kiếm outlier với threshold={self._threshold}")
        return outlier

class IQROutlierDetection(OutlierDetectionStrategy):
    def detected_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Phát hiện outlier bằng phương pháp IQR")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outlier = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Hoàn tất việc tìm kiếm outlier với phương pháp IQR")
        return outlier

class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Chọn phương pháp xử lý outlier")
        self._strategy = strategy
    
    def detected_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Thực thi phương pháp xử lý outlier đã chọn")
        return self._strategy.detected_outlier(df)  # ✅ Fixed method name
    
    def handle_outlier(self, df: pd.DataFrame, method="remove", **kwargs) -> pd.DataFrame:
        outliers = self.detected_outlier(df)
        if method == 'remove':
            logging.info("Xóa các outlier của dataset")
            df_clean = df[(~outliers).all(axis=1)]
        elif method == 'cap':
            logging.info("Giới hạn outlier trong dataset")
            """
                - lower: giá trị thấp nhất cho phép
                - upper: giá trị cao nhất cho phép
            """
            df_clean = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.info("Không áp dụng method nào và không có outlier nào được xử lý.")
            return df

        logging.info("Đã xử lý được outlier")
        return df_clean

    def visualize_outlier(self, df: pd.DataFrame, features: list):
        logging.info(f"Vẽ biểu đồ để xem các outlier của các cột {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        
        logging.info("Đã vẽ xong các biểu đồ boxplot cho các feature.")

if __name__ == "__main__":
    pass
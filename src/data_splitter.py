from abc import ABC, abstractmethod
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, target_column: str):
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, df: pd.DataFrame, target_column: str):
        logging.info("Thực hiện chia dữ liệu train và test.")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X phải là DataFrame, nhận được {type(X)}")
        if not isinstance(y, pd.Series):
            if isinstance(y, pd.DataFrame):
                logging.warning("y đang là DataFrame, converting sang Series...")
                y = y.squeeze() 
        
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"X type: {type(X)}, y type: {type(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.squeeze()
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.squeeze()
        
        logging.info(f"Train set - X: {X_train.shape}, y: {y_train.shape}")
        logging.info(f"Test set - X: {X_test.shape}, y: {y_test.shape}")
        logging.info("Đã chia xong tập train và test.")
        
        return X_train, y_train, X_test, y_test

class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataSplittingStrategy):
        logging.info("Chuyển đổi phương pháp chia dữ liệu")
        self.strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        logging.info("Chia dữ liệu theo method đã chọn")
        return self.strategy.split(df, target_column)

if __name__ == "__main__":
    pass
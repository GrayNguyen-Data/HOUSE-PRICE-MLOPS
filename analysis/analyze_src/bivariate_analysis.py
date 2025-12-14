from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        pass

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """Dùng biểu đồ scatter plot để thể hiện mối quan hệ giữa 2 biến số"""
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """Sử dụng biểu đồ box plot"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()

class BivariateAnalyer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self.__strategy = strategy
    
    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self.__strategy = strategy
    def excute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        self.__strategy.analyze(df, feature1, feature2)

if __name__ == "__main__":
    pass
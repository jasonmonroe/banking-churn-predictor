# src/model_perf




# Vendor Libraries
import pandas as pd
from tensorflow.keras import Sequential

# Local Libraries
from src.eda import model_performance_classification
from src.utils import show_banner





class ModelPerformance:
    def __init__(self):
        self._title = ""
        self.data = None

    def get(self, title: str, model: Sequential, x_data: pd.DataFrame, y_data: pd.Series) -> None:
        # Get Model Performance
        self._title = title
        self.data = model_performance_classification(model, x_data, y_data)

    def show(self) -> None:
        if self.data is not None:
            show_banner(f"{self._title} Model Performance", [self.data])
        else:
            ValueError("🚩Error: Model Performance has not been run.")

    @staticmethod
    def final_results(train_matrix, val_matrix) -> float | None:
        if train_matrix != val_matrix:
            return train_matrix.loc["F1"] - val_matrix.loc["F1"]
        else:
            ValueError("🚩Error: Training and Validation Performances are identical!")
            return None

    @staticmethod
    def create_comparisons(models: list) -> list:
        # ❗Treat this like a private method
        def _get_matrices(titles: list, perfs: pd.DataFrame) -> pd.DataFrame:
            """
            Creates a performance matrix
            :param titles:
            :param perfs:
            :return:
            """
            perfs_matrix = perfs
            perfs_matrix.index = titles
            perfs_matrix = perfs_matrix.T

            return perfs_matrix

        titles, train_perfs, val_perfs = [], [], []
        train_perfs_matrix, val_perfs_matrix = None, None


        #comparison_titles = []
        #comparison_train_perfs = []
        #comparison_val_perfs = []
        #comparisons = []

        for model in models:
            titles.append(model.title)
            train_perfs.append(model.train_perf)
            val_perfs.append(model.val_perfs)


        # After loading all model comparisons lets get the matrices

        train_perfs_matrix = _get_matrices(titles, train_perfs)
        val_perfs_matrix = _get_matrices(titles, val_perfs)

        return train_perfs_matrix, val_perfs_matrix


    # @TODO - delete
    @staticmethod
    def create_comparison_titles(models: list) -> list:
        titles = []
        for model in models:
            titles.append(model.title)

        return titles

    @staticmethod
    def create_matrix(titles: list, perfs: pd.DataFrame) -> pd.DataFrame:
        perfs_matrix = perfs
        perfs_matrix.index = titles
        perfs_matrix = perfs_matrix.T

        return perfs_matrix
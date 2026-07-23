# src/model_perf

# Vendor Libraries
import pandas as pd
from src.constants import OVERFITTING_THRESHOLD
from tensorflow.keras import Sequential

# Local Libraries
from src.eda import model_performance_classification
from src.utils import show_banner, format_performance


class ModelPerformance:
    def __init__(self):
        self._title = ""
        self.data = None

    def get(self, title: str, model: Sequential, x_data: pd.DataFrame, y_data: pd.Series) -> pd.DataFrame:
        """
        Get Model Performance based on x, y data.  Sets data attribute for subsequent usage.
        :param title: Title for the performance run
        :param model: The Keras Sequential model
        :param x_data: Features for evaluation
        :param y_data: Ground truth labels
        :return: DataFrame of classification metrics
        """

        self._title = title
        self.data = model_performance_classification(model, x_data, y_data)
        return self.data

    def show(self) -> None:
        if self.data is not None:
            subtitles = format_performance(self.data)
            show_banner(f"{self._title} Model Performance", subtitles)
        else:
            raise ValueError("🚩Error: Model Performance has not been run.")

    """
    +---------------------------------------------------------------------------+
    |                               STATIC METHODS                               |
    +---------------------------------------------------------------------------+
    """

    @staticmethod
    def create_comparisons(models: list) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates multiple lists of comparisons (training, validation, titles, and then converts to matrices.

        :param models:
        :return:
        """

        # ❗Treat this like a private method
        def _get_matrices(model_titles: list, perfs: list) -> pd.DataFrame:
            """
            Creates a performance matrix
            :param model_titles:
            :param perfs:
            :return:
            """

            # Convert list to a single DataFrame
            perfs_matrix = pd.concat(perfs)
            perfs_matrix.index = model_titles
            perfs_matrix = perfs_matrix.T

            return perfs_matrix

        # Define comparison variables
        titles, train_perfs, val_perfs = [], [], []

        # Build model list for both performances
        for model in models:
            titles.append(model.title)
            train_perfs.append(model.train_perf)
            val_perfs.append(model.val_perf)

        # After loading all model comparisons lets get the matrices
        train_perfs_matrix = _get_matrices(titles, train_perfs)
        val_perfs_matrix = _get_matrices(titles, val_perfs)

        return train_perfs_matrix, val_perfs_matrix

    @staticmethod
    def get_best_model_name(validation_perf_matrix: pd.DataFrame, train_perf_matrix: pd.DataFrame | None = None) -> str:
        """
        Identifies the best model based strictly on validation performance,
        with optional overfitting checks against training performance[cite: 4].

        :param validation_perf_matrix:
        :param train_perf_matrix:
        :return:
        """

        if validation_perf_matrix.empty:
            raise ValueError("🚩Error: Validation performance matrix is empty.")

        if train_perf_matrix is not None:
            for model_name in validation_perf_matrix.columns:
                train_f1 = train_perf_matrix.loc["F1", model_name]
                val_f1 = validation_perf_matrix.loc["F1", model_name]
                gap = train_f1 - val_f1

                if gap > OVERFITTING_THRESHOLD:
                    print(f"⚠️ Warning: Model {model_name} is overfitting! (Gap: {gap:.2%})")

        # Select best model based on Validation F1 score
        best_model_name = validation_perf_matrix.loc["F1"].idxmax()
        best_f1 = validation_perf_matrix.loc["F1"].max() * 100

        print(f"🏆 Best Validation Model Identified: {best_model_name} with Validation F1-Score: {best_f1:.4f}")
        return best_model_name

    @staticmethod
    def get_final_model(best_model_name: str, models: list):
        """
        Finds and returns the actual model object matching the best validation name.
        """
        for model in models:
            if model.title == best_model_name:
                return model

        raise ValueError(f"🚩Error: Could not find model matching name: {best_model_name}")

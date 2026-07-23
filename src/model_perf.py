# src/model_perf

# Vendor Libraries
import pandas as pd
from src.constants import OVERFITTING_THRESHOLD
from tensorflow.keras import Sequential

# Local Libraries
from src.eda import model_performance_classification
from src.utils import show_banner


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
            show_banner(f"{self._title} Model Performance", [self.data])
        else:
            raise ValueError("🚩Error: Model Performance has not been run.")


    # --- Static Methods --- #

    @staticmethod
    def get_best_model_name(performance_matrix: pd.DataFrame, comparison_matrix: pd.DataFrame | None = None) -> str:
        """
        Identifies the best performing model based on the highest F1-score in the provided matrix.

        :param performance_matrix: The primary DataFrame to evaluate (Validation or Test).
        :param comparison_matrix: Optional secondary DataFrame (Training) to check for overfitting.
        :return: String name of the best model.
        """
        if performance_matrix.empty:
            raise ValueError("🚩Error: Performance matrix is empty.")

        if comparison_matrix is not None:
            # Actual overfitting check: calculate the gap between Training and Validation
            for model_name in performance_matrix.columns:
                train_f1 = comparison_matrix.loc["F1", model_name]
                val_f1 = performance_matrix.loc["F1", model_name]
                gap = train_f1 - val_f1
                
                if gap > OVERFITTING_THRESHOLD: # Threshold of 15% difference
                    print(f"⚠️ Warning: Model {model_name} is overfitting! (Gap: {gap:.2%})")

        # Find the model name with the highest F1 score
        best_model_name = performance_matrix.loc["F1"].idxmax()
        best_f1 = performance_matrix.loc["F1"].max()
        
        print(f"🏆 Best Model Identified: {best_model_name} with F1-Score: {best_f1:.4f}")
        return best_model_name

    @staticmethod
    def get_test_model_perfs(models: list) -> list:
        """
        Generates evaluation Series for all models using hold-out test data.

        :param models: List of model objects containing trained models and test data
        :return: List of pandas Series objects
        """

        metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
        test_model_perfs = []
        for test_model in models:
            test_model.model_perf.get(
                f"{test_model.title} (Test Set)",
                test_model.model,
                test_model.x_test_norm,
                test_model.y_test
            )

            # Create Model Series with the model title. This embeds the title directly inside the data object.
            test_model_series = pd.Series(
                test_model.model_perf.data.to_numpy().flatten(),
                index=metric_labels,
                name=test_model.title,
            )
            test_model_perfs.append(test_model_series)

        return test_model_perfs

    @staticmethod
    def get_final_model(test_model_perfs: list, models: list) -> object:
        """
        Gets the final model that will be used for customer results based on test data.
        :param test_model_perfs: List of performance Series objects
        :param models: List of original model wrapper objects
        :return: The winning model object
        """

        # Convert the list of evaluation series into a single DataFrame for the best_model logic
        test_model_perfs_matrix = pd.concat(test_model_perfs, axis=1)

        best_model_name = ModelPerformance.get_best_model_name(test_model_perfs_matrix, comparison_matrix=None)

        # Now that we have the best model name let's return the final model
        final_model = None
        for model in models:
            if model.title == best_model_name:
                final_model = model
                break

        return final_model

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
            :param titles:
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

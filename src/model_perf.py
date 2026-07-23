"""
Module for managing and comparing the performance of machine learning models.

This module defines the `ModelPerformance` class, which is responsible
for calculating, storing, and displaying performance metrics for trained
models. It also provides static methods for comparing multiple models
and identifying the best-performing one based on validation metrics.
"""

# Standard Library Imports
from typing import List, Tuple, Union

# Third-party Imports
import pandas as pd
from tensorflow.keras import Sequential

# Local Imports
from src.constants import OVERFITTING_THRESHOLD
from src.eda import model_performance_classification
from src.utils import format_performance, show_banner


class ModelPerformance:
    """
    Handles the calculation, storage, and display of a single model's
    performance metrics.

    Attributes:
        _title (str): The title or name of the model whose performance
                      is being managed.
        data (pd.DataFrame | None): A DataFrame containing the
                                    classification metrics for the model.
                                    None if performance has not been run.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the ModelPerformance class.
        """
        self._title: str = ""
        self.data: Union[pd.DataFrame, None] = None

    def get(
        self,
        title: str,
        model: Sequential,
        x_data: pd.DataFrame,
        y_data: pd.Series,
    ) -> pd.DataFrame:
        """
        Calculates and stores the performance metrics for a given model.

        Args:
            title (str): A descriptive title for the performance run.
            model (Sequential): The trained Keras Sequential model to evaluate.
            x_data (pd.DataFrame): The features (independent variables)
                                   for evaluation.
            y_data (pd.Series): The true labels (ground truth) for evaluation.

        Returns:
            pd.DataFrame: A DataFrame containing the classification metrics.
        """
        self._title = title
        self.data = model_performance_classification(model, x_data, y_data)
        return self.data

    def show(self) -> None:
        """
        Displays the model's performance metrics using a formatted banner.

        Raises:
            ValueError: If `get()` has not been called and no performance
                        data is available.
        """
        if self.data is not None:
            subtitles = format_performance(self.data)
            show_banner(f"{self._title} Model Performance", subtitles)
        else:
            raise ValueError("🚩Error: Model Performance has not been run.")

    @staticmethod
    def create_comparisons(
        models: List[object],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates comparison matrices for training and validation
        performances across multiple models.

        Args:
            models (List[object]): A list of model objects, each expected
                                   to have `title`, `train_perf`, and
                                   `val_perf` attributes.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two
                                               DataFrames:
                                               - The training performance matrix.
                                               - The validation performance matrix.
        """

        def _get_matrices(
            model_titles: List[str], perfs: List[pd.DataFrame]
        ) -> pd.DataFrame:
            """
            Helper function to create a performance matrix from a list
            of model titles and their performance DataFrames.

            Args:
                model_titles (List[str]): A list of model titles.
                perfs (List[pd.DataFrame]): A list of performance
                                            DataFrames.

            Returns:
                pd.DataFrame: A consolidated DataFrame of performances.
            """
            # Convert list to a single DataFrame
            perfs_matrix = pd.concat(perfs)
            perfs_matrix.index = model_titles
            perfs_matrix = perfs_matrix.T
            return perfs_matrix

        # Define comparison variables
        titles: List[str] = []
        train_perfs: List[pd.DataFrame] = []
        val_perfs: List[pd.DataFrame] = []

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
    def get_best_model_name(
        validation_perf_matrix: pd.DataFrame,
        train_perf_matrix: Union[pd.DataFrame, None] = None,
    ) -> str:
        """
        Identifies the best model based strictly on validation F1 score,
        with an optional check for overfitting against training performance.

        Args:
            validation_perf_matrix (pd.DataFrame): DataFrame containing
                                                   validation performance
                                                   metrics for all models.
            train_perf_matrix (pd.DataFrame | None, optional): DataFrame
                                                               containing
                                                               training
                                                               performance
                                                               metrics.
                                                               Used for
                                                               overfitting
                                                               checks.
                                                               Defaults to
                                                               None.

        Returns:
            str: The name of the best-performing model.

        Raises:
            ValueError: If the validation performance matrix is empty.
        """
        if validation_perf_matrix.empty:
            raise ValueError("🚩Error: Validation performance matrix is empty.")

        if train_perf_matrix is not None:
            for model_name in validation_perf_matrix.columns:
                train_f1 = train_perf_matrix.loc["F1", model_name]
                val_f1 = validation_perf_matrix.loc["F1", model_name]
                gap = train_f1 - val_f1

                if gap > OVERFITTING_THRESHOLD:
                    print(
                        f"⚠️ Warning: Model {model_name} is overfitting! "
                        f"(Gap: {gap:.2%})"
                    )

        # Select best model based on Validation F1 score
        best_model_name: str = validation_perf_matrix.loc["F1"].idxmax()
        best_f1: float = validation_perf_matrix.loc["F1"].max() * 100

        print(
            f"🏆 Best Validation Model Identified: {best_model_name} with "
            f"Validation F1-Score: {best_f1:.4f}"
        )
        return best_model_name

    @staticmethod
    def get_final_model(best_model_name: str, models: List[object]) -> object:
        """
        Finds and returns the actual model object that matches the
        identified best model name.

        Args:
            best_model_name (str): The name of the best-performing model.
            models (List[object]): A list of model objects to search through.

        Returns:
            object: The model object corresponding to `best_model_name`.

        Raises:
            ValueError: If no model matching the `best_model_name` is found.
        """
        for model in models:
            if model.title == best_model_name:
                return model

        raise ValueError(
            f"🚩Error: Could not find model matching name: {best_model_name}"
        )
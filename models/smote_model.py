"""
Module for implementing the SMOTE (Synthetic Minority Over-sampling Technique)
model to handle imbalanced datasets.

This module provides the `SmoteModel` class, which generates synthetic
samples for the minority class to balance the dataset, thereby improving
the performance of machine learning models on imbalanced data.
"""

# Third-party Imports
import pandas as pd
from imblearn.over_sampling import SMOTE

# Local Imports
from src.constants import SEED, SMOTE_K_NEIGHBORS, SMOTE_SAMPLING_STRATEGY


class SmoteModel:
    """
    SMOTE (Synthetic Minority Over-sampling Technique) implementation.

    SMOTE creates new synthetic examples for the minority class by
    looking at the "distance" between existing minority points
    (K-Nearest Neighbors). This helps to balance the dataset and
    improve model performance on imbalanced classification problems.

    Attributes:
        x (pd.DataFrame): The resampled features after applying SMOTE.
        y (pd.Series): The resampled target labels after applying SMOTE.
        model (SMOTE): The configured SMOTE instance.
    """

    def __init__(self, dataset: dict) -> None:
        """
        Initializes the SmoteModel and applies SMOTE to the training data.

        Args:
            dataset (dict): A dictionary containing the dataset splits,
                            expected to have 'x_train_norm' (normalized
                            training features) and 'y_train' (training
                            target labels).
        """
        self.x: pd.DataFrame
        self.y: pd.Series

        self.model: SMOTE = self._create()
        self.x, self.y = self.model.fit_resample(
            dataset["x_train_norm"], dataset["y_train"]
        )

        # Check the shapes
        print("\n# --- Loading SMOTE Model --- #")
        print(f"Shape of x_smote: {self.x.shape}")
        print(f"Shape of y_smote: {self.y.shape}")

    def _create(self) -> SMOTE:
        """
        Creates and configures a SMOTE instance.

        Returns:
            SMOTE: A configured SMOTE object with specified sampling
                   strategy, k-neighbors, and random state.
        """
        return SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=SEED,
        )
"""
Module for defining the Stochastic Gradient Descent (SGD) Neural Network
model with SMOTE for handling imbalanced datasets.

This module implements an SGD-optimized neural network model that
leverages SMOTE (Synthetic Minority Over-sampling Technique) to address
class imbalance. It extends the `BaseModel` and configures the model
architecture and the SGD optimizer with specific learning rate and momentum.
"""

# Third-party Imports
from typing import Optional, Union

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Local Imports
from models.base_model import BaseModel
from src.constants import (
    CUSTOMER_CHURN_PROB_THRESHOLD,
    NEURON_DEFAULT_CNT,
    NEURON_SINGLE_CNT,
    SGD_OPT_MOMENTUM,
    SGD_SMOTE_LEARNING_RATE,
)


class SGDSmoteModel(BaseModel):
    """
    Implements a Neural Network model optimized with Stochastic Gradient
    Descent (SGD) and trained with SMOTE-augmented data.

    This model is designed to perform well on imbalanced datasets by
    synthetically increasing the number of minority class samples during
    training.

    Attributes:
        title (str): The title of the model, set to
                     "Neural Network (SGD with SMOTE)".
        y_predictor (Optional[np.ndarray]): The predicted labels for the
                                            test set, generated after
                                            model training.
    """

    def __init__(self, dataset: dict) -> None:
        """
        Initializes the SGDSmoteModel, setting up the optimizer and
        building the model architecture.

        Args:
            dataset (dict): A dictionary containing split and normalized
                            data for model training and evaluation.
        """
        super().__init__(dataset)

        self.title = "Neural Network (SGD with SMOTE)"
        self._optimizer = SGD(
            learning_rate=SGD_SMOTE_LEARNING_RATE, momentum=SGD_OPT_MOMENTUM
        )
        self.model = self._create()
        self.y_predictor: Optional[np.ndarray] = None

    def _create(self) -> Sequential:
        """
        Creates the neural network architecture for the SGD with SMOTE model.

        Extends the base model's input layer with two hidden layers
        and an output layer.

        Returns:
            Sequential: The configured Keras Sequential model.
        """
        model = super()._create()

        model.add(
            Dense(
                NEURON_DEFAULT_CNT,
                kernel_initializer="he_uniform",
                activation="relu",
                name="sgd_smote_model_layer_01",
            )
        )  # Second hidden layer
        model.add(
            Dense(
                NEURON_DEFAULT_CNT,
                kernel_initializer="he_uniform",
                activation="relu",
                name="sgd_smote_model_layer_02",
            )
        )  # Third hidden layer
        model.add(
            Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="sgd_smote_model_layer_03")
        )  # Output layer for binary classification

        return model

    def predict_test_data(self) -> np.ndarray:
        """
        Generates binary predictions for the test dataset using the
        trained model and a predefined churn probability threshold.

        Returns:
            np.ndarray: An array of binary predictions (0 or 1) for the
                        test set.
        """
        if self.model is None:
            raise ValueError("Model not created. Cannot make predictions.")
        return (
            self.model.predict(self.x_test_norm, verbose=0)
            > CUSTOMER_CHURN_PROB_THRESHOLD
        ).astype(int)

    def run(
        self,
        x_smote_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_smote_data: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """
        Executes the model training and evaluation pipeline, specifically
        for the SGD with SMOTE model.

        This method orchestrates the model creation, compilation, training
        with SMOTE data, and performance plotting. It also sets the
        `y_predictor` attribute after training.

        Args:
            x_smote_data (Optional[Union[pd.DataFrame, np.ndarray]]):
                                                                    SMOTE
                                                                    features
                                                                    to use
                                                                    for
                                                                    training.
                                                                    Defaults
                                                                    to None.
            y_smote_data (Optional[Union[pd.Series, np.ndarray]]):
                                                                   SMOTE
                                                                   labels
                                                                   to use
                                                                   for
                                                                   training.
                                                                   Defaults
                                                                   to None.
        """
        super().run(x_smote_data, y_smote_data)
        self.y_predictor = self.predict_test_data()
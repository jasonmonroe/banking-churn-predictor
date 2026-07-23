"""
Module for defining the Adam-optimized Neural Network model with Dropout
and SMOTE for handling imbalanced datasets and preventing overfitting.

This module implements an Adam-optimized neural network model that
combines Dropout regularization with SMOTE (Synthetic Minority
Over-sampling Technique). It extends the `BaseModel` and configures
the model architecture to include Dropout layers and the Adam optimizer
with a specific learning rate.
"""

# Standard Library Imports
from typing import Optional, Union

# Third-party Imports
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Local Imports
from models.base_model import BaseModel
from src.constants import (
    ADAM_SMOTE_DROPOUT_LEARNING_RATE,
    CUSTOMER_CHURN_PROB_THRESHOLD,
    DROPOUT_SMOTE_RATE,
    NEURON_DEFAULT_CNT,
    NEURON_SINGLE_CNT,
)


class AdamSmoteDropoutModel(BaseModel):
    """
    Implements a Deep Neural Network using the Adam optimizer, Dropout
    regularization, and trained with SMOTE-augmented data.

    This model is designed to address both class imbalance and overfitting,
    providing a robust solution for churn prediction.

    Attributes:
        title (str): The title of the model, set to
                     "Neural Network (Adam with Dropout and SMOTE)".
        y_predictor (Optional[np.ndarray]): The predicted labels for the
                                            test set, generated after
                                            model training.
    """

    def __init__(self, dataset: dict) -> None:
        """
        Initializes the AdamSmoteDropoutModel, setting up the optimizer
        and building the model architecture.

        Args:
            dataset (dict): A dictionary containing split and normalized
                            data for model training and evaluation.
        """
        super().__init__(dataset)

        self.title = "Neural Network (Adam with Dropout and SMOTE)"
        self._optimizer = Adam(learning_rate=ADAM_SMOTE_DROPOUT_LEARNING_RATE)
        self.model = self._create()
        self.y_predictor: Optional[np.ndarray] = None

    def _create(self) -> Sequential:
        """
        Architects the Sequential model layers, incorporating Dropout
        layers for regularization.

        Extends the base model's input layer with a Dropout layer, a
        hidden layer, another Dropout layer, and an output layer.

        Returns:
            Sequential: A Keras Sequential model with hidden layers and
                        dropout regularization.
        """
        model = super()._create()

        model.add(Dropout(DROPOUT_SMOTE_RATE, name="adam_smote_dropout_model_layer_01"))
        model.add(
            Dense(
                NEURON_DEFAULT_CNT,
                activation="relu",
                name="adam_smote_dropout_model_layer_02",
            )
        )
        model.add(Dropout(DROPOUT_SMOTE_RATE, name="adam_smote_dropout_model_layer_03"))

        # Output layer for binary classification
        model.add(
            Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="adam_smote_dropout_model_layer_04")
        )

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
        for the Adam with Dropout and SMOTE model.

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
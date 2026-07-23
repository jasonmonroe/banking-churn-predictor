"""
Module for defining the Adam-optimized Neural Network model.

This module implements an Adam-optimized neural network model by extending
the `BaseModel`. It configures the model architecture and the Adam
optimizer with a specific learning rate.
"""

# Third-party Imports
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Local Imports
from models.base_model import BaseModel
from src.constants import ADAM_LEARNING_RATE, NEURON_DEFAULT_CNT, NEURON_SINGLE_CNT


class AdamModel(BaseModel):
    """
    Implements a Neural Network model optimized with the Adam algorithm.

    Adam (Adaptive Moment Estimation) is an advanced optimization
    algorithm used to train neural networks. It automatically calculates
    a unique, dynamically changing learning rate for every single
    parameter in the network, addressing some limitations of traditional
    Stochastic Gradient Descent (SGD).

    Attributes:
        title (str): The title of the model, set to
                     "Neural Network (Adam Optimizer)".
    """

    def __init__(self, dataset: dict) -> None:
        """
        Initializes the AdamModel, setting up the optimizer and building
        the model architecture.

        Args:
            dataset (dict): A dictionary containing split and normalized
                            data for model training and evaluation.
        """
        super().__init__(dataset)

        self.title = "Neural Network (Adam Optimizer)"
        self._optimizer = Adam(learning_rate=ADAM_LEARNING_RATE)
        self.model = self._create()

    def _create(self) -> Sequential:
        """
        Creates the neural network architecture for the Adam model.

        Extends the base model's input layer with a hidden layer and an
        output layer.

        Returns:
            Sequential: The configured Keras Sequential model.
        """
        model = super()._create()

        # Adding a hidden layer with 32 neurons and ReLU activation.
        model.add(
            Dense(NEURON_DEFAULT_CNT, activation="relu", name="adam_model_layer_01")
        )
        # Adding the output layer with one neuron and sigmoid activation
        # for binary classification.
        model.add(
            Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="adam_model_layer_02")
        )

        return model
# models/adam_model.py

# Vendor Libraries
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Local Libraries
from models.base_model import BaseModel
from src.constants import (
    ADAM_LEARNING_RATE,
    NEURON_DEFAULT_CNT,
    NEURON_SINGLE_CNT,
)


class AdamModel(BaseModel):
    """
    Adam (short for Adaptive Moment Estimation) is not a type of neural network model itself. Instead, it is an
    advanced optimization algorithm used to train neural networks.
    Adam solves SGD by automatically calculating a unique, dynamically changing learning rate for every single
    parameter in your network.
    """

    def __init__(self, dataset: dict) -> None:
        super().__init__(dataset)

        self.title = "Neural Network (Adam Optimizer)"
        self._optimizer = Adam(learning_rate=ADAM_LEARNING_RATE)
        self.model = self._create()

    def _create(self) -> Sequential:
        # Choose the metric of choice with proper rationale - Train a Neural Network model with SGD as an optimizer
        model = super()._create()

        model.add(Dense(NEURON_DEFAULT_CNT, activation="relu", name="adam_model_layer_01"))  # Second hidden layer
        model.add(Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="adam_model_layer_02"))  # Output layer for binary classification

        return model

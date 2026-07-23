"""
Module for defining the Stochastic Gradient Descent (SGD) Neural Network model.

This module implements an SGD-optimized neural network model by extending
the `BaseModel`. It configures the model architecture and the SGD
optimizer with specific learning rate and momentum.
"""

# Third-party Imports
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Local Imports
from models.base_model import BaseModel
from src.constants import (
    NEURON_DEFAULT_CNT,
    NEURON_SINGLE_CNT,
    SGD_LEARNING_RATE,
    SGD_OPT_MOMENTUM,
)


class SGDModel(BaseModel):
    """
    Implements a Neural Network model optimized with Stochastic Gradient
    Descent (SGD).

    SGD is a fundamental optimization algorithm that updates model
    parameters using a single training example or a small subset
    (mini-batch) at a time, making it efficient for large datasets.

    Attributes:
        title (str): The title of the model, set to "Neural Network (SGD)".
    """

    def __init__(self, dataset: dict) -> None:
        """
        Initializes the SGDModel, setting up the optimizer and building
        the model architecture.

        Args:
            dataset (dict): A dictionary containing split and normalized
                            data for model training and evaluation.
        """
        super().__init__(dataset)

        self.title = "Neural Network (SGD)"

        self._optimizer = SGD(
            learning_rate=SGD_LEARNING_RATE, momentum=SGD_OPT_MOMENTUM
        )
        self.model = self._create()

    def _create(self) -> Sequential:
        """
        Creates the neural network architecture for the SGD model.

        Extends the base model's input layer with two hidden layers
        and an output layer.

        Returns:
            Sequential: The configured Keras Sequential model.
        """
        model = super()._create()

        # Adding the first and second hidden layer with 32 neurons,
        # relu as activation function and, he_uniform as weight initializer.
        model.add(
            Dense(
                NEURON_DEFAULT_CNT,
                activation="relu",
                kernel_initializer="he_uniform",
                name="sgd_model_layer_01",
            )
        )
        model.add(
            Dense(
                NEURON_DEFAULT_CNT,
                activation="relu",
                kernel_initializer="he_uniform",
                name="sgd_model_layer_02",
            )
        )

        # Adding the output layer with one neuron and sigmoid as activation.
        # This squashes the output to a probability between 0 and 1,
        # which is necessary for binary classification.
        model.add(
            Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="sgd_model_layer_03")
        )

        return model
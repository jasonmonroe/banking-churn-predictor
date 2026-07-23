"""
Module for defining the Adam-optimized Neural Network model with Dropout
regularization.

This module implements an Adam-optimized neural network model with Dropout
layers by extending the `BaseModel`. It configures the model architecture
to include Dropout for mitigating overfitting and uses the Adam optimizer
with a specific learning rate and weight decay.
"""

# Third-party Imports
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Local Imports
from models.base_model import BaseModel
from src.constants import ADAM_DROPOUT_LEARNING_RATE, DROPOUT_RATE, NEURON_DEFAULT_CNT, NEURON_SINGLE_CNT


class AdamDropoutModel(BaseModel):
    """
    Implements a Deep Neural Network using the Adam optimizer and Dropout
    regularization.

    This model is designed to mitigate overfitting in churn prediction
    through stochastic neuron deactivation and weight decay, enhancing
    generalization performance.

    Attributes:
        title (str): The title of the model, set to
                     "Neural Network (Adam and Dropout)".
    """

    def __init__(self, dataset: dict) -> None:
        """
        Initializes the AdamDropoutModel.

        Args:
            dataset (dict): Dictionary containing split and normalized
                            dataframes for model training and evaluation.
        """
        super().__init__(dataset)

        self.title = "Neural Network (Adam and Dropout)"
        self._optimizer = Adam(
            learning_rate=ADAM_DROPOUT_LEARNING_RATE, weight_decay=1e-4
        )
        self.model = self._create()

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
        # BaseModel._create() adds the input layer (Dense)
        model = super()._create()

        model.add(Dropout(DROPOUT_RATE, name="dropout_input_tuning"))
        model.add(
            Dense(
                NEURON_DEFAULT_CNT,
                activation="relu",
                kernel_initializer="he_uniform",
                name="hidden_layer_01",
            )
        )
        model.add(Dropout(DROPOUT_RATE, name="dropout_hidden_tuning"))
        model.add(
            Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="output_layer")
        )

        return model
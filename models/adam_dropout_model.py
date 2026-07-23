# models/adam_dropout_model.py

# Vendor Libraries
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Local Libraries
from models.base_model import BaseModel
from src.constants import (
    DROPOUT_RATE,
    NEURON_DEFAULT_CNT,
    NEURON_SINGLE_CNT,
    ADAM_DROPOUT_LEARNING_RATE
)


class AdamDropoutModel(BaseModel):
    """
    A Deep Neural Network implementation using the Adam optimizer and Dropout.
    
    This model is designed to mitigate overfitting in churn prediction through
    stochastic neurons deactivation and weight decay.
    """

    def __init__(self, dataset: dict) -> None:
        """
        Initializes the AdamDropoutModel.

        Args:
            dataset (dict): Dictionary containing split and normalized dataframes.
        """
        super().__init__(dataset)

        self.title = "Neural Network (Adam and Dropout)"
        self._optimizer = Adam(learning_rate=ADAM_DROPOUT_LEARNING_RATE, weight_decay=1e-4)
        self.model = self._create()

    def _create(self) -> Sequential:
        """
        Architects the Sequential model layers.

        Returns:
            Sequential: A compiled Keras model with hidden layers and dropout.
        """
        # BaseModel._create() adds the input layer (Dense)
        model = super()._create()

        model.add(Dropout(DROPOUT_RATE, name="dropout_input_tuning"))
        model.add(Dense(NEURON_DEFAULT_CNT, activation="relu", kernel_initializer="he_uniform", name="hidden_layer_01"))
        model.add(Dropout(DROPOUT_RATE, name="dropout_hidden_tuning"))
        model.add(Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="output_layer"))

        return model
        
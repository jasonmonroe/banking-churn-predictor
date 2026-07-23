# models/adam_model.py

# Vendor Libraries
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Local Libraries
from models.base_model import BaseModel
from src.constants import LEARNING_RATE, DROPOUT_RATE, NEURON_DEFAULT_CNT, NEURON_SINGLE_CNT


class AdamDropoutModel(BaseModel):
    def __init__(self, dataset: dict) -> None:
        super().__init__(dataset)

        self.title = "Neural Network (Adam and Dropout)"
        self._optimizer = Adam(learning_rate=LEARNING_RATE)
        self.model = self._create()

    def _create(self) -> Sequential:
        model = super()._create()

        model.add(Dropout(DROPOUT_RATE, name="adam_dropout_model_layer_01"))  # Dropout with 25% rate
        model.add(Dense(NEURON_DEFAULT_CNT, activation="relu", kernel_initializer="he_uniform", name="adam_dropout_model_layer_02"))  # Second hidden layer
        model.add(Dropout(DROPOUT_RATE, name="adam_dropout_model_layer_03"))  # Dropout with 25% rate
        model.add(Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="adam_dropout_model_layer_04"))  # Output layer for binary classification

        return model
        
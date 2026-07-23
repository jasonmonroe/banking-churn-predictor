# models/adam_smote_model.py

# Vendor Libraries
from keras import Sequential
from tensorflow.keras.layers import Dense

# Local Libraries
from models.base_model import BaseModel
from src.constants import NEURON_DEFAULT_CNT, NEURON_SINGLE_CNT


class AdamSmoteModel(BaseModel):
    def __init__(self, dataset: dict) -> None:
        super().__init__(dataset)

        self.title = "Neural Network (Adam with SMOTE)"
        self._optimizer = "adam"
        self.model = self._create()

    def _create(self) -> Sequential:
        model = super()._create()

        model.add(Dense(NEURON_DEFAULT_CNT, activation="relu", kernel_initializer="he_uniform", name="adam_smote_model_layer_01"))  # Second hidden layer
        model.add(Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="adam_smote_model_layer_02"))

        return model

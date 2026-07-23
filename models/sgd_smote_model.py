# models/sgd_smote_model.py

# Vendor Libraries
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Local Libraries
from models.base_model import BaseModel
from src.constants import (
    LEARNING_RATE,
    NEURON_DEFAULT_CNT,
    NEURON_SINGLE_CNT,
    CUSTOMER_CHURN_PROB_THRESHOLD,
    SGD_OPT_MOMENTUM,
)


class SGDSmoteModel(BaseModel):
    def __init__(self, dataset: dict) -> None:
        super().__init__(dataset)

        self.title = "Neural Network (SGD with SMOTE)"
        self._optimizer = SGD(learning_rate=LEARNING_RATE, momentum=SGD_OPT_MOMENTUM)
        self.model = self._create()
        self.y_predictor = self._get_predictor()

    def _create(self) -> Sequential:
        model = super()._create()

        model.add(Dense(NEURON_DEFAULT_CNT, kernel_initializer="he_uniform", activation="relu", name="sgd_smote_model_layer_01"))  # Second hidden layer
        model.add(Dense(NEURON_DEFAULT_CNT, kernel_initializer="he_uniform", activation="relu", name="sgd_smote_model_layer_02"))  # Third hidden layer
        model.add(Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="sgd_smote_model_layer_03"))  # Output layer for binary classification

        return model

    def _get_predictor(self) -> int:
        return (self.model.predict(self.x_test_norm) > CUSTOMER_CHURN_PROB_THRESHOLD).astype(int)

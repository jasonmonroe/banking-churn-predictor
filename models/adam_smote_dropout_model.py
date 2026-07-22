# models/adam_smote_dropout_model.py



# Vendor Libraries
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Local Libraries
from models.base_model import BaseModel
from src.constants import LEARNING_RATE, DROPOUT_SMOTE_RATE, NEURON_DEFAULT_CNT, NEURON_SINGLE_CNT


class AdamSmoteDropoutModel(BaseModel):
    def __init__(self, dataset: dict) -> None:
        super().__init__(dataset)

        self.title = "Neural Network (Adam and Dropout with SMOTE)"
        self._optimizer = Adam(learning_rate=LEARNING_RATE)
        self.model = self._create()

    def _create(self) -> Sequential:
        model = super()._create()

        #model.add(Dense(NEURON_CNT, activation='relu', input_shape=feature_cnt))  # First hidden layer
        model.add(Dropout(DROPOUT_SMOTE_RATE))  # Dropout with 30% rate
        model.add(Dense(NEURON_DEFAULT_CNT, activation='relu'))  # Second hidden layer
        model.add(Dropout(DROPOUT_SMOTE_RATE))  # Dropout with 30% rate
    
        # Output layer for binary classification
        model.add(Dense(NEURON_SINGLE_CNT, activation='sigmoid'))  # Output layer with sigmoid for binary classification
        
        return model

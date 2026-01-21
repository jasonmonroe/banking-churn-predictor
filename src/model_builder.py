import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import Union

# Import local libraries
import src.constants as const
from src.evaluation import model_performance_classification
from src.utils import start_timer, show_banner, show_timer


class ModelBuilder:
    def __init__(self, title: str, optimizer: Union[Adam, SGD, str]):
        self.title = title
        self.optimizer = optimizer
        self.model = None

        # Set callbacks
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=10,
            restore_best_weights=True
        )

        self.model_checkpoint = ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        self.clear_session()

    def get(self):
        return self.model
    
    
    def clear_session(self) -> None:
        # Clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
        tf.keras.backend.clear_session()


    def evaluate(self, data_sp):
        test_loss, test_accuracy, test_precision, test_recall, test_auc = self.model.evaluate(
            data_sp.testing['normalized'],
            data_sp.testing['y'],
            verbose=2
        )

        print(f"Test Loss: {test_loss:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}")
        print(f"Test Precision: {test_precision:.2f}")
        print(f"Test Recall: {test_recall:.2f}")
        print(f"Test AUC: {test_auc:.2f}")

        return test_loss, test_accuracy, test_precision, test_recall, test_auc


    def build(self, data_sp, x_smote_data=None, y_smote_data=None) -> History:

        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC'],
        )

        # Fit Model
        start_time = start_timer()

        # If Smote model, use the smote data
        x_training_data = data_sp.training['normalized']
        y_training_data = data_sp.training['y']
        use_smote = False
        if x_smote_data is not None and y_smote_data is not None:
            use_smote = True
            x_training_data = x_smote_data
            y_training_data = y_smote_data

        model_history = self.model.fit(
            x_training_data, #data_sp.training['normalized'],
            y_training_data, #data_sp.training['y'],
            validation_data=(data_sp.validation['normalized'], data_sp.validation['y']),
            epochs=const.EPOCH_CNT,
            batch_size=const.BATCH_CNT,
            verbose=2,
            callbacks=[self.early_stopping, self.model_checkpoint],
        )

        self.evaluate(data_sp)

        # Display model training performance
        show_banner(self.title, 'Training model...')
        show_timer(start_time)

        # Note: Smote Uses normalized
        self.show_model_perf('Training', data_sp.training['normalized'], data_sp.training['y'])
        self.show_model_perf('Validation', data_sp.validation['normalized'], data_sp.validation['y'])

        return model_history

    def get_model_perf(self, x_data, y_data):
        return model_performance_classification(self.model, x_data, y_data)

    def show_model_perf(self, model_type: str, x_data, y_data):
        model_perf = self.get_model_perf(x_data, y_data)
        show_banner(self.title, model_type + ' Model Performance')
        print(model_perf)


    def init_model(self, feature_cnt: int) -> Sequential:
        # Choose the metric of choice with proper rationale - Train a Neural Network model with SGD as an optimizer

        # Initializing the model
        model = Sequential()

        # Adding input layer with 64 neurons, relu as activation function and, he_uniform as weight initializer.
        model.add(Dense(const.NEURON_CNT, activation='relu', kernel_initializer='he_uniform', input_dim=feature_cnt))

        return model

    def create_sgd_model(self, feature_cnt: int) -> Sequential:
        # Choose the metric of choice with proper rationale - Train a Neural Network model with SGD as an optimizer
        # Initializing the model
        #sgd_model = Sequential()
    
        # Adding input layer with 64 neurons, relu as activation function and, he_uniform as weight initializer.
        #sgd_model.add(Dense(const.NEURON_CNT, activation='relu', kernel_initializer='he_uniform', input_dim=feature_cnt))

        model = self.init_model(feature_cnt)
    
        # Adding the first hidden layer with 32 neurons, relu as activation function and, he_uniform as weight initializer
        model.add(Dense(const.DEFAULT_NEURON_CNT, activation='relu', kernel_initializer='he_uniform'))
    
        # Adding the second hidden layer with 32 neurons, relu as activation function and, he_uniform as weight initializer
        model.add(Dense(const.DEFAULT_NEURON_CNT, activation='relu', kernel_initializer='he_uniform'))
    
        # Adding the output layer with one neuron and sigmoid as activation.  This squashes the output to a probability 
        # between 0 and 1, which is necessary for classification.
        model.add(Dense(1, activation='sigmoid'))
    
        # Output is the model summary from Keras/TensorFlow.
        model.summary()

        # Set model
        self.model = model
    
        return model

    def create_adam_model(self, feature_cnt: int) -> Sequential:
        adam_model = Sequential()
        adam_model.add(Dense(const.NEURON_CNT, activation='relu', input_dim=feature_cnt))  # First hidden layer
        adam_model.add(Dense(const.DEFAULT_NEURON_CNT, activation='relu'))  # Second hidden layer
        adam_model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        adam_model.summary()

        self.model = adam_model

    def create_adam_dropout_model(self, feature_cnt: int) -> Sequential:
        #adam_dropout_model = Sequential()

        # Add layers using .add()
        #adam_dropout_model.add(Dense(const.NEURON_CNT, activation='relu', kernel_initializer='he_uniform', input_dim=feature_cnt))  # First hidden layer

        model = self.init_model(feature_cnt)

        model.add(Dropout(const.DROPOUT_RATE))  # Dropout with 25% rate
        model.add(Dense(const.DEFAULT_NEURON_CNT, activation='relu', kernel_initializer='he_uniform'))  # Second hidden layer
        model.add(Dropout(const.DROPOUT_RATE))  # Dropout with 25% rate
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        model.summary()
        self.model = model

        return model

    def create_smote_model(self, feature_cnt: int):
        # Build the Neural Network
        #sgd_smote_model = Sequential()

        # Add layers using .add()
        #sgd_smote_model.add(Dense(const.NEURON_CNT, activation='relu', kernel_initializer='he_uniform', input_dim=feature_cnt))  # First hidden layer

        model = self.init_model(feature_cnt)

        model.add(Dense(const.DEFAULT_NEURON_CNT, kernel_initializer='he_uniform', activation='relu'))  # Second hidden layer
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        model.summary()
        self.model = model

        return model

    def create_adam_smote_model(self, feature_cnt: int):
        # Build the Neural Network
        model = self.init_model(feature_cnt)

        model.add(Dense(const.DEFAULT_NEURON_CNT, activation='relu', kernel_initializer='he_uniform'))  # Second hidden layer
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        self.model = model

        return model

    def create_adam_smote_dropout_model(self, feature_cnt: int):
        # Build the Neural Network
        model = Sequential()
        model.add(Dense(const.NEURON_CNT, activation='relu', input_shape=feature_cnt))  # First hidden layer
        model.add(Dropout(const.DROPOUT_SMOTE_RATE))  # Dropout with 30% rate
        model.add(Dense(const.DEFAULT_NEURON_CNT, activation='relu'))  # Second hidden layer
        model.add(Dropout(const.DROPOUT_SMOTE_RATE))  # Dropout with 30% rate

        # Output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification
        model.summary()
        self.model = model

        return model

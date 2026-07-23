# models/base_model.py

# Python Libraries
from abc import ABC, abstractmethod

# Vendor Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from src.model_perf import ModelPerformance
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Sequential

# Local Libraries
from src.constants import NEURON_CNT, EPOCH_CNT, BATCH_CNT
from src.eda import plot_model_performance
from src.utils import start_timer, show_banner, show_timer, get_time


class BaseModel(ABC):
    def __init__(self, dataset: dict) -> None:
        self._clear_session()
        self._early_stopping = self._init_early_stopping()
        self._model_checkpoint = self._init_model_checkpoint()
        self._lr_scheduler = self._init_lr_scheduler()

        self.title = ""
        self.model = None
        self._optimizer = None
        self.run_time = ""

        # Performance attributes
        self.model_perf = ModelPerformance()
        self.train_perf = None
        self.val_perf = None
        self.x_data_perf = None
        self.y_data_perf = None

        # Split data attributes
        self.x_train = pd.DataFrame()
        self.y_train = pd.Series(dtype="float64")
        self.x_val = pd.DataFrame()
        self.y_val = pd.Series()
        self.x_test = pd.DataFrame()
        self.y_test = pd.Series()

        # Normalized attributes
        self.x_train_norm = pd.DataFrame()
        self.x_val_norm = pd.DataFrame()
        self.x_test_norm = pd.DataFrame()

        self._set_attrs(dataset)

        # Get feature count for model creation
        self._feature_cnt = self._count_features()

    def _set_attrs(self, dataset) -> None:
        """
        Set attributes
        :param dataset:
        :return:
        """
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _count_features(self) -> int:
        count = self.x_train_norm.shape[1]
        if count == 0:
            raise ValueError("🚩Error: Feature Count cannot be 0. Please check your data.")

        return int(count)

    @abstractmethod
    def _create(self) -> Sequential:
        # Initializing the model
        # https://www.tensorflow.org/guide/keras/sequential_model
        model = Sequential()

        # Adding input layer with 64 neurons, relu as activation function and, he_uniform as weight initializer.
        model.add(Dense(NEURON_CNT, activation="relu", kernel_initializer="he_uniform", input_dim=self._feature_cnt, name="base_model_layer"))

        return model

    def _compile(self) -> None:
        self.model.compile(
            optimizer=self._optimizer,
            loss="binary_crossentropy", # should i use this or 'sparse_categorical_crossentropy'
            metrics=["accuracy", "precision", "recall", "auc"],
        )

    def _show_summary(self) -> None:
        # Output is the model summary from Keras/TensorFlow.
        self.model.summary()

    def _clear_session(self) -> None:
        # Clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
        tf.keras.backend.clear_session()

    def _init_early_stopping(self) -> EarlyStopping:
        return EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            restore_best_weights=True
        )

    def _init_model_checkpoint(self) -> ModelCheckpoint:
        return ModelCheckpoint(
            "best_model.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True
        )

    def _init_lr_scheduler(self) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            monitor="val_loss",     # Metric to monitor (e.g., "val_loss" or "val_auc")
            factor=0.5,             # Factor by which the learning rate will be reduced (new_lr = lr * factor)
            patience=3,             # Number of epochs with no improvement after which learning rate will be reduced
            verbose=1,              # Message type: 1 prints a message when the lr is reduced
            min_delta=1e-4,         # Threshold for measuring the new optimum, to only focus on significant changes
            mode="min",             # "min" means learning rate will be reduced when the monitored quantity stops decreasing
            cooldown=0,             # Number of epochs to wait before resuming normal operation after lr has been reduced
            min_lr=1e-6             # Lower bound on the learning rate
        )

    def _build(self, input_data: pd.DataFrame | np.ndarray | None, target_data: pd.Series | np.ndarray | None) -> History:
        """
        https://geeksforgeeks.org/deep-learning/model-fit-in-tensorflow/

        :param x_data:
        :param y_data:
        :return:
        """

        """
        Build Model Logic
        To get model history use model.fit(x=x_data, y=y_data) with validation data (x_val_norm, y_val)
        #Training: x_data=x_train_norm, y_data=y_train, validation_data=(x_val_norm, y_val)
        #Validation: x_data=x_val_norm, y_data=y_val, validation_data=(x_val_norm, y_val)
        Testing: x_data=x_test_norm, y_data=y_test, NO FITTING, just evaluate(), no plotting model performance
        """

        start_time = start_timer()

        # Get the data for fitting the model
        # ℹ️ Note: Data is either normalized or SMOTE
        x_data, y_data, perf_title = self._get_train_data(input_data, target_data)

        # Fit model
        # https://geeksforgeeks.org/deep-learning/model-fit-in-tensorflow/
        model_history = self.model.fit(
            x=x_data,
            y=y_data,
            batch_size=BATCH_CNT,
            epochs=EPOCH_CNT,
            verbose=2,
            validation_data=(self.x_val_norm, self.y_val),
            validation_split=0.0,
            callbacks=[self._lr_scheduler, self._early_stopping, self._model_checkpoint],
            shuffle=True, # Prevents Batch Bias and improves convergence
        )

        self.run_time = get_time(start_time)
        show_timer(start_time)

        # Display Training and Validation results
        self.train_perf = self._run_model_perf(perf_title + "Training", self.x_train_norm, self.y_train)
        self.val_perf = self._run_model_perf(perf_title + "Validation", self.x_val_norm, self.y_val)

        return model_history

    def evaluate(self) -> tuple:
        """
        Evaluates model.

        https://www.tensorflow.org/guide/keras/training_with_built_in_methods
        https://https://www.geeksforgeeks.org/deep-learning/model-evaluate-in-tensorflow/


        model.evaluate(
        x=None,
        y=None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        use_multiprocessing=False
        )

        :param x_data:
        :param y_data:
        :return:
        """

        test_loss, test_accuracy, test_precision, test_recall, test_auc = self.model.evaluate(
            x=self.x_train_norm,
            y=self.y_test,
            batch_size=BATCH_CNT,
            verbose=2,
            return_dict=True,
        )

        # Output Evaluation Results
        subtitles = [
            f"{'Test Loss':<15}: {test_loss:.4f}",
            f"{'Test Accuracy':<15}: {test_accuracy:.4f}",
            f"{'Test Precision':<15}: {test_precision:.4f}",
            f"{'Test Recall':<15}: {test_recall:.4f}",
            f"{'Test AUC':<15}: {test_auc:.4f}"
        ]

        show_banner(f"{self.title} Evaluation Test Results", subtitles)

        return test_loss, test_accuracy, test_precision, test_recall, test_auc

    def _run_model_perf(self, title: str, x_data: pd.DataFrame, y_data: pd.Series) -> pd.DataFrame:
        self.model_perf.get(title, self.model, x_data, y_data)
        self.model_perf.show()

        return self.model_perf.data

    def _get_train_data(self, x_data: pd.DataFrame | np.ndarray | None, y_data: pd.Series | np.ndarray | None):
        """
        Non-Smote Usage:
        Training: x_data=self.x_train_norm, y_data=self.y_train
        Validation: x_data=self.x_val_norm, y_data=self.y_val
        Testing: x_data=self.x_test_normm y_data=self.y_test

        Smote Usage
        Training: x_data=x_smote, y_data=y_smote,
        Validation: x_data=self.x_val_norm, y_data=self.y_val
        Testing: Not applicable

        :param x_data:
        :param y_data:
        :return:
        """

        if x_data is not None and y_data is not None:
            # Smote Data is being used
            return x_data, y_data, self.title + " SMOTE "
        else:
            return self.x_train_norm, self.y_train, self.title

    def run(self, x_smote_data=pd.DataFrame | np.ndarray | None, y_smote_data=pd.Series | np.ndarray | None) -> None:
        """
        Run the model to get the training validation performances.
        Then plot the models performance  based on accuracy and loss.

        ❗Important: If smote data is used as arguments override the other data. Only use x_smote, y_smote
        for training, validation will still use x_val_norm, y_val


        :param x_smote_data:
        :param y_smote_data:
        :return:
        """

        print(f"\n# --- Running {self.title} model --- #\n")

        self._show_summary()
        self._compile()

        # If x_smote and y_smote are None then we will use normalized data
        history = self._build(x_smote_data, y_smote_data)

        # Plot Model Performance
        plot_model_performance(history, "accuracy", self.title)
        plot_model_performance(history, "loss", self.title)

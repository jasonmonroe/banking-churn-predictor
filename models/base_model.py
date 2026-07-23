# models/base_model.py

# Python Libraries
from abc import ABC, abstractmethod

# Vendor Libraries
import pandas as pd
import tensorflow as tf
from src.model_perf import ModelPerformance
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

# Local Libraries
from src.constants import NEURON_CNT, EPOCH_CNT, BATCH_CNT
from src.eda import model_performance_classification, plot_model_performance
from src.utils import start_timer, show_banner, show_timer, get_time


class BaseModel(ABC):
    def __init__(self, dataset: dict) -> None:
        self._clear_session()
        self._early_stopping = self._init_early_stopping()
        self._model_checkpoint = self._init_model_checkpoint()

        self.title = ""
        self.model = None
        self._optimizer = None
        self.run_time = ""

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
        model = Sequential()

        # Adding input layer with 64 neurons, relu as activation function and, he_uniform as weight initializer.
        init_layer = Dense(NEURON_CNT, activation="relu", kernel_initializer="he_uniform", input_dim=self._feature_cnt)

        model.add(init_layer)

        return model

    def _compile(self) -> None:
        self.model.compile(
            optimizer=self._optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall", "AUC"],
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

    def _build(self, x_data=None, y_data=None) -> History:
        start_time = start_timer()

        # Get the training data for fitting the model
        # Note: Data is either normalized or SMOTE
        x_train, y_train, perf_title = self._get_train_data(x_data, y_data)

        # Fit model
        model_history = self.model.fit(
            x_train,
            y_train,
            validation_data=(self.x_val_norm, self.y_val),
            epochs=EPOCH_CNT,
            batch_size=BATCH_CNT,
            verbose=2,
            callbacks=[self._early_stopping, self._model_checkpoint],
        )

        # Evaluate model
        self.evaluate()

        self.run_time = get_time(start_time)
        show_timer(start_time)

        # Display Training and Validation results
        self.train_perf = self._run_model_perf(perf_title + "Training", self.x_train_norm, self.y_train)
        self.val_perf = self._run_model_perf(perf_title + "Validation", self.x_val_norm, self.y_val)

        return model_history

    def evaluate(self):
        test_loss, test_accuracy, test_precision, test_recall, test_auc = self.model.evaluate(
            self.x_test_norm,
            self.y_test,
            verbose=2
        )

        # Output Evaluation Results
        subtitles = [
            f"{'Test Loss':<15}: {test_loss:.4f}",
            f"{'Test Accuracy':<15}: {test_accuracy:.4f}",
            f"{'Test Precision':<15}: {test_precision:.4f}",
            f"{'Test Recall':<15}: {test_recall:.4f}",
            f"{'Test AUC':<15}: {test_auc:.4f}"
        ]

        show_banner(f"{self.title} Evaluation Results", subtitles)

        return test_loss, test_accuracy, test_precision, test_recall, test_auc

    def _run_model_perf(self, title: str, x_data: pd.DataFrame, y_data: pd.Series):
        self.model_perf.get(title, self.model, x_data, y_data)
        self.model_perf.show()

        return self.model_perf.data

    def _get_train_data(self, x_data: pd.DataFrame | None, y_data: pd.Series | None) -> tuple[pd.DataFrame, pd.Series, str]:
        """
        Get normalized or SMOTE data for model performance.
        :param x_data:
        :param y_data:
        :return:
        """

        # If smote model, use the smote data
        title = self.title + " "
        x_train = self.x_train_norm
        y_train = self.y_train

        if x_data is not None and y_data is not None:
            title = f"{self.title} SMOTE "
            x_train = x_data
            y_train = y_data

        return x_train, y_train, title

    def run(self, x_smote_data=None, y_smote_data=None) -> None:
        """
        Run the model to get the training validation performances.
        Then plot the models performance  based on accuracy and loss.

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




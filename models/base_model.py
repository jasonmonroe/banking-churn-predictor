"""
Base module for defining abstract machine learning models.

This module provides the `BaseModel` abstract base class, which
establishes a common interface and shared functionalities for all
concrete model implementations in the project. It includes methods for
session management, early stopping, model checkpointing, learning rate
scheduling, data attribute setting, feature counting, model compilation,
evaluation, and performance tracking.
"""

# Standard Library Imports
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

# Third-party Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import F1Score

# Local Imports
from src.constants import BATCH_CNT, EPOCH_CNT, NEURON_CNT
from src.eda import plot_model_performance
from src.model_perf import ModelPerformance
from src.utils import get_time, show_banner, show_timer, start_timer


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in the project.

    Provides common functionalities such as TensorFlow session management,
    callback initialization, data attribute setting, model building,
    compilation, training, evaluation, and performance tracking.

    Attributes:
        title (str): The title or name of the model.
        model (Optional[Sequential]): The Keras Sequential model instance.
        _optimizer (Optional[tf.keras.optimizers.Optimizer]): The optimizer
                                                              used for the
                                                              model.
        run_time (str): The formatted string of the model's training run time.
        model_perf (ModelPerformance): An instance of ModelPerformance
                                       to track and display metrics.
        train_perf (Optional[pd.DataFrame]): Performance metrics on the
                                              training set.
        val_perf (Optional[pd.DataFrame]): Performance metrics on the
                                            validation set.
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        x_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        x_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        x_train_norm (pd.DataFrame): Normalized training features.
        x_val_norm (pd.DataFrame): Normalized validation features.
        x_test_norm (pd.DataFrame): Normalized testing features.
        _feature_cnt (int): Number of features in the input data.
    """

    def __init__(self, dataset: Dict[str, pd.DataFrame]) -> None:
        """
        Initializes the BaseModel with dataset and common configurations.

        Args:
            dataset (Dict[str, pd.DataFrame]): A dictionary containing
                                               split and normalized data
                                               (e.g., x_train, y_train,
                                               x_val_norm, etc.).
        """
        self._clear_session()
        self._early_stopping: EarlyStopping = self._init_early_stopping()
        self._model_checkpoint: ModelCheckpoint = self._init_model_checkpoint()
        self._lr_scheduler: ReduceLROnPlateau = self._init_lr_scheduler()

        self.title: str = ""
        self.model: Optional[Sequential] = None
        self._optimizer: Optional[tf.keras.optimizers.Optimizer] = None
        self.run_time: str = ""

        # Performance attributes
        self.model_perf: ModelPerformance = ModelPerformance()
        self.train_perf: Optional[pd.DataFrame] = None
        self.val_perf: Optional[pd.DataFrame] = None

        # Split data attributes
        self.x_train: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series(dtype="float64")
        self.x_val: pd.DataFrame = pd.DataFrame()
        self.y_val: pd.Series = pd.Series(dtype="float64")
        self.x_test: pd.DataFrame = pd.DataFrame()
        self.y_test: pd.Series = pd.Series(dtype="float64")

        # Normalized attributes
        self.x_train_norm: pd.DataFrame = pd.DataFrame()
        self.x_val_norm: pd.DataFrame = pd.DataFrame()
        self.x_test_norm: pd.DataFrame = pd.DataFrame()

        self._set_attrs(dataset)

        # Get feature count for model creation
        self._feature_cnt: int = self._count_features()

    def _set_attrs(self, dataset: Dict[str, pd.DataFrame]) -> None:
        """
        Sets instance attributes from the provided dataset dictionary.

        Args:
            dataset (Dict[str, pd.DataFrame]): A dictionary where keys
                                               match instance attribute
                                               names and values are the
                                               corresponding dataframes/series.
        """
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _count_features(self) -> int:
        """
        Counts the number of features in the normalized training data.

        Returns:
            int: The number of features.

        Raises:
            ValueError: If the feature count is zero, indicating an issue
                        with data preparation.
        """
        count: int = self.x_train_norm.shape[1]
        if count == 0:
            raise ValueError(
                "🚩Error: Feature Count cannot be 0. Please check your data."
            )
        return count

    @abstractmethod
    def _create(self) -> Sequential:
        """
        Abstract method to create the Keras Sequential model architecture.

        Concrete model classes must implement this method to define their
        specific neural network layers.

        Returns:
            Sequential: An initialized Keras Sequential model.
        """
        # Initializing the model
        # https://www.tensorflow.org/guide/keras/sequential_model
        model = Sequential()

        # Adding input layer with 64 neurons, relu as activation function
        # and, he_uniform as weight initializer.
        model.add(
            Dense(
                NEURON_CNT,
                activation="relu",
                kernel_initializer="he_uniform",
                input_dim=self._feature_cnt,
                name="base_model_layer",
            )
        )
        return model

    def _compile(self) -> None:
        """
        Compiles the Keras model with the specified optimizer, loss function,
        and metrics.

        F1Score is included as a metric, which is crucial for imbalanced
        classification problems like churn prediction.
        """
        # Adding F1Score as it's the gold standard for imbalanced churn data
        f1_metric = F1Score(name="f1_score", dtype=None, threshold=0.5)

        if self.model is None:
            raise ValueError("Model not created. Call _create() first.")
        if self._optimizer is None:
            raise ValueError("Optimizer not set. Set _optimizer in subclass.")

        self.model.compile(
            optimizer=self._optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall", "auc", f1_metric],
        )

    def _show_summary(self) -> None:
        """
        Prints the summary of the Keras model architecture, including
        layer names, output shapes, and number of parameters.
        """
        if self.model:
            self.model.summary()
        else:
            print("Model not created yet. Cannot show summary.")

    def _clear_session(self) -> None:
        """
        Clears the current Keras session.

        This resets all layers and models previously created, freeing up
        memory and resources, which is useful when training multiple models.
        """
        tf.keras.backend.clear_session()

    def _init_early_stopping(self) -> EarlyStopping:
        """
        Initializes and returns an EarlyStopping callback.

        Monitors validation loss, stops training if no improvement for
        a specified patience, and restores the best weights found.

        Returns:
            EarlyStopping: Configured EarlyStopping callback.
        """
        return EarlyStopping(
            monitor="val_loss", mode="min", patience=8, restore_best_weights=True
        )

    def _init_model_checkpoint(self) -> ModelCheckpoint:
        """
        Initializes and returns a ModelCheckpoint callback.

        Saves the best model (based on validation loss) to a file during
        training.

        Returns:
            ModelCheckpoint: Configured ModelCheckpoint callback.
        """
        return ModelCheckpoint(
            "best_model.keras", monitor="val_loss", mode="min", save_best_only=True
        )

    def _init_lr_scheduler(self) -> ReduceLROnPlateau:
        """
        Initializes and returns a ReduceLROnPlateau callback.

        Reduces the learning rate when the validation loss stops improving
        to help the model converge better.

        Returns:
            ReduceLROnPlateau: Configured ReduceLROnPlateau callback.
        """
        return ReduceLROnPlateau(
            monitor="val_loss",  # Metric to monitor
            factor=0.5,  # Factor by which the learning rate will be reduced
            patience=3,  # Number of epochs with no improvement
            verbose=1,  # Prints a message when the lr is reduced
            min_delta=1e-4,  # Threshold for measuring the new optimum
            mode="min",  # Learning rate reduced when monitored quantity stops decreasing
            cooldown=0,  # Epochs to wait before resuming normal operation
            min_lr=1e-6,  # Lower bound on the learning rate
        )

    def _build(
        self,
        input_data: Optional[Union[pd.DataFrame, np.ndarray]],
        target_data: Optional[Union[pd.Series, np.ndarray]],
    ) -> History:
        """
        Builds and trains the model using the provided data.

        This method handles the fitting process, including the application
        of callbacks for early stopping, model checkpointing, and learning
        rate scheduling. It also calculates and displays training and
        validation performance.

        Args:
            input_data (Optional[Union[pd.DataFrame, np.ndarray]]): The
                                                                    input
                                                                    features
                                                                    for
                                                                    training.
                                                                    If None,
                                                                    `x_train_norm`
                                                                    is used.
            target_data (Optional[Union[pd.Series, np.ndarray]]): The
                                                                   target
                                                                   labels
                                                                   for
                                                                   training.
                                                                   If None,
                                                                   `y_train`
                                                                   is used.

        Returns:
            History: The Keras History object containing training logs.
        """
        start_time: float = start_timer()

        # Get the data for fitting the model (either normalized or SMOTE)
        x_data, y_data, perf_title = self._get_train_data(input_data, target_data)

        if self.model is None:
            raise ValueError("Model not created. Call _create() first.")

        # Fit model
        # https://geeksforgeeks.org/deep-learning/model-fit-in-tensorflow/
        model_history: History = self.model.fit(
            x=x_data,
            y=y_data,
            batch_size=BATCH_CNT,
            epochs=EPOCH_CNT,
            verbose=2,
            validation_data=(self.x_val_norm, self.y_val),
            validation_split=0.0,  # validation_data is provided separately
            callbacks=[self._lr_scheduler, self._early_stopping, self._model_checkpoint],
            shuffle=True,  # Prevents Batch Bias and improves convergence
        )

        self.run_time = get_time(start_time)
        show_timer(start_time)

        # Display Training and Validation results
        self.train_perf = self._run_model_perf(
            perf_title + "Training", self.x_train_norm, self.y_train
        )
        self.val_perf = self._run_model_perf(
            perf_title + "Validation", self.x_val_norm, self.y_val
        )

        return model_history

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the model's performance on the test dataset.

        Uses the `x_test_norm` and `y_test` attributes for evaluation.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not created. Cannot evaluate.")

        results: Dict[str, float] = self.model.evaluate(
            x=self.x_test_norm,
            y=self.y_test,
            batch_size=BATCH_CNT,
            verbose=2,
            return_dict=True,
        )

        # Output Evaluation Results
        subtitles: List[str] = [
            f"{key.replace('_', ' ').title():<15}: {float(np.mean(value)):.4f}"
            for key, value in results.items()
        ]
        show_banner(f"{self.title} Evaluation Test Results", subtitles)

        return results

    def _run_model_perf(
        self, title: str, x_data: pd.DataFrame, y_data: pd.Series
    ) -> pd.DataFrame:
        """
        Calculates and displays model performance using the `ModelPerformance`
        instance.

        Args:
            title (str): The title for the performance report.
            x_data (pd.DataFrame): Features for performance calculation.
            y_data (pd.Series): True labels for performance calculation.

        Returns:
            pd.DataFrame: A DataFrame containing the performance metrics.
        """
        if self.model is None:
            raise ValueError("Model not created. Cannot run model performance.")
        self.model_perf.get(title, self.model, x_data, y_data)
        self.model_perf.show()

        return self.model_perf.data

    def _get_train_data(
        self,
        x_data: Optional[Union[pd.DataFrame, np.ndarray]],
        y_data: Optional[Union[pd.Series, np.ndarray]],
    ) -> tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], str]:
        """
        Determines which training data (normalized or SMOTE) to use based
        on provided arguments.

        Args:
            x_data (Optional[Union[pd.DataFrame, np.ndarray]]): SMOTE
                                                                features,
                                                                if provided.
            y_data (Optional[Union[pd.Series, np.ndarray]]): SMOTE labels,
                                                              if provided.

        Returns:
            Tuple[Union[pd.DataFrame, np.ndarray],
                  Union[pd.Series, np.ndarray], str]:
                A tuple containing:
                - The features to use for training.
                - The labels to use for training.
                - A title prefix for performance reporting.
        """
        # If SMOTE data is provided, use it; otherwise, use normalized data
        if x_data is not None and y_data is not None:
            print(f"\nℹ️ SMOTE data is being used for {self.title}.")
            return x_data, y_data, self.title + " SMOTE "
        else:
            return self.x_train_norm, self.y_train, self.title

    def run(
        self,
        x_smote_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_smote_data: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """
        Executes the model training and evaluation pipeline.

        This method orchestrates the model creation, compilation, training,
        and performance plotting. It can optionally use SMOTE-generated
        data for training.

        Args:
            x_smote_data (Optional[Union[pd.DataFrame, np.ndarray]]):
                                                                    SMOTE
                                                                    features
                                                                    to use
                                                                    for
                                                                    training.
                                                                    Defaults
                                                                    to None.
            y_smote_data (Optional[Union[pd.Series, np.ndarray]]):
                                                                   SMOTE
                                                                   labels
                                                                   to use
                                                                   for
                                                                   training.
                                                                   Defaults
                                                                   to None.
        """
        print(f"\n# --- Running {self.title} model --- #\n")

        self.model = self._create()  # Create the model architecture
        self._show_summary()
        self._compile()

        # If x_smote and y_smote are None then we will use normalized data
        history: History = self._build(x_smote_data, y_smote_data)

        # Plot Model Performance
        plot_model_performance(history, "accuracy", self.title)
        plot_model_performance(history, "loss", self.title)
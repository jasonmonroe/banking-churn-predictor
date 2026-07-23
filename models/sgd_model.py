# models/sgd_model.py

# Vendor Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Local Libraries
from models.base_model import BaseModel
from src.constants import NEURON_DEFAULT_CNT, NEURON_SINGLE_CNT, SGD_LEARNING_RATE, SGD_OPT_MOMENTUM


class SGDModel(BaseModel):
    """
    Stochastic Gradient Descent (SGD) is a fundamental optimization algorithm in machine learning that minimizes a loss
    function by updating model parameters using a single training example or a small subset (mini-batch) at a time.
    This approach is particularly efficient for large datasets compared to traditional gradient descent, which uses the
    entire dataset for each update.

    https://scikit-learn.org/stable/modules/sgd.html
    """

    def __init__(self, dataset: dict) -> None:
        super().__init__(dataset)

        self.title = "Neural Network (SGD)"

        self._optimizer = SGD(learning_rate=SGD_LEARNING_RATE, momentum=SGD_OPT_MOMENTUM)
        self.model = self._create()

    def _create(self) -> Sequential:
        # Initializing the model
        # Choose the metric of choice with proper rationale - Train a Neural Network model with SGD as an optimizer.
        model = super()._create()

        # Adding the first and second hidden layer with 32 neurons, relu as activation function and, he_uniform as
        # weight initializer.
        model.add(Dense(NEURON_DEFAULT_CNT, activation="relu", kernel_initializer="he_uniform", name="sgd_model_layer_01"))
        model.add(Dense(NEURON_DEFAULT_CNT, activation="relu", kernel_initializer="he_uniform", name="sgd_model_layer_02"))

        # Adding the output layer with one neuron and sigmoid as activation.  This squashes the output to a probability
        # between 0 and 1, which is necessary for classification.
        model.add(Dense(NEURON_SINGLE_CNT, activation="sigmoid", name="sgd_model_layer_03"))

        return model

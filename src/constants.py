"""
Module for storing all constants used across the banking churn predictor project.

This module centralizes configuration parameters, magic numbers, and
other fixed values to ensure consistency and easy management throughout
the application.
"""

from typing import Final, List

# Arguments
# Note: If any individual model is referenced, the "all" command is
# automatically set to False!
ARG_PARAMS: Final[List[str]] = [
    "--all",
    "--eda",  # Show Exploratory Data Analysis
    # Models
    "--model:sgd",
    "--model:adam",
    "--model:adam-dropout",
    "--model:sgd-smote",
    "--model:adam-smote",
    "--model:adam-smote-dropout",
]

# Data Files
FILE_NAME: Final[str] = "source_data.csv"
SOURCE_PATH: Final[str] = "data/"
DATA_FILE_PATH: Final[str] = SOURCE_PATH + FILE_NAME

# Data Splitting Parameters (Temp 80%, 20% Test), (Temp): 75% Training, 25% validation
HALF_SPLIT: Final[float] = 0.5
TEMP_SPLIT: Final[float] = 0.80
TESTING_SPLIT: Final[float] = 0.20
TRAINING_SPLIT: Final[float] = 0.75
VALIDATION_SPLIT: Final[float] = 0.25  # 25% of 80% ~ 20%

# Time Parameters
MSEC: Final[int] = 1000
SECS_IN_MIN: Final[int] = 60

# Model Parameters
BATCH_CNT: Final[int] = 32
DROPOUT_RATE: Final[float] = 0.175
DROPOUT_SMOTE_RATE: Final[float] = 0.3
EPOCH_CNT: Final[int] = 100
NEURON_CNT: Final[int] = 64
NEURON_DEFAULT_CNT: Final[int] = 32
NEURON_SINGLE_CNT: Final[int] = 1

# Learning Rates
ADAM_LEARNING_RATE: Final[float] = 0.001
ADAM_DROPOUT_LEARNING_RATE: Final[float] = 0.001
ADAM_SMOTE_LEARNING_RATE: Final[float] = 0.0005
ADAM_SMOTE_DROPOUT_LEARNING_RATE: Final[float] = 0.0005
SGD_LEARNING_RATE: Final[float] = 0.01
SGD_SMOTE_LEARNING_RATE: Final[float] = 0.005

SMOTE_SAMPLING_STRATEGY: Final[float] = 0.40
SMOTE_K_NEIGHBORS: Final[int] = 3

# Misc Parameters
BALANCE_THRESHOLD: Final[int] = 20000  # In USD ($)
CUSTOMER_CHURN_PROB_THRESHOLD: Final[float] = 0.35
OVERFITTING_THRESHOLD: Final[float] = 0.15
PEP8_LINE_LEN: Final[int] = 79
SEED: Final[int] = 42

# Adds speed (inertia) from past steps. A value of 0.9 means the optimizer
# keeps 90% of its previous direction and adds 10% of the new gradient
# direction. This helps the model push past small bumps and move faster
# down the learning curve.
SGD_OPT_MOMENTUM: Final[float] = 0.9

# Column Parameters
CATEGORICAL_COLS: Final[List[str]] = ["gender", "geography"]
IRRELEVANT_COLS: Final[List[str]] = ["row_number", "customer_id", "surname"]
METRIC_COLS: Final[List[str]] = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
TARGET_COL: Final[str] = "exited"
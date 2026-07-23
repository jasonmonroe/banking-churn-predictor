# src/constants.py

"""
+---------------------------------------------------------------------------+
| CONSTANTS                                                                  |
+---------------------------------------------------------------------------+
"""

# Arguments
# ℹ️ Note: If any individual model is referenced the "all" command is automatically set to False!
ARG_PARAMS = [
    "--all",
    "--eda",    # Show Exploratory Data Analysis

    # Models
    "--model:sgd",
    "--model:adam",
    "--model:adam-dropout",
    "--model:sgd-smote",
    "--model:adam-smote",
    "--model:adam-smote-dropout",
]

# Data Files
FILE_NAME = "source_data.csv"
SOURCE_PATH = "data/"
DATA_FILE_PATH = SOURCE_PATH + FILE_NAME

# Data Splitting Parameters (Temp 80%, 20% Test), (Temp): 75% Training, 25% validation
HALF_SPLIT = 0.5
TEMP_SPLIT = 0.80
TESTING_SPLIT = 0.20
TRAINING_SPLIT = 0.75
VALIDATION_SPLIT = 0.25 # 25% of 80% ~ 20%

# Time Parameters
MSEC = 1000
SECS_IN_MIN = 60

# Model Parameters
BATCH_CNT = 32
DROPOUT_RATE = 0.175
DROPOUT_SMOTE_RATE = 0.3
EPOCH_CNT = 100
NEURON_CNT = 64
NEURON_DEFAULT_CNT = 32
NEURON_SINGLE_CNT = 1
LEARNING_RATE = 0.001
SMOTE_SAMPLING_STRATEGY = 0.40 # Keep minority class capped at 40% of majority size
SMOTE_K_NEIGHBORS = 3          # Look at tighter neighbor bounds to reduce overlap noise

# Misc Parameters
BALANCE_THRESHOLD = 20000 # In USD ($)
CUSTOMER_CHURN_PROB_THRESHOLD = 0.35
OVERFITTING_THRESHOLD = 0.15
PEP8_LINE_LEN = 79
SEED = 42

# Adds speed (inertia) from past steps. A value of 0.9 means the optimizer keeps 90% of its previous direction and adds 
# 10% of the new gradient direction. This helps the model push past small bumps and move faster down the learning 
# curve.
SGD_OPT_MOMENTUM = 0.9

# Column Parameters
CATEGORICAL_COLS = ["gender", "geography"]
IRRELEVANT_COLS = ["row_number", "customer_id", "surname"]
METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
TARGET_COL = "exited"

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
    "--build",  # Build AI Agent
    "--eda",    # Show Exploratory Data Analysis
    "--log",    # Log information

    # Models
    "--model:adam",
    "--model:adam_d",
    "--model:adam_smote",
    "--model:adam_smote_d",
    "--model:sgd",
    "--model:sgd_smote",
]

# Data Files
FILE_NAME = "source_data.csv"
SOURCE_PATH = "data/"
DATA_FILE_PATH = SOURCE_PATH + FILE_NAME
MAX_DATA_ROWS = 10000

# Data Spitting Parameters (Temp 80%, 20% Test), (Temp): 75% Training, 25% validation
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
DROPOUT_RATE = 0.25
DROPOUT_SMOTE_RATE = 0.3
EPOCH_CNT = 100
NEURON_CNT = 64
NEURON_DEFAULT_CNT = 32
NEURON_SINGLE_CNT = 1
LEARNING_RATE = 0.001

# Misc Parameters
BALANCE_THRESHOLD = 20000 # In USD ($)
PEP8_LINE_LEN = 79
PREDICTION_PROB_THRESHOLD = 0.5
SEED = 42

# Adds speed (inertia) from past steps. A value of 0.9 means the optimizer keeps 90% of its previous direction and adds 
# 10% of the new gradient direction. This helps the model push past small bumps and move faster down the learning 
# curve.
SGD_OPT_MOMENTUM = 0.9

CATEGORIAL_COLS = ["gender", "geography"]
IRRELEVANT_COLS = ["row_number", "customer_id", "surname"]
SCALING_COLS = ["x_train", "x_val", "x_test"]
TARGET_COL = "exited"

# constants.py
# ==================================
#  CONSTANTS
# ==================================

FILE_NAME = 'sample_data.csv'
SOURCE_PATH = ''
DATA_FILE_PATH = SOURCE_PATH + FILE_NAME

MAX_DATA_ROWS = 10000

BALANCE_THRESHOLD = 20000 # In $ (dollars)
BATCH_CNT = 32
DROPOUT_RATE = 0.25
DROPOUT_SMOTE_RATE = 0.3
EPOCH_CNT = 100
LEARNING_RATE = 0.001
MSEC = 1000
NEURON_CNT = 64
PREDICTION_PROB_THRESHOLD = 0.5
DEFAULT_NEURON_CNT = 32
SECS_IN_MIN = 60
SEED = 42 # Hard coded seed value for random state
TARGET_COL = 'exited'

# Splitting data
# Original Data = 100%
# First split = Temp 80%, 20% Test
TEMP_SPLIT = 0.80
TESTING_SPLIT = 0.20

# Second split = (Temp): 75% Training, 25% validation
TRAINING_SPLIT = 0.75
VALIDATION_SPLIT = 0.25 # 25% of 80% ~ 20%

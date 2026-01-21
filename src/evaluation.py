import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from tensorflow.keras.models import Sequential

# Define a function to compute different metrics to check the performance of a classification model built using stats models
def model_performance_classification(mod: Sequential, predictors: pd.DataFrame, target: pd.Series, threshold:float=0.5) -> pd.DataFrame:

    """
    Function to compute different metrics to check classification model performance
    model: classifier
    predictors: independent variables
    target: target variable
    threshold: threshold for classification
    """

    # Checking which probabilities are greater than a threshold
    pred = mod.predict(predictors) > threshold

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, average='weighted')
    recall = recall_score(target, pred, average='weighted')
    f1 = f1_score(target, pred, average='weighted')

    return pd.DataFrame({'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1': [f1]})

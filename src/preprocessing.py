import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

import src.constants as const

def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:

    # --- Encoding Categorial Variables ---
    # Gender, Geography should be converted to integers
    categorical_columns = ['gender', 'geography']

    # Generate dummy variables and drop the first category to prevent multicollinearity
    dummy_obj = pd.get_dummies(df[categorical_columns], drop_first=True)

    # Concatenate the dummy variables with the original dataset and drop original categorical columns
    df = pd.concat([df.drop(categorical_columns, axis=1), dummy_obj], axis=1)

    return df

def get_smote(data_sp):
    # Use the SEED from constants for reproducibility
    smote = SMOTE(random_state=const.SEED)

    # SMOTE should be applied to the ALREADY normalized training data
    # This ensures synthetic points follow the same 0-1 rules as your Val/Test sets
    x_smote, y_smote = smote.fit_resample(data_sp.training['normalized'], data_sp.training['y'])

    # Check the shapes
    print(f"Shape of X_smote: {x_smote.shape}")
    print(f"Shape of y_smote: {y_smote.shape}")

    # No need to re-scale. x_smote is already in the 0-1 range.
    return x_smote, y_smote

# Not in use
def split_data(df: pd.DataFrame):

    """
    @deprecated
    Train-Test Split, and scaling logic goes here.

    :param df:
    :return:
    """

    target_col = const.TARGET_COL

    # --- Split Data ----
    X = df.drop(target_col, axis=1) # independent variables (dataframe)
    target_df = df[target_col]              # dependent (target) variable

    # You split the original data into a Training Pool (80%) and a Test Set (20%).
    # The first split is 80% training data and 20% testing data
    x_temp_data, x_testing_data, y_temp_data, y_testing_data = train_test_split(
        X,
        target_df,
        test_size=const.TESTING_SPLIT,
        random_state=const.SEED,
        stratify=target_df # ensures that the target variable's class proportions are maintained across all three sets, which
        # is critical for imbalanced datasets.
    )

    # You split the Training Pool again. To achieve a final 60/20/20 ratio, your VALIDATION_SPLIT must be 0.25
    # (since 25% of 80% is 20%).

    # The second split is the remaining 80% into validation and training data
    x_training_data, x_validation_data, y_training_data, y_validation_data = train_test_split(
        x_temp_data,
        y_temp_data,
        test_size=const.VALIDATION_SPLIT,
        random_state=const.SEED,
        stratify=y_temp_data
    )

    # Printing the shapes
    print(f'Shape of x_training_data: {x_training_data.shape}')
    print(f'Shape of X_valid: {x_validation_data.shape}')
    print(f'Shape of X_test: {x_testing_data.shape}')
    print(f'Shape of y_train: {y_training_data.shape}')
    print(f'Shape of y_valid: {y_validation_data.shape}')
    print(f'Shape of y_test: {y_testing_data.shape}')

    # --- Scaling ---
    numeric_features = x_training_data.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()

    # Training
    # Fit the scaler on the numeric features of the training data
    x_training_data_normalized = x_training_data.copy() # Create a copy to avoid modifying the original DataFrame
    x_training_data_normalized[numeric_features] = scaler.fit_transform(x_training_data[numeric_features])

    # Validation
    # Transform the validation and test data using the fitted scaler, only for numeric features
    x_validation_normalized = x_validation_data.copy()
    x_validation_normalized[numeric_features] = scaler.transform(x_validation_data[numeric_features])

    # Testing
    x_testing_normalized = x_testing_data.copy()
    x_testing_normalized[numeric_features] = scaler.transform(x_testing_data[numeric_features])

    return x_training_data_normalized, x_validation_normalized, x_testing_normalized, y_training_data, y_validation_data, y_testing_data

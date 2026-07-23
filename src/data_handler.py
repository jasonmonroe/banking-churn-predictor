# src/data_handler.py

# Vendor Libraries
import pandas as pd
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Local Libraries
from src.constants import (
    CATEGORICAL_COLS,
    DATA_FILE_PATH,
    IRRELEVANT_COLS,
    SEED,
    TARGET_COL,
    TESTING_SPLIT,
    VALIDATION_SPLIT,
)

class DataHandler:
    def __init__(self):
        self._scaler = MinMaxScaler()

        # Load and set raw data.  We will export the dataset later.
        self.data = self._load(DATA_FILE_PATH).copy()
        self.filtered_data = self._filter(self.data.copy())
        self.dataset = self._get(self.filtered_data.copy())

    def _load(self, filepath: str) -> TextFileReader | pd.DataFrame:
        return pd.read_csv(filepath)

    def _get(self, data: pd.DataFrame) -> dict:
        """
        Returns dataset
        :param data:
        :return:
        """

        # Split data
        dataset = self._split(data)

        # Scale and normalize data
        dataset = self._normalize(dataset)

        return dataset

    def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
        # Drop  irrelevant columns
        data = data.drop(columns=IRRELEVANT_COLS, errors='ignore')

        # Generate dummy variables: This handles dropping original cols and concentration automatically
        data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=True)

        return data

    def _normalize(self, dataset: dict) -> dict:
        """
        Scales data by normalizing the dataset based on split data (training, validation, testing)

        Checks if we have training data, then returns columns that are numerical and to be converted to normalized data.
        Normalize data logic

        Training - create x_train_norm with scaler.fit_transform()
        Validation - create x_val_norm with scaler.transform()
        Testing - create x_test_norm with scaler.transform()

        :param dataset:
        :return:
        """

        if dataset.get("x_train") is None:
            raise ValueError("⚠ Error: Data must be split before attempting to fit model.")

        x_train = dataset.get("x_train")
        x_val = dataset.get("x_val")
        x_test = dataset.get("x_test")

        # Identify numeric columns ONCE from the training data
        numeric_features = x_train.select_dtypes(include=["number"]).columns

        # Normalize Training
        x_train_norm = x_train.copy()
        x_train_norm[numeric_features] = self._scaler.fit_transform(x_train[numeric_features])

        # Normalize Validation
        x_val_norm = x_val.copy()
        x_val_norm[numeric_features] = self._scaler.transform(x_val[numeric_features])

        # Normalize Testing
        x_test_norm = x_test.copy()
        x_test_norm[numeric_features] = self._scaler.transform(x_test[numeric_features])

        # Add back to the dataset
        dataset["x_train_norm"] = x_train_norm
        dataset["x_val_norm"] = x_val_norm
        dataset["x_test_norm"] = x_test_norm

        return dataset

    def describe(self) -> None:
        print("# --- 📊 Describe Data 📊 --- #")
        print(self.data.head())
        print(self.data.tail())
        print(self.data.shape)
        print(self.data.info())
        print(self.data.describe().T)

        # Checking the dtypes of the variables in the data
        print(self.data.dtypes)

        # Find any missing values
        print(f"Total rows with missing values: {self.data.isnull().sum()}")

    def _split(self, features: pd.DataFrame ) -> dict:

        # Ensure features is a numpy array for efficient slicing and processing
        target = features[TARGET_COL]

        # Drop target column from independent variables
        features = features.drop(TARGET_COL, axis=1)

        # Defensive Assertion Check: Ensure absolute size alignment
        assert len(features) == len(target), f"🚩Data length mismatch! Features: {len(features)}, Labels: {len(target)}"

        # --- Split data into 80% Training and 20% Temporary Data
        x_train, x_temp, y_train, y_temp = train_test_split(
            features,
            target,
            test_size=TESTING_SPLIT,
            random_state=SEED,
            stratify=target
        )

        # --- Then take the remaining temporary data 20% and split in half --- #
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=VALIDATION_SPLIT,
            random_state=SEED,
            stratify=y_temp,
        )

        print("\n# --- (Split) Data Shapes --- #")
        print(f"Shape of X Training: {x_train.shape}")
        print(f"Shape of Y Training: {y_train.shape}")
        print(f"Shape of X Validation: {x_val.shape}")
        print(f"Shape of Y Validation: {y_val.shape}")
        print(f"Shape of X Testing: {x_test.shape}")
        print(f"Shape of Y Testing: {y_test.shape}")

        print("\n--- (Split) Data Types ---")
        print(f"Data type of X Training: {x_train.dtypes}")
        print(f"Data type of Y Training: {y_train.dtypes}")

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test
        }

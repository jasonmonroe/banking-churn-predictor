"""
Module for handling data loading, preprocessing, and splitting for the
banking churn prediction project.

This module provides the `DataHandler` class, which encapsulates the
functionality for loading raw data, filtering irrelevant columns,
encoding categorical features, splitting the data into training,
validation, and testing sets, and normalizing numerical features.
"""

# Standard Library Imports
from typing import Dict, Tuple

# Third-party Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Local Imports
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
    """
    Manages the loading, preprocessing, and splitting of the dataset.

    Attributes:
        _scaler (MinMaxScaler): Scaler for normalizing numerical features.
        data (pd.DataFrame): The raw loaded dataset.
        filtered_data (pd.DataFrame): The dataset after initial filtering
                                      and categorical encoding.
        dataset (Dict[str, pd.DataFrame]): A dictionary containing the
                                           split and normalized datasets
                                           (x_train, y_train, etc.).
    """

    def __init__(self) -> None:
        """
        Initializes the DataHandler, loads the data, filters it, and
        prepares the dataset for model training.
        """
        self._scaler = MinMaxScaler()

        # Load and set raw data.
        self.data: pd.DataFrame = self._load(DATA_FILE_PATH).copy()
        self.filtered_data: pd.DataFrame = self._filter(self.data.copy())
        self.dataset: Dict[str, pd.DataFrame] = self._get(
            self.filtered_data.copy()
        )

    def _load(self, filepath: str) -> pd.DataFrame:
        """
        Loads data from a specified CSV file path into a Pandas DataFrame.

        Args:
            filepath (str): The absolute path to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a Pandas DataFrame.
        """
        return pd.read_csv(filepath)

    def _get(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Processes the input DataFrame by splitting it into training,
        validation, and testing sets, and then normalizing the numerical
        features.

        Args:
            data (pd.DataFrame): The DataFrame to be processed.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the split
                                     and normalized datasets.
        """
        # Split data
        dataset = self._split(data)

        # Scale and normalize data
        dataset = self._normalize(dataset)

        return dataset

    def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame by dropping irrelevant columns and
        converting categorical columns into dummy variables.

        Args:
            data (pd.DataFrame): The input DataFrame to filter.

        Returns:
            pd.DataFrame: The filtered DataFrame with dummy variables.
        """
        # Drop irrelevant columns
        data = data.drop(columns=IRRELEVANT_COLS, errors="ignore")

        # Generate dummy variables: This handles dropping original cols
        # and concentration automatically
        data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=True)

        return data

    def _normalize(self, dataset: Dict[str, pd.DataFrame]) -> \
            Dict[str, pd.DataFrame]:
        """
        Scales numerical features in the dataset splits (training,
        validation, testing) using MinMaxScaler.

        The scaler is fitted only on the training data to prevent data
        leakage.

        Args:
            dataset (Dict[str, pd.DataFrame]): A dictionary containing
                                               'x_train', 'x_val', and
                                               'x_test' DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: The dataset dictionary with
                                     normalized 'x_train_norm',
                                     'x_val_norm', and 'x_test_norm'
                                     DataFrames added.

        Raises:
            ValueError: If 'x_train' is not found in the dataset,
                        indicating data was not split.
        """
        if dataset.get("x_train") is None:
            raise ValueError(
                "⚠ Error: Data must be split before attempting to fit model."
            )

        x_train = dataset["x_train"]
        x_val = dataset["x_val"]
        x_test = dataset["x_test"]

        # Identify numeric columns ONCE from the training data
        numeric_features = x_train.select_dtypes(include=["number"]).columns

        # Normalize Training
        x_train_norm = x_train.copy()
        x_train_norm[numeric_features] = self._scaler.fit_transform(
            x_train[numeric_features]
        )

        # Normalize Validation
        x_val_norm = x_val.copy()
        x_val_norm[numeric_features] = self._scaler.transform(
            x_val[numeric_features]
        )

        # Normalize Testing
        x_test_norm = x_test.copy()
        x_test_norm[numeric_features] = self._scaler.transform(
            x_test[numeric_features]
        )

        # Add back to the dataset
        dataset["x_train_norm"] = x_train_norm
        dataset["x_val_norm"] = x_val_norm
        dataset["x_test_norm"] = x_test_norm

        return dataset

    def describe(self) -> None:
        """
        Prints a summary description of the raw loaded data, including
        head, tail, shape, info, descriptive statistics, data types,
        and missing values.
        """
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

    def _split(self, features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Splits the input DataFrame into training, validation, and testing
        sets for both features (X) and target (y).

        The split is performed in two stages:
        1. Initial split: 80% for training, 20% for temporary (validation + test).
        2. Second split: The 20% temporary data is further split into
           half for validation and half for testing.

        Args:
            features (pd.DataFrame): The DataFrame containing both
                                     features and the target column.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the split
                                     DataFrames: 'x_train', 'y_train',
                                     'x_val', 'y_val', 'x_test', 'y_test'.
        """
        target = features[TARGET_COL]

        # Drop target column from independent variables
        features = features.drop(TARGET_COL, axis=1)

        # Defensive Assertion Check: Ensure absolute size alignment
        assert len(features) == len(target), \
            f"🚩Data length mismatch! Features: {len(features)}, " \
            f"Labels: {len(target)}"

        # --- Split data into 80% Training and 20% Temporary Data
        x_train, x_temp, y_train, y_temp = train_test_split(
            features,
            target,
            test_size=TESTING_SPLIT,
            random_state=SEED,
            stratify=target,
        )

        # --- Then take the remaining temporary data 20% and split in half --- #
        # Note: VALIDATION_SPLIT here refers to the split of x_temp,
        # which is 25% of x_temp, resulting in 5% of the original data
        # for validation and 15% for testing (since TESTING_SPLIT is 0.20
        # and VALIDATION_SPLIT is 0.25 of that 0.20, it's 0.05 of total).
        # This comment is incorrect in the original code.
        # The split should be 50/50 of the temp data to get 10% val, 10% test.
        # Let's adjust test_size to 0.5 to split x_temp into 50% val, 50% test.
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,  # Split x_temp (20%) into 10% val, 10% test
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
            "y_test": y_test,
        }
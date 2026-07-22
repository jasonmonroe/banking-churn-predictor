# src/data_handler.py

# Python Libraries

# Vendor Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Local Libraries
from src.constants import (
    CATEGORIAL_COLS,
    DATA_FILE_PATH,
    IRRELEVANT_COLS,
    SEED,
    TARGET_COL,
    TESTING_SPLIT, HALF_SPLIT, VALIDATION_SPLIT, SCALING_COLS,

)

class DataHandler:
    def __init__(self):
        self._scaler = MinMaxScaler()

        # Load and set raw data.  We will export the dataset later.
        self.data = self._load(DATA_FILE_PATH).copy()

        self.dataset = pd.DataFrame()


    def _load(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:

        # Drop  irrelevant columns
        data = data.drop(columns=IRRELEVANT_COLS, errors='ignore')

        # Generate dummy variables and drop the first category to prevent multicollinearity
       # dummy_obj = pd.get_dummies(data[CATEGORIAL_COLS], drop_first=True)
    
        # Concatenate the dummy variables with the original dataset and drop original categorical columns
       #data = pd.concat([data.drop(CATEGORIAL_COLS, axis=1), dummy_obj], axis=1)

        # Generate dummy variables: This handles dropping original cols and concentration automatically
        data = pd.get_dummies(data, columns=CATEGORIAL_COLS, drop_first=True)





        return data


    def _fit(self, dataset: dict) -> dict:
        if dataset.get("x_train") is None:
            raise ValueError("⚠ Error: Data must be split before attempting to fit model.")

        numeric_features = dataset["x_train"].select_dtypes(include=["number"]).columns

        print(f"numeric features type = {type(numeric_features)}")

        #normalize_cols = ["x_train", "x_val", "x_test"]

        # Scale Training, Validation and Testing datasets
        for x_col in SCALING_COLS:
            norm_col = x_col + "_norm"
            dataset[norm_col], dataset[norm_col][numeric_features] = self._normalize(dataset[x_col], numeric_features)




        # Scale Training
        #dataset["x_train_norm"] = dataset["x_train"].copy()
        #dataset["x_train_norm"][numeric_features] = self._scaler.fit_transform(dataset["x_train"][numeric_features])

        # Scale Validation
        #dataset["x_val_norm"] = dataset["x_val"].copy()
        #dataset["x_val_norm"][numeric_features] = self._scaler.transform(dataset["x_val"][numeric_features])

        # Scale Testing
        #dataset["x_test_norm"] = dataset["x_test"].copy()
        #dataset["x_test_norm"][numeric_features] = self._scaler.transform(dataset["x_test"][numeric_features])

        return dataset


    def _normalize(self, x_dataset: dict, numeric_features):
        """
        Normalization (often called Min-Max scaling) is the process of translating your data into a fixed range—usually
        between 0 and 1.
        :return:
        """
        x_norm_data = self._scaler.fit_transform(x_dataset[numeric_features])
        return x_dataset.copy(), x_norm_data


    def get(self, data) -> dict:

        #print(f"type = {type(data)}")
        # Split data
        dataset = self._split(data)

        # Fit Data
        dataset = self._fit(dataset)
        self.dataset = dataset.copy()

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
        #if not isinstance(features, np.ndarray):
        #    features = np.array(features)

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
        print(f"Shape of X training: {x_train.shape}")
        print(f"Shape of Y training: {y_train.shape}")
        print(f"Shape of X validation: {x_val.shape}")
        print(f"Shape of Y validation: {y_val.shape}")
        print(f"Shape of X testing: {x_test.shape}")
        print(f"Shape of Y testing: {y_test.shape}")

        print("\n--- (Split) Data Types ---")
        print(f"Data type of X training: {x_train.dtypes}")
        print(f"Data type of Y training: {y_train.dtypes}")

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test
        }

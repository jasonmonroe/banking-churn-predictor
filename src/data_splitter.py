import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import src.constants as const

class SplitData:
    def __init__(self, data: pd.DataFrame):
        self.scaler = MinMaxScaler()
        self.data = data

        # Initialize as None to make logic checks easier
        self.temp = self.init_data()
        self.training = self.init_data()
        self.testing = self.init_data()
        self.validation = self.init_data()

        self.independent_vars = self.data.drop(const.TARGET_COL, axis=1)
        self.dependent_var = self.data[const.TARGET_COL]

        self.get_all_data()

    def init_data(self) -> dict:
        return {'x': None, 'y': None, 'normalized': None}

    def first_split(self):
        x_temp, x_test, y_temp, y_test = train_test_split(
            self.independent_vars,
            self.dependent_var,
            test_size=const.TESTING_SPLIT,
            random_state=const.SEED,
            stratify=self.dependent_var
        )
        self.temp['x'], self.testing['x'] = x_temp, x_test
        self.temp['y'], self.testing['y'] = y_temp, y_test

    def second_split(self):
        # Error handling: ensure the first split happened
        if self.temp['x'] is None:
            raise ValueError("Must call first_split() before second_split()")

        x_train, x_val, y_train, y_val = train_test_split(
            self.temp['x'],
            self.temp['y'],
            test_size=const.VALIDATION_SPLIT,
            random_state=const.SEED,
            stratify=self.temp['y']
        )
        self.training['x'], self.validation['x'] = x_train, x_val
        self.training['y'], self.validation['y'] = y_train, y_val

    def fit(self):
        # Ensure data exists before scaling
        if self.training['x'] is None:
            raise ValueError("Must split data before calling fit()")

        numeric_features = self.training['x'].select_dtypes(include=['number']).columns

        # Scale Training
        self.training['normalized'] = self.training['x'].copy()
        self.training['normalized'][numeric_features] = self.scaler.fit_transform(self.training['x'][numeric_features])

        # Scale Validation
        self.validation['normalized'] = self.validation['x'].copy()
        self.validation['normalized'][numeric_features] = self.scaler.transform(self.validation['x'][numeric_features])

        # Scale Testing
        self.testing['normalized'] = self.testing['x'].copy()
        self.testing['normalized'][numeric_features] = self.scaler.transform(self.testing['x'][numeric_features])

    def get_all_data(self):
        """Helper to run the full pipeline in order"""
        self.first_split()
        self.second_split()
        self.fit()
        return self.training, self.validation, self.testing

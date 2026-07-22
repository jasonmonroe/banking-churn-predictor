# models/smote_model.py

# Vendor Libraries
from imblearn.over_sampling import SMOTE

# Local Libraries
from src.constants import SEED


class SmoteModel:
    """
    SMOTE (Synthetic Minority Over-sampling Technique)
    SMOTE creates new synthetic examples by looking at the "distance" between existing minority points
    (K-Nearest Neighbors).
    """

    def __init__(self, dataset: dict) -> None:
        self.x = None
        self.y = None

        self.model = self._create()
        self.x, self.y = self._fit(dataset)

        # Check the shapes
        print(f"Shape of X_smote: {self.x.shape}")
        print(f"Shape of y_smote: {self.y.shape}")

    def _create(self) -> SMOTE:
        return SMOTE(random_state=SEED)

    def _fit(self, dataset: dict) -> tuple:
        """
        Fit the SMOTE Model
        :param dataset:
        :return: tuple
        """
        x_data = dataset["x_train_norm"]
        y_data = dataset["y_train"]

        return self.model.fit_resample(x_data, y_data)

# models/smote_model.py

# Vendor Libraries
from imblearn.over_sampling import SMOTE

# Local Libraries
from src.constants import SEED, SMOTE_SAMPLING_STRATEGY, SMOTE_K_NEIGHBORS


class SmoteModel:
    """
    SMOTE (Synthetic Minority Over-sampling Technique)
    SMOTE creates new synthetic examples by looking at the "distance" between existing minority points
    (K-Nearest Neighbors).
    https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    """

    def __init__(self, dataset: dict) -> None:
        self.x = None
        self.y = None

        self.model = self._create()
        self.x, self.y = self._fit(dataset)

        # Check the shapes
        print("\n# --- Loading SMOTE Model --- #")
        print(f"Shape of x_smote: {self.x.shape}")
        print(f"Shape of y_smote: {self.y.shape}")

    def _create(self) -> SMOTE:
        return SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=SEED
        )

    def _fit(self, dataset: dict) -> tuple:
        """
        Fit the SMOTE Model
        :param dataset:
        :return: tuple
        """

        return self.model.fit_resample(dataset["x_train_norm"], dataset["y_train"])

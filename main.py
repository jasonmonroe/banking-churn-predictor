"""
Main entry point for the Bank Churn Predictor application.

This script orchestrates the entire machine learning pipeline, including
data loading, exploratory data analysis (EDA), model training for various
neural network architectures (with and without SMOTE and Dropout),
model comparison, and final customer churn prediction.

The application supports command-line arguments to control which models
are run and whether EDA is performed.
"""

__author__ = "Jason Monroe (jason@jasonmonroe.com)"
__copyright__ = "Copyright © 2011-2026 Monroe Labs"
__date__ = "2024-11-21"
__version__ = "1.1.0"

# Standard Library Imports
import random
import sys
import warnings  # To suppress warnings
from typing import Dict, List, Tuple, Union

# Third-party Imports
import numpy as np
import pandas as pd
import tensorflow as tf

# Local Imports
from models.adam_dropout_model import AdamDropoutModel
from models.adam_model import AdamModel
from models.adam_smote_dropout_model import AdamSmoteDropoutModel
from models.adam_smote_model import AdamSmoteModel
from models.sgd_model import SGDModel
from models.sgd_smote_model import SGDSmoteModel
from models.smote_model import SmoteModel
from src.constants import ARG_PARAMS, CUSTOMER_CHURN_PROB_THRESHOLD, PEP8_LINE_LEN, SEED
from src.data_handler import DataHandler
from src.eda import (
    show_classification_report,
    show_correlation_matrix,
    show_plot_distributions,
    show_salary_barplot_visualization,
    show_visualizations,
)
from src.model_perf import ModelPerformance
from src.utils import format_performance, get_run_id, show_banner, show_timer, start_timer


def run_data_pipeline(args: Dict[str, bool]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Executes the data loading and optional Exploratory Data Analysis (EDA) pipeline.

    Initializes random seeds for reproducibility, loads data using DataHandler,
    and performs EDA if the 'eda' flag is set in the arguments.

    Args:
        args (Dict[str, bool]): A dictionary of command-line arguments,
                                including an 'eda' flag.

    Returns:
        Tuple[Dict[str, pd.DataFrame], pd.DataFrame]: A tuple containing:
            - dataset_df (Dict[str, pd.DataFrame]): Processed and split
                                                    dataframes (train, val, test).
            - raw_df (pd.DataFrame): The original raw dataframe.
    """
    # Seed data with random integer for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    subtitles: List[str] = ["We keep your banking customers from leaving!"]
    show_banner("QUANTUM BANK", subtitles, center_subtitle_text=True)

    # Create data handler and return datasets
    data_handler = DataHandler()
    raw_df: pd.DataFrame = data_handler.data
    dataset_df: Dict[str, pd.DataFrame] = data_handler.dataset
    filtered_df: pd.DataFrame = data_handler.filtered_data

    if args.get("eda"):
        data_handler.describe()
        show_visualizations(raw_df)
        show_salary_barplot_visualization(raw_df)
        show_plot_distributions(raw_df)
        show_correlation_matrix(filtered_df)

    return dataset_df, raw_df


def run_model_pipeline(args: Dict[str, bool], dataset: Dict[str, pd.DataFrame]) -> List[object]:
    """
    Runs the machine learning model training pipeline based on command-line arguments.

    Initializes and trains selected models (SGD, Adam, with/without Dropout,
    with/without SMOTE) and collects their instances.

    Args:
        args (Dict[str, bool]): A dictionary of command-line arguments,
                                including model selection flags.
        dataset (Dict[str, pd.DataFrame]): The processed and split dataset.

    Returns:
        List[object]: A list of trained model instances.
    """
    smote_model_instance = SmoteModel(dataset)
    x_smote, y_smote = smote_model_instance.x, smote_model_instance.y

    models: List[object] = []

    # 1) Building Neural Network Model (Stochastic gradient descent)
    if args.get("all") or args.get("model:sgd"):
        sgd_model = SGDModel(dataset)
        sgd_model.run()
        models.append(sgd_model)

    # 2) Building Neural Network Model w/ Adam Optimizer
    if args.get("all") or args.get("model:adam"):
        adam_model = AdamModel(dataset)
        adam_model.run()
        models.append(adam_model)

    # 3) Build Adam Optimized Model with Dropout
    if args.get("all") or args.get("model:adam-dropout"):
        adam_dropout_model = AdamDropoutModel(dataset)
        adam_dropout_model.run()
        models.append(adam_dropout_model)

    # --- SMOTE Models --- #

    # 4) Build Neural Network (SGD with SMOTE)
    if args.get("all") or args.get("model:sgd-smote"):
        sgd_smote_model = SGDSmoteModel(dataset)
        sgd_smote_model.run(x_smote, y_smote)

        # Generate SMOTE Classification Report
        show_banner(
            sgd_smote_model.title,
            "Classification Report by Class and Summary",
            center_subtitle_text=True,
        )
        show_classification_report(sgd_smote_model.y_test, sgd_smote_model.y_predictor)

        models.append(sgd_smote_model)

    # 5) Build Neural Network (Adam with SMOTE)
    if args.get("all") or args.get("model:adam-smote"):
        adam_smote_model = AdamSmoteModel(dataset)
        adam_smote_model.run(x_smote, y_smote)

        # Generate SMOTE Classification Report
        show_banner(
            adam_smote_model.title,
            "Classification Report by Class and Summary",
            center_subtitle_text=True,
        )
        show_classification_report(adam_smote_model.y_test, adam_smote_model.y_predictor)

        models.append(adam_smote_model)

    # 6) Build Neural Network Adam and Dropout with SMOTE
    if args.get("all") or args.get("model:adam-smote-dropout"):
        adam_smote_dropout_model = AdamSmoteDropoutModel(dataset)
        adam_smote_dropout_model.run(x_smote, y_smote)

        # Generate SMOTE Classification Report
        show_banner(
            adam_smote_dropout_model.title,
            "Classification Report by Class and Summary",
            center_subtitle_text=True,
        )
        show_classification_report(
            adam_smote_dropout_model.y_test, adam_smote_dropout_model.y_predictor
        )

        models.append(adam_smote_dropout_model)

    return models


def run_model_comparison_pipeline(
    models: List[object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares the performance of two or more models.

    Args:
        models (List[object]): A list of trained model instances.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - model_comparison_train_perfs_matrix (pd.DataFrame):
              Comparison matrix of training performances.
            - model_comparison_val_perfs_matrix (pd.DataFrame):
              Comparison matrix of validation performances.
            Returns empty DataFrames if fewer than two models are provided.
    """
    # There must be at least two models to run a comparison.
    if len(models) > 1:
        model_comparison_train_perfs_matrix, model_comparison_val_perfs_matrix = (
            ModelPerformance.create_comparisons(models)
        )

        formatted_train_perfs: List[str] = format_performance(
            model_comparison_train_perfs_matrix
        )
        formatted_val_perfs: List[str] = format_performance(
            model_comparison_val_perfs_matrix
        )

        # --- Display Comparisons --- #
        show_banner("Comparison Model Training Performances", formatted_train_perfs)
        show_banner("Comparison Model Validation Performances", formatted_val_perfs)

        return model_comparison_train_perfs_matrix, model_comparison_val_perfs_matrix

    else:
        print("⚠ Warning: A minimum of two models are required to make a comparison.")
        return pd.DataFrame(), pd.DataFrame()


def run_customer_churn_results(final_model: object, raw_csv_data: pd.DataFrame) -> List[str]:
    """
    Generates and formats customer churn prediction results for display.

    Args:
        final_model (object): The best-performing trained model instance.
        raw_csv_data (pd.DataFrame): The original raw dataframe to extract
                                     customer information.

    Returns:
        List[str]: A list of formatted strings representing the churn
                   prediction report.
    """
    total_rows: int = raw_csv_data.shape[0]

    # Get all predictions
    predictions: np.ndarray = final_model.model.predict(final_model.x_test_norm, verbose=0)
    customer_count: int = len(predictions)

    subtitles: List[str] = [
        "Will the customer leave the bank within the next six months❓",
        f"There are {total_rows} rows of data to process.",
        f"This bank has {customer_count} customers.  Customer Churn Rate is "
        f"{CUSTOMER_CHURN_PROB_THRESHOLD * 100:.1f}%",
    ]

    customer_churn_ctr: int = 0
    full_line: str = "-" * (PEP8_LINE_LEN - 4)
    header: str = "Row | Customer ID | Probability | Status"
    results: List[str] = [full_line, header, full_line]

    for idx, probability in enumerate(predictions):
        prob: float = probability[0]
        is_churning: bool = prob > CUSTOMER_CHURN_PROB_THRESHOLD

        row_number: int = raw_csv_data.iloc[idx]["row_number"]
        customer_id: int = raw_csv_data.iloc[idx]["customer_id"]
        probability_pct_text: str = f"{prob * 100:.1f}%"
        status: str = "❌" if is_churning else "✅"

        if is_churning:
            customer_churn_ctr += 1

        line: str = (
            f"{row_number:<3} | {customer_id:<11} | {probability_pct_text:<11} | {status}"
        )
        results.append(line)

    # Output Final results
    customer_churn_pct: float = (customer_churn_ctr / customer_count) * 100

    results.append(full_line)
    results.append("Report:")
    results.append(f"{customer_churn_ctr} customers are at risk of churning.")
    results.append(f"That's {customer_churn_pct:.1f}% of our customers!")

    subtitles.extend(results)
    return subtitles


def run_main_pipeline(args: Dict[str, bool]) -> None:
    """
    Orchestrates the main machine learning pipeline.

    This function drives the entire process from data preparation and
    model training to comparison and final evaluation.

    Args:
        args (Dict[str, bool]): A dictionary of command-line arguments.
    """
    dataset, raw_csv_data = run_data_pipeline(args)
    models = run_model_pipeline(args, dataset)

    if not models:
        raise ValueError("🚩No models found to run!")

    if len(models) == 1:
        print(
            "Only 1 model was run. No comparison can be made, but its "
            "evaluation will be displayed."
        )
        model = models[0]
        model_data: List[str] = format_performance(model.model_perf.data)
        show_banner(f"Final Model (Only): {model.title}", model_data)

        # Explicitly run evaluation for the single model
        model.evaluate()

        # Predict customer churn using the single model
        customer_churn_results: List[str] = run_customer_churn_results(model, raw_csv_data)
        show_banner("Quantum Bank Churn Predictor Results".upper(), customer_churn_results)

        return None

    # --- Multiple Models --- #

    # 1. Compare validation matrices to find the champion
    train_matrix, val_matrix = run_model_comparison_pipeline(models)
    best_model_name: str = ModelPerformance.get_best_model_name(val_matrix, train_matrix)
    show_banner("Best Performing Model (Validation)", [f"{best_model_name}"], center_subtitle_text=True)

    # 2. Grab the winning model object
    final_model: object = ModelPerformance.get_final_model(best_model_name, models)

    # 3. Run the official .evaluate() test on the unseen test set *only* for this winner
    final_model.evaluate()

    # 4. Predict customer churn using the final model
    customer_churn_results: List[str] = run_customer_churn_results(final_model, raw_csv_data)
    show_banner("Quantum Bank Churn Predictor Results".upper(), customer_churn_results)

    return None


def _parse_args(command_line_args: List[str]) -> Dict[str, bool]:
    """
    Parses command-line arguments provided to the script.

    Args:
        command_line_args (List[str]): A list of command-line arguments
                                       (e.g., `sys.argv[1:]`).

    Returns:
        Dict[str, bool]: A dictionary where keys are argument names (without
                         '--') and values are booleans indicating presence.
    """
    args: Dict[str, bool] = {
        arg.strip("--"): (arg in command_line_args) for arg in ARG_PARAMS
    }

    args["all"] = _check_arg_all(args)

    return args


def _check_arg_all(args: Dict[str, bool]) -> bool:
    """
    Determines the effective state of the 'all' flag.

    If any specific model flags are set, 'all' is implicitly False unless
    explicitly set. If no model flags are set, 'all' defaults to True.

    Args:
        args (Dict[str, bool]): The parsed command-line arguments.

    Returns:
        bool: The effective boolean value for the 'all' flag.
    """
    # Check if any model-specific flags are True
    any_model_selected: bool = any(
        "model" in key and value is True for key, value in args.items()
    )

    # If 'all' is explicitly requested, it overrides individual model selections
    if args.get("all"):
        return True
    # If no specific models are selected, and 'all' wasn't explicitly False,
    # then default to running all models.
    elif not any_model_selected:
        return True
    # Otherwise, if specific models were selected, 'all' is False.
    else:
        return False


# --- Start Program --- #
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    prog_start_time: float = start_timer()
    run_id: str = get_run_id()
    print(f"\n----- ⏱️START RUN ID: {run_id} ⏱️-----")

    args: Dict[str, bool] = _parse_args(sys.argv[1:])
    run_main_pipeline(args)
    show_timer(prog_start_time)

    print(f"\n----- ⏱️ END RUN ID: {run_id} ⏱️-----")

# --- End of Program ---
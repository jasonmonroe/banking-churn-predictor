# main.py

"""
+---------------------------------------------------------------------------+
|                           BANK CHURN PREDICTOR                             |
+---------------------------------------------------------------------------+
| In service industries, particularly banking, mitigating customer churn is |
| a critical business imperative. Understanding the factors contributing to |
| customer attrition is essential for sustained profitability. We propose   |
| developing a Machine Learning model to rigorously quantify the feature    |
| importance of various service attributes (e.g., transaction speed, fee    |
| structures, customer support quality) on a customer's likelihood to       |
| terminate their service agreement. The resulting prioritized list of      |
| influential factors will guide management in making data-driven decisions |
| to refine the service offering and optimize retention strategies.         |
+---------------------------------------------------------------------------+
"""

__author__ = "Jason Monroe (jason@jasonmonroe.com)"
__copyright__ = "Copyright © 2011-2026 Monroe Labs"
__date__ = "2024-11-21"
__version__ = "1.1.0"

# Python Libraries
import numpy as np
import random
import sys
import warnings  # To suppress warnings

# Vendor Libraries
import pandas as pd
import tensorflow as tf
from src.data_handler import DataHandler
from src.model_perf import ModelPerformance

# Local Libraries
from models.adam_dropout_model import AdamDropoutModel
from models.adam_model import AdamModel
from models.adam_smote_dropout_model import AdamSmoteDropoutModel
from models.adam_smote_model import AdamSmoteModel
from models.sgd_model import SGDModel
from models.sgd_smote_model import SGDSmoteModel
from models.smote_model import SmoteModel
from src.constants import ARG_PARAMS, SEED, CUSTOMER_CHURN_PROB_THRESHOLD, LEARNING_RATE
from src.utils import show_banner, start_timer, get_run_id, format_performance
from src.eda import (
    show_salary_barplot_visualization,
    show_plot_distributions,
    show_correlation_matrix,
    show_visualizations, show_classification_report
)


def run_data_pipeline(args: dict) -> tuple[dict, pd.DataFrame]:
    # Seed data with random integer
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    subtitles = ["We keep your banking customers from leaving!"]
    show_banner("BANK CHURN PREDICTIONS", subtitles, center_subtitle_text=True)

    data_handler = DataHandler()

    # load data
    data = data_handler.data
    df = data.copy()

    if args.get("eda"):
        data_handler.describe()
        show_visualizations(df)
        show_salary_barplot_visualization(df)
        show_plot_distributions(df)

    # Clean the data
    df = data_handler.filter(df)

    if args.get("eda"):
        show_correlation_matrix(df)

    dataset = data_handler.get(df)

    return dataset, data


def run_model_pipeline(args: dict, dataset: dict) -> list:
    """
    Return all models requested in the command line.
    :param args:
    :param dataset:
    :return:
    """

    print("\n# --- LOADING SMOTE MODEL --- #")

    smote_model = SmoteModel(dataset)
    x_smote, y_smote = smote_model.x, smote_model.y

    models = []

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
        show_banner(sgd_smote_model.title, "Classification Report")
        show_classification_report(sgd_smote_model.y_test, sgd_smote_model.y_predictor)

        models.append(sgd_smote_model)

    # 5) Build Neural Network (Adam with SMOTE)
    if args.get("all") or args.get("model:adam-smote"):
        adam_smote_model = AdamSmoteModel(dataset)
        adam_smote_model.run(x_smote, y_smote)
        models.append(adam_smote_model)

    # 6) Build Neural Network Adam and Dropout with SMOTE
    if args.get("all") or args.get("model:adam-smote-dropout"):
        adam_smote_dropout_model = AdamSmoteDropoutModel(dataset)
        adam_smote_dropout_model.run()

        print(f"Run {adam_smote_dropout_model.title} again but with SMOTE data...")
        adam_smote_dropout_model.run(x_smote, y_smote)

        models.append(adam_smote_dropout_model)

    return models


def run_model_comparison_pipeline(models: list) -> tuple[pd.DataFrame, pd.DataFrame]:

    # There must be at least two models to run a comparison.
    if len(models) > 1:
        model_comparison_train_perfs_matrix, model_comparison_val_perfs_matrix = ModelPerformance.create_comparisons(models)

        formatted_train_perfs = format_performance(model_comparison_train_perfs_matrix)
        formatted_val_perfs = format_performance(model_comparison_val_perfs_matrix)

        # --- Display Comparisons --- #
        #show_banner("Comparison Model Training Performances".upper(), [model_comparison_train_perfs_matrix.to_string(float_format="%.4f")])
        #show_banner("Comparison Model Validation Performances".upper(), [model_comparison_val_perfs_matrix.to_string(float_format="%.4f")])

        show_banner("Comparison Model Training Performances", formatted_train_perfs)
        show_banner("Comparison Model Validation Performances", formatted_val_perfs)

        return model_comparison_train_perfs_matrix, model_comparison_val_perfs_matrix

    else:
        print("⚠ Warning: A minimum of two models are required to make a comparison.")
        return pd.DataFrame(), pd.DataFrame()


def run_customer_churn_results(final_model, raw_csv_data: pd.DataFrame) -> None:

    customer_count = raw_csv_data.shape[0]
    subtitles = [
        "Will the customer leave the bank within the next six months❓",
        f"This bank has {customer_count} customers.  Customer Churn Rate is {CUSTOMER_CHURN_PROB_THRESHOLD * 100}%",
    ]

    # Get all predictions
    predictions = final_model.model.predict(final_model.x_test_norm, verbose=0)

    header = "Row | Customer ID | Probability | Status"
    results = [header, "-" * len(header)]
    for idx, probability in enumerate(predictions):
        prob = probability[0]
        is_churning = prob > CUSTOMER_CHURN_PROB_THRESHOLD

        row_number = raw_csv_data.iloc[idx]["row_number"]
        customer_id = raw_csv_data.iloc[idx]["customer_id"]
        probability_pct = prob * 100
        probability_pct_text = f"{probability_pct:.1f}%"
        status = "❌" if is_churning else "✅"

        line = f"{row_number:<3} | {customer_id:<11} | {probability_pct_text:<11} | {status}"
        results.append(line)

        # Test only 10
        if idx > 4:
            break

    subtitles.extend(results)
    show_banner("Bank Churn Predictor Results".upper(), subtitles)


def run_main_pipeline(args: dict):
    """
    Runs main pipeline for this project.
    :param args:
    :return:
    """

    dataset, raw_csv_data = run_data_pipeline(args)
    models = run_model_pipeline(args, dataset)
    train_matrix, val_matrix = run_model_comparison_pipeline(models)

    best_model_name = ModelPerformance.get_best_model_name(train_matrix, val_matrix)
    show_banner("Best Performing Model", [f"{best_model_name}"], center_subtitle_text=True)

    # --- Final Test Evaluation --- #
    """
    This is the Real World check using test data in it's evaluation.  Each model inherits the base class that has the 
    entire dataset.  Test data is unused at this point and will be the same for each model.
    """
    test_model_perfs = ModelPerformance.get_test_model_perfs(models)
    final_model = ModelPerformance.get_final_model(test_model_perfs, models)

    subtitles = [
        f"Final Model: {final_model.title}",
        "Performance",
    ]

    for metric_name, series_data in final_model.model_perf.data.items():
        metric_value = series_data.iloc[0]
        metric_value_pct = metric_value * 100
        metric_value_text = f"{metric_value_pct:.4f}%"
        perf_line = f"{metric_name:<10}: {metric_value_text}"
        subtitles.append(perf_line)

    show_banner("Final Evaluation", subtitles)

    # Evaluate customer churn probabilities using the final prediction pipeline.
    run_customer_churn_results(final_model, raw_csv_data)


def _parse_args(command_line_args: list[str]) -> dict:
    """
    Parse command line arguments.

    :param command_line_args:
    :return: dict
    """
    args = {arg.strip("--"): (arg in command_line_args) for arg in ARG_PARAMS}

    args["all"] = _check_arg_all(args)

    return args


def _check_arg_all(args: dict) -> bool:
    """
    Check to see if any models were called in command line.  If so, turn the all flag to False.
    If the `all` flag is referenced, override and set all to true.
    :param args:
    :return:
    """

    return (not any("model" in key and value == True for key, value in args.items())) or args.get("all")


# --- Start Program --- #
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    prog_start_time = start_timer()
    run_id = get_run_id()
    print(f"\n----- ⏱️START RUN ID: {run_id} ⏱️-----")

    args = _parse_args(sys.argv[1:])
    print(f"line 445: args={args}")
    run_main_pipeline(args)

    print(f"\n----- ⏱️ END RUN ID: {run_id} ⏱️-----")

# --- End of Program ---

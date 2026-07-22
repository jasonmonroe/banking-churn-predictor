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


from sklearn.metrics import (
    classification_report,
)
import tensorflow as tf
from src.data_handler import DataHandler
from src.model_perf import ModelPerformance
from tensorflow.keras.optimizers import Adam, SGD

# Local Libraries
from models.adam_dropout_model import AdamDropoutModel
from models.adam_model import AdamModel
from models.adam_smote_dropout_model import AdamSmoteDropoutModel
from models.adam_smote_model import AdamSmoteModel
from models.sgd_model import SGDModel
from models.sgd_smote_model import SGDSmoteModel
from models.smote_model import SmoteModel

from src.constants import ARG_PARAMS, SEED, PREDICTION_PROB_THRESHOLD, LEARNING_RATE

from src.data_splitter import SplitData
from src.data_loader import load_and_clean_data
from src.model_builder import ModelBuilder
from src.preprocessing import preprocessing_data, get_smote
from src.utils import show_banner, seed_script, start_timer, get_run_id
from src.eda import (
    plot_model_performance,
    show_salary_barplot_visualization,
    show_plot_distributions,
    show_correlation_matrix,
    observe_data,
    show_visualizations, show_classification_report
)


def run_data_pipeline(args: dict) -> dict:
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

    return dataset

def run_model_pipeline(args: dict, dataset: dict) -> list:

    models = []

    # Return all models requested in the command line.

    if args.get("all") or args.get("model:sgd"):
        pass
    if args.get("all") or args.get("model:adam"):
        pass

    if args.get("all") or args.get("model:adam-dropout"):
        pass
    if args.get("all") or args.get("model:sgd-smote"):
        pass
    if args.get("all") or args.get("model:adam-smote"):
        pass
    if args.get("all") or args.get("model:adam-smote-dropout"):
        pass



    return models

def run_model_comparison_pipeline(args: dict, models: list) -> tuple[dict, dict]:
    comparison_models = {}

    # There must be at least two models to run a comparison.

    if len(models) > 1:
        pass
    else:
        print("⚠ Warning: A minimum of two models are requred to make a comparison.")
        return {}, {}

def run_main_pipeline(args: dict):

    dataset = run_data_pipeline(args)

    #print(f"dataset[x_train]=dataset{dataset['x_train']}")
    #print(f"dataset[x_train_norm]=dataset{dataset['x_train_norm']}")
    import sys
    #sys.exit(0)

    # --- Build & Train Models --- #
    # Training: x, y, normalized
    # Validation: x, y, normalized
    # Testing: x, y, normalized

    #training_norm = dataset['x_train']
    #training_y = dataset['y_train']
    ##validation_norm = dataset['x_val']
    #validation_y = dataset['y_val']
    #testing_norm = dataset['x_test']
    #testing_y = dataset['y_test']

    # Get feature count for model creation
    #feature_cnt = dataset["x_train_norm"].shape[1]

    models = run_model_pipeline(args, dataset)

    # 1) Building Neural Network Model (Stochastic gradient descent)
    sgd_model = SGDModel(dataset)
    #sgd_model.run()

    #sys.exit(0)

    # 2) Building Neural Network Model w/ Adam Optimizer
    adam_model = AdamModel(dataset)
    #adam_model.run()

    # 3) Build Adam Optimized Model with Dropout
    adam_dropout_model = AdamDropoutModel(dataset)
    #adam_dropout_model.run()

    print("\n# --- LOADING SMOTE MODELS --- #")
    smote_model = SmoteModel(dataset)
    x_smote, y_smote = smote_model.x, smote_model.y
    #
    #x_smote=(12740, 11), y_smote=(12740,)

    # 4) Build Neural Network (SGD with SMOTE)
    sgd_smote_model = SGDSmoteModel(dataset)
    #sgd_smote_model.run(x_smote, y_smote)

    # 5) Generate SMOTE Classification Report
    #show_banner(sgd_smote_model.title, "Classification Report")
    #show_classification_report(sgd_smote_model.y_test, sgd_smote_model.y_predictor)

    # 6) Build Neural Network (Adam with SMOTE)
    adam_smote_model = AdamSmoteModel(dataset)
    #adam_smote_model.run(x_smote, y_smote)

    # Build Neural Network Adam and Dropout with SMOTE
    adam_smote_dropout_model = AdamSmoteDropoutModel(dataset)
    adam_smote_dropout_model.run()
    print(f"# --- Run {adam_smote_dropout_model.title} it again with SMOTE data --- #")
    adam_smote_dropout_model.run(x_smote, y_smote)

    model_comparison_train, model_comparison_val = run_model_comparison_pipeline(args, models)




    # Ignore Below

    # --- Compare Models ---
    model_comparison_titles = [
        sgd_model.title,
        adam_model.title,
        adam_dropout_model.title,
        sgd_smote_model.title,
        adam_smote_model.title,
        adam_smote_dropout_model.title,
    ]

    # --- Training Performance Comparison ---
    model_comparison_train_perfs = pd.concat([
        sgd_model.train_perf,
        adam_model.train_perf,
        adam_dropout_model.train_perf,
        sgd_smote_model.train_perf,
        adam_smote_model.train_perf,
        adam_smote_dropout_model.train_perf
    ])

    # Create Model Comparison Training Performance Matrix
    #model_comparison_train_perfs_matrix = model_comparison_train_perfs
    #model_comparison_train_perfs_matrix.index = model_comparison_titles
    #model_comparison_train_perfs_matrix = model_comparison_train_perfs_matrix.T

    model_comparison_train_perfs_matrix = ModelPerformance.create_matrix(model_comparison_titles, model_comparison_train_perfs)
    
    #model_comparisons_train = {
    #    "titles": model_comparison_titles,
    #    "perfs": model_comparison_perfs_matrix
    #}


    # --- Validation Performance Comparison ---
    model_comparison_val_perfs = pd.concat([
        sgd_model.val_perf,
        adam_model.val_perf,
        adam_dropout_model.val_perf,
        sgd_smote_model.val_perf,
        adam_smote_model.val_perf,
        adam_smote_dropout_model.val_perf
    ])

    # type = <class 'pandas.DataFrame'>
    print(f"model_comparison_val_perfs type = {type(model_comparison_val_perfs)}")

    #model_comparison_val_perfs_matrix = model_comparison_val_perfs
    #model_comparison_val_perfs_matrix.index = model_comparison_titles
    #model_comparison_val_perfs_matrix = model_comparison_val_perfs_matrix.T
    
    model_comparison_val_perfs_matrix = ModelPerformance.create_matrix(model_comparison_titles, model_comparison_val_perfs)

    #model_comparisons_val = {
    #    "titles": model_comparison_titles,
    #    "perfs": model_comparison_perfs_matrix
    #}

    # --- Display Comparisons --- #
    show_banner("Model Training Performances".upper(), [model_comparison_train_perfs_matrix])
    show_banner("Model Validation Performances".upper(), [model_comparison_val_perfs_matrix])


    # --- Final Results --- #
    #final_results = model_comparison_train_perfs_matrix.loc["F1"] - model_comparison_val_perfs_matrix.loc["F1"]
    final_results = ModelPerformance.final_results(model_comparison_train_perfs_matrix, model_comparison_val_perfs_matrix)
    show_banner("Bank Churn Predictor Final Results".upper(), [final_results])

    # @TODO - Question: Where does the testing data that was split from the source_data.csv go? When and where do we
    # utilize it?



def main():
    warnings.filterwarnings('ignore')

    # Set seeds for reproducibility
    seed_script(SEED)

    # Load and clean data
    data = load_and_clean_data(DATA_FILE_PATH)
    print('Data loaded successfully.')

    # --- Observe Data ---
    observe_data(data)

    # --- Show Data Visualizations ---
    show_visualizations(data)

    # Call stacked barplot with the modified DataFrame
    show_salary_barplot_visualization(data)

    show_plot_distributions(data)

    # Drop columns, that are least relevant to the data.
    # Drop columns, ignoring errors if a column does not exist.
    data = data.drop(columns=['row_number', 'customer_id', 'surname'], axis=1, errors='ignore')

    # Show Data Correlation Matrix of Bank Customer Churn
    show_correlation_matrix(data)

    # --- Pre Processing Data ---
    data = preprocessing_data(data)

    # --- Split Data ---
    data_sp = SplitData(data)
    print(f'data_sp {type(data_sp)}')

    # --- Build & Train Models --- #

    training_norm = data_sp.training['normalized']
    training_y = data_sp.training['y']
    validation_norm = data_sp.validation['normalized']
    validation_y = data_sp.validation['y']

    """
    Stochastic Gradient Descent (SGD) is a fundamental optimization algorithm in machine learning that minimizes a loss 
    function by updating model parameters using a single training example or a small subset (mini-batch) at a time. 
    This approach is particularly efficient for large datasets compared to traditional gradient descent, which uses the 
    entire dataset for each update. 
    """
    # 1) Building Neural Network Model (Stochastic gradient descent)
    sgd_model = ModelBuilder('Neural Network (SGD)', 'SGD')
    sgd_model.create_sgd_model(training_norm.shape[1])
    sgd_model_history = sgd_model.build(data_sp)
    plot_model_performance(sgd_model_history, 'accuracy', sgd_model.title)
    plot_model_performance(sgd_model_history, 'loss', sgd_model.title)

    # 2) Building Neural Network Model w/ Adam Optimizer
    adam_model = ModelBuilder('Neural Network (Adam Optimizer)', Adam(learning_rate=LEARNING_RATE))
    adam_model.create_adam_model(training_norm.shape[1])
    adam_model_history = adam_model.build(data_sp)
    #adam_model.evaluate(data_sp)
    plot_model_performance(adam_model_history, 'accuracy', adam_model.title)
    plot_model_performance(adam_model_history, 'loss', adam_model.title)

    # 3) Build Adam Optimized Model with Dropout
    adam_dropout_model = ModelBuilder('Neural Network (Adam and Dropout)', Adam(learning_rate=LEARNING_RATE))
    adam_dropout_model.create_adam_dropout_model(training_norm.shape[1])
    adam_dropout_model_history = adam_dropout_model.build(data_sp)
    #adam_dropout_model.evaluate(data_sp)
    plot_model_performance(adam_dropout_model_history, 'accuracy', adam_dropout_model.title)
    plot_model_performance(adam_dropout_model_history, 'loss', adam_dropout_model.title)

    # -- SMOTE --
    x_smote, y_smote = get_smote(data_sp)
    # ------------

    # Build Neural Network (SGD with SMOTE)
    sgd_smote_model = ModelBuilder('Neural Network (SGD with SMOTE)',
                                   SGD(learning_rate=LEARNING_RATE, momentum=0.9))
    sgd_smote_model.create_smote_model(x_smote.shape[1])
    sgd_smote_model_history = sgd_smote_model.build(data_sp, x_smote, y_smote)

    # Generate Smote Classification Report
    show_banner(sgd_smote_model.title, 'Classification Report')
    y_predictor = (sgd_smote_model.predict(data_sp.testing['normalized']) > PREDICTION_PROB_THRESHOLD).astype(int)
    print(classification_report(data_sp.testing['y'], y_predictor))

    # Uses the correct arguments
    sgd_smote_model.show_model_perf('Smote Training', training_norm, training_y)
    sgd_smote_model.show_model_perf('Smote Validation', validation_norm, validation_y)
    plot_model_performance(sgd_smote_model_history, 'accuracy', sgd_smote_model.title)
    plot_model_performance(sgd_smote_model_history, 'loss', sgd_smote_model.title)

    # Build Neural Network (Adam with SMOTE)
    adam_smote_model = ModelBuilder('Neural Network (Adam with SMOTE)', 'adam')
    adam_smote_model.create_adam_smote_model(x_smote.shape[1])
    adam_smote_model_history = adam_smote_model.build(data_sp, x_smote, y_smote)
    plot_model_performance(adam_smote_model_history, 'accuracy', adam_smote_model.title)
    plot_model_performance(adam_smote_model_history, 'loss', adam_smote_model.title)

    # Build Neural Network Adam and Dropout with SMOTE
    adam_smote_dropout_model = ModelBuilder('Neural Network (Adam and Dropout with SMOTE)',
                                            Adam(learning_rate=LEARNING_RATE))
    adam_smote_dropout_model.create_adam_smote_dropout_model(x_smote.shape[1])
    adam_smote_dropout_model_history = adam_smote_dropout_model.build(data_sp)
    plot_model_performance(adam_smote_dropout_model_history, 'accuracy', adam_smote_dropout_model.title)
    plot_model_performance(adam_smote_dropout_model_history, 'loss', adam_smote_dropout_model.title)

    # --- Compare Models ---
    model_titles = [
        sgd_model.title,
        adam_model.title,
        adam_dropout_model.title,
        sgd_smote_model.title,
        adam_smote_model.title,
        adam_smote_dropout_model.title
    ]

    # Training Performance Comparison

    model_training_perfs = pd.concat([
        sgd_model.get_model_perf(training_norm, training_y),
        adam_model.get_model_perf(training_norm, training_y),
        adam_dropout_model.get_model_perf(training_norm, training_y),
        sgd_smote_model.get_model_perf(x_smote, y_smote),
        adam_smote_model.get_model_perf(training_norm, training_y),
        adam_smote_dropout_model.get_model_perf(training_norm, training_y)
    ])

    model_training_perfs.index = model_titles
    model_training_perfs = model_training_perfs.T

    # Validation Performance Comparison

    model_validation_perfs = pd.concat([
        sgd_model.get_model_perf(validation_norm, validation_y),
        adam_model.get_model_perf(validation_norm, validation_y),
        adam_dropout_model.get_model_perf(validation_norm, validation_y),
        sgd_smote_model.get_model_perf(validation_norm, validation_y),
        adam_smote_model.get_model_perf(validation_norm, validation_y),
        adam_smote_dropout_model.get_model_perf(validation_norm, validation_y)
    ])

    model_validation_perfs.index = model_titles
    model_validation_perfs = model_validation_perfs.T  # Transpose so metrics are columns, models are rows

    # --- Display Performance Comparisons

    # Training
    print(model_training_perfs)

    # Validation
    print(model_validation_perfs)

    # --- FINAL RESULT(S) --- #
    final = model_training_perfs.loc['F1'] - model_validation_perfs.loc['F1']

    show_banner('Final Results')
    print(final)

    # --- End of Main ---


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

    #print(f"line 447: args={args.keys()}")
    # If no individual models are referenced in the command line run them all.

    #return param_args

    #import sys
    #sys.exit(0)
    run_main_pipeline(args)

    print(f"\n----- ⏱️ END RUN ID: {run_id} ⏱️-----")

# --- End of Program ---

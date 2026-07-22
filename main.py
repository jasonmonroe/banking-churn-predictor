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

        print(f"# --- Run {adam_smote_dropout_model.title} it again with SMOTE data --- #")
        adam_smote_dropout_model.run(x_smote, y_smote)

        models.append(adam_smote_dropout_model)

    return models

def run_model_comparison_pipeline(args: dict, models: list) -> tuple[pd.DataFrame, pd.DataFrame]:

    # There must be at least two models to run a comparison.

    if len(models) > 1:
        model_comparison_train_perfs_matrix, model_comparison_val_perfs_matrix = ModelPerformance.create_comparisons(models)

        # --- Display Comparisons --- #
        show_banner("Model Training Performances".upper(), [model_comparison_train_perfs_matrix])
        show_banner("Model Validation Performances".upper(), [model_comparison_val_perfs_matrix])

        return model_comparison_train_perfs_matrix, model_comparison_val_perfs_matrix

    else:
        print("⚠ Warning: A minimum of two models are required to make a comparison.")
        return pd.DataFrame(), pd.DataFrame()

def run_customer_churn_results(final_model, dataset: dict, raw_csv_data: pd.DataFrame):

    subtitles = ["Will the customer leave the bank within the next six months❓"]

    predictions = None
    show_banner("BANK CHURN PREDICTOR RESULTS", subtitles)


def run_customer_churn_results_orig(models: list, best_model_name: str, dataset: dict):
    subtitles = ["Will the customer leave the bank within the next six months❓"]
    show_banner("BANK CHURN PREDICTOR RESULTS", subtitles)

    # 1. Find the best model object from our list
    best_model_obj = next((m for m in models if m.title == best_model_name), models[0])
    
    # 2. Use the test data (or full data) for inference
    # We use x_test_norm because it is already scaled and ready for the Neural Network
    predictions = best_model_obj.model.predict(dataset["x_test_norm"], verbose=0)
    
    # 3. Zip predictions with the actual y values or IDs to show results
    # Note: In a production scenario, you'd map these back to the 'customer_id' column
    results = []
    for i, prob in enumerate(predictions[:10]): # Showing first 10 for clarity
        is_churning = prob[0] > PREDICTION_PROB_THRESHOLD
        status = "❌ LEAVING" if is_churning else "✅ STAYING"
        results.append(f"Customer Index {i}: Probability: {prob[0]:.4f} -> {status}")

    show_banner("Individual Predictions (Sample)", results)

    # Pick best model by test data.

    # Evaluate F1-Score and Recall on the test set.


    show_banner("BANK CHURN PREDICTOR RESULTS", subtitles)

def run_main_pipeline(args: dict):

    dataset, raw_csv_data = run_data_pipeline(args)

    #print(f"dataset[x_train]=dataset{dataset['x_train']}")
    #print(f"dataset[x_train_norm]=dataset{dataset['x_train_norm']}")
    #import sys
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


    train_matrix, val_matrix = run_model_comparison_pipeline(args, models)

    # ---- Get Best Model ---- #
    """
    •
What it does: It identifies the "Development Champion." It looks at the models' performance during the training phase to see which architecture (Adam, SGD, SMOTE, etc.) handled the training-to-validation transition best.
•
How it got the results: It uses the train_matrix and val_matrix generated by run_model_comparison_pipeline. These matrices contain metrics recorded during the training process (the .fit() stage).
•
Purpose: It helps you detect overfitting. If a model has a perfect score in train_matrix but a poor score in val_matrix, best_model logic (as we refactored in model_perf.py) would flag that model as unreliable.
    """
    best_model = ModelPerformance.best_model(train_matrix, val_matrix)
    print(f"final results type = {type(best_model)}")

    show_banner("Best Model", [f"🏆 {best_model}"], center_subtitle_text=True)


    # --- Final Test Evaluation --- #
    # This is the Real World check using the test data
    # Each model inherits the base class that has the entire dataset.  Test data is unused at this point and
    # will be the same for each model.
    test_model_perfs = []
    for test_model in models:
        test_model.model_perf.get(
            f"{test_model.title} (Test Set)",
            test_model.model,
            test_model.x_test_norm,
            test_model.y_test
        )

        #line = f"{model.title}: {model.model_perf.data}"
        test_model_perfs.append(test_model.model_perf.data)

        #evals.append(line)

    """
    
What it does: It identifies the "Real-World Champion." This is the model the bank would actually put into production. It evaluates the models on data they have never seen before—not even during the validation tuning.
•
How it got the results:
i.
The code loops through each model in your models list.
ii.
It calls test_model.model_perf.get() using the x_test_norm and y_test hold-out data.
iii.
It collects these "final exam" scores into a list called test_model_perfs.
iv.
ModelPerformance.final_model() then aggregates these scores to pick the ultimate winner based on the highest F1-Score on that unseen data.
•
Purpose: This is the most honest metric. A model might "cheat" or overfit the validation data if you tune it too much, but it can't "cheat" the test set.
    """

    final_model = ModelPerformance.final_model(test_model_perfs)
    show_banner("Final Evaluation", [final_model])


    run_customer_churn_results(final_model, dataset, raw_csv_data)




    sys.exit(0)




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


# @TODO - old and needs to be deleted
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

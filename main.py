
"""
@link https://peps.python.org/pep-0008/
"""

# Import Python Libraries
import warnings  # To suppress warnings

# Import Vendor Libraries
import pandas as pd
from sklearn.metrics import (
    classification_report,
)
from tensorflow.keras.optimizers import Adam, SGD

# Import local libraries
import src.constants as const
from src.data_splitter import SplitData
from src.data_loader import load_and_clean_data
from src.model_builder import ModelBuilder
from src.preprocessing import preprocessing_data, get_smote

from src.utils import show_banner, seed_script
from src.visualization import (plot_model_performance,
                               show_salary_barplot_visualization,
                               show_plot_distributions,
                               show_correlation_matrix,
                               observe_data,
                               show_visualizations
                               )

"""
Terminal:

>_ python3 -m venv venv && source venv/bin/activate  # On Windows: venv\Scripts\activate
>_ pip install -r requirements.txt 
>_ python main.py
"""

def main():

    warnings.filterwarnings('ignore')

    # Set seeds for reproducibility
    seed_script(const.SEED)

    # Load and clean data
    data = load_and_clean_data(const.DATA_FILE_PATH)
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
    adam_model = ModelBuilder('Neural Network (Adam Optimizer)', Adam(learning_rate=const.LEARNING_RATE))
    adam_model.create_adam_model(training_norm.shape[1])
    adam_model_history = adam_model.build(data_sp)
    #adam_model.evaluate(data_sp)
    plot_model_performance(adam_model_history, 'accuracy', adam_model.title)
    plot_model_performance(adam_model_history, 'loss', adam_model.title)

    # 3) Build Adam Optimized Model with Dropout
    adam_dropout_model = ModelBuilder('Neural Network (Adam and Dropout)', Adam(learning_rate=const.LEARNING_RATE))
    adam_dropout_model.create_adam_dropout_model(training_norm.shape[1])
    adam_dropout_model_history = adam_dropout_model.build(data_sp)
    #adam_dropout_model.evaluate(data_sp)
    plot_model_performance(adam_dropout_model_history, 'accuracy', adam_dropout_model.title)
    plot_model_performance(adam_dropout_model_history, 'loss', adam_dropout_model.title)

    # -- SMOTE --
    x_smote, y_smote = get_smote(data_sp)
    # ------------

    # Build Neural Network (SGD with SMOTE)
    sgd_smote_model = ModelBuilder('Neural Network (SGD with SMOTE)', SGD(learning_rate=const.LEARNING_RATE, momentum=0.9))
    sgd_smote_model.create_smote_model(x_smote.shape[1])
    sgd_smote_model_history = sgd_smote_model.build(data_sp, x_smote, y_smote)

    # Generate Smote Classification Report
    show_banner(sgd_smote_model.title, 'Classification Report')
    y_predictor = (sgd_smote_model.predict(data_sp.testing['normalized']) > const.PREDICTION_PROB_THRESHOLD).astype(int)
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
    adam_smote_dropout_model = ModelBuilder('Neural Network (Adam and Dropout with SMOTE)', Adam(learning_rate=const.LEARNING_RATE))
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
    model_validation_perfs = model_validation_perfs.T # Transpose so metrics are columns, models are rows

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


if __name__ == '__main__':
    main()

# --- End of Program ---

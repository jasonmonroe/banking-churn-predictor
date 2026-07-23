# Banking Churn Predictor

![Hero Image](data/hero.jpeg)

## Context
In service industries, particularly banking, mitigating customer churn is a critical business imperative. Understanding
the factors contributing to customer attrition is essential for sustained profitability. We propose developing a Machine
Learning model to rigorously quantify the feature importance of various service attributes (e.g., transaction speed,
fee structures, customer support quality) on a customer's likelihood to terminate their service agreement. The
resulting prioritized list of influential factors will guide management in making data-driven decisions to refine the
service offering and optimize retention strategies.

## Objective
As a Data Scientist, you are tasked with architecting and implementing a Deep Learning model—specifically a Neural
Network—for binary classification. This model must utilize historical customer data to generate a probability score
indicating the likelihood of an individual customer terminating their service agreement with the bank within the
upcoming half-year period.
This project predicts customer churn using machine learning. It includes data processing pipelines and model training
scripts refactored from experimental notebooks.

## 📊Project Structure
- `src/`: Contains the source code for data processing and modeling.
- `notebooks/`: Jupyter notebooks used for EDA and experimental modeling.
- `data/`: Directory for dataset storage (not included in repo).

```
banking-churn-predictor/
├── main.py                     # CLI entry point for data seeding, EDA, and model execution
├── requirements.txt            # Python dependencies
├── data/                       # Dataset storage
│   ├── hero.png               # Hero graphic / banner image
│   └── source_data.csv        # Standard input dataset
├── models/
│   └── adam_dropout_model.py
│   └── adam_model.py
│   └── adam_smote_dropout_model.py
│   └── adam_smote_model.py
│   └── sgd_model.py
│   └── sgd_smote_dropout_model.py
│   └── sgd_smote_model.py
│   └── smote_model.py
├── notebooks/                                   # Full Jupyter Notebook files
│   └── banking_churn_predictor_notebook.html   # Full Notebook in HTML format
│   └── banking_churn_predictor_notebook.ipynb  # Full Jupyter Notebook
│   └── banking_churn_predictor_notebook.py     # Converted Python file
├── src/
│   ├── constants.py        # Application configurations and constants
│   ├── data_handler.py     # Handles the data extraction and manipulation
│   ├── eda.py              # Exploratory Data Analysis module
│   ├── model_perf.py       # Gets model performance and show the data results
│   └── utils.py            # Logging, plotting, and common utilities
└── tests/                   # Unit Tests
└── venv/                    # Virtual environment (git-ignored)
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd banking-churn-predictor
   ```

2. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   ⚠️ Warning: You must run this on Python version 3.11 for this project to work!

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Just in case you have issues viewing the EDA consoles it's recommended to download Tcl/Tk
    ```bash
    brew install python-tk@3.11
    ```

     ℹ️ Homebrew's Python installation decouples the GUI framework (Tcl/Tk) from the core runtime to save space. Unless
     explicitly installed via Homebrew, Python cannot find the graphic components needed to pop open interactive windows.

## Usage

To run the main pipeline:

```bash
python main.py
```

### Command-line Options

The `main.py` script accepts several command-line arguments to control its behavior, including which models to run and whether to perform Exploratory Data Analysis (EDA).

**General Options:**

*   `--all`: This flag runs all available models. If any specific model flags (e.g., `--model:sgd`) are provided, the `--all` flag is automatically set to `False`, and only the specified models will run. If no model flags are provided, `--all` defaults to `True`.
*   `--eda`: This flag enables Exploratory Data Analysis. When set, the script will display various visualizations and statistical summaries of the dataset, including salary bar plots, distribution plots, and correlation matrices.

**Model-Specific Options:**

You can specify individual models to run using the following flags. If you specify one or more model flags, the `--all` flag will be ignored, and only the models you explicitly list will be executed.

*   `--model:sgd`: Runs the Neural Network model with Stochastic Gradient Descent (SGD) optimizer.
*   `--model:adam`: Runs the Neural Network model with the Adam optimizer.
*   `--model:adam-dropout`: Runs the Neural Network model with the Adam optimizer and Dropout regularization.
*   `--model:sgd-smote`: Runs the Neural Network model with SGD optimizer and SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced datasets.
*   `--model:adam-smote`: Runs the Neural Network model with the Adam optimizer and SMOTE.
*   `--model:adam-smote-dropout`: Runs the Neural Network model with the Adam optimizer, Dropout regularization, and SMOTE.

**Examples:**

1.  **Run all models and perform EDA:**
    ```bash
    python main.py --all --eda
    ```
2.  **Run only the SGD and Adam models without EDA:**
    ```bash
    python main.py --model:sgd --model:adam
    ```
3.  **Run the Adam model with Dropout and SMOTE, and perform EDA:**
    ```bash
    python main.py --model:adam-smote-dropout --eda
    ```
4.  **Run all models (default behavior if no model flags are specified) without EDA:**
    ```bash
    python main.py
    ```

## License
[MIT](https://choosealicense.com/licenses/mit/)
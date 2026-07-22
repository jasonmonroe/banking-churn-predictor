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

banking-churn-predictor/
├── main.py                     # CLI entry point for data seeding, EDA, and model execution
├── requirements.txt            # Python dependencies
├── data/                       # Dataset storage
│   ├── hero.png                # Hero graphic / banner image
│   └── source_data.csv         # Standard input dataset
├── models/
│   └── adam_dropout_model.py
│   └── adam_model.py
│   └── adam_smote_dropout_model.py
│   └── adam_smote_model.py
│   └── sgd_model.py
│   └── sgd_smote_dropout_model.py
│   └── sgd_smote_model.py
├── notebooks/                    # Full Jupyter Notebook files
│   └── banking_churn_predictor_notebook.html   # Full Notebook in HTML format
│   └── banking_churn_predictor_notebook.ipynb  # Full Jupyter Notebook
│   └── banking_churn_predictor_notebook.py     # Converted Python file
├── src/
│   ├── config.py           # Application configurations and constants
│   ├── eda.py              # Exploratory Data Analysis module
│   ├── modeling.py         # Model training, prediction, and helper functions
│   ├── pre_processing.py   # Data loading and cleaning pipelines
│   ├── seeder.py           # Synthetic dataset generator module
│   └── utils.py            # Logging, plotting, and common utilities
└── venv/                    # Virtual environment (git-ignored)


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

## Usage

To run the main pipeline:

```bash
python main.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
# Banking Churn Predictor

## Overview
This project predicts customer churn using machine learning. It includes data processing pipelines and model training scripts refactored from experimental notebooks.

## Project Structure
- `src/`: Contains the source code for data processing and modeling.
- `notebooks/`: Jupyter notebooks used for EDA and experimental modeling.
- `data/`: Directory for dataset storage (not included in repo).

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd banking-churn-predictor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

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
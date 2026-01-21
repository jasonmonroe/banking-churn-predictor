
import os
import pandas as pd
from src.preprocessing import drop_columns
from src.seeder import Seeder

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    load and clean data
    :param file_path:
    :return:
    """

    # If data doesn't exists, seed data
    seed_type = prompt()

    if seed_type == 1 or os.path.isfile(file_path):
        df = pd.read_csv(file_path)

        if len(df) == 0:
            df = generate_data()

        df.fillna(0, inplace=True)
    else:
        df = generate_data()

    # Drop columns that are least relevant to the data.
    dropped = ['row_number', 'customer_id', 'surname']
    df = df.drop(columns=dropped, axis=1, errors='ignore')

    return df


def prompt() -> int:
    message = 'Press 1 to use pre-defined sample data.  Press 2 to generate sample data.'
    seed_type = input(message)

    while seed_type not in [1, 2]:
        seed_type = input(message)

    return seed_type

def generate_data():
    count = int(input('How many rows do you want to generate?'))
    return Seeder(count).seed_data()

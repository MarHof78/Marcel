import pandas as pd


def get_data(filepath, seperator):
    """Creates a dataset for the data

    :param filepath: Path to the csv-File
    :param seperator: The used seperator for the data
    :return: dataset for data
    """
    df = pd.read_table(filepath,
                       sep=seperator,
                       header=0,
                       na_values=-99999)
    return df

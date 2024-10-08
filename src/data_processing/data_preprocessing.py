import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Implement preprocessing steps
    data = data.dropna()
    return data

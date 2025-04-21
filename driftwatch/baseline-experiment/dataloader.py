import pandas as pd
import torch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon (Metal Performance Shaders)
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")  # Fallback to CPU


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        data = pd.read_json(self.filepath, lines=True)
        data = data[data["rating"] != 3]
        data = data[data["timestamp"] > "2015-01-01"]
        data["year"] = data["timestamp"].apply(lambda x: x.year)
        data["month"] = data["timestamp"].apply(lambda x: x.month)
        data["day"] = data["timestamp"].apply(lambda x: x.day)
        data["rating"] = data["rating"].transform(lambda x: 1 if x > 3 else 0)
        data.set_index(data["timestamp"], inplace=True)
        data.sort_index(inplace=True)
        print(f"Data shape after filtering: {data.shape}")
        return data

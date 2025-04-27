import pandas as pd

def load_data():
    df = pd.read_csv("data/processed_features.csv")
    X = df.iloc[:, 1:-1].values
    y = df["label"].values
    return X, y

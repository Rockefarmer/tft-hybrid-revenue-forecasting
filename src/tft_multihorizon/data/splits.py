import pandas as pd


def chronological_split(df: pd.DataFrame, time_col: str, train_end, val_end):
    train = df[df[time_col] <= train_end].copy()
    val = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test = df[df[time_col] > val_end].copy()
    return train, val, test

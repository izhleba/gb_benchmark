import pandas as pd

def data_csc_hw1_spring19(num_rows):
    df = pd.read_csv('../data/csc-hw1-spring19/train.csv',nrows=num_rows)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y
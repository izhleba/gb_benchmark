import pandas as pd

def data_heart_disease(num_rows):
    df = pd.read_csv('data/heart-disease/heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return X,y
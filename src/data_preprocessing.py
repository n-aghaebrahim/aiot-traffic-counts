import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from math import sqrt
import argparse

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.fillna(0)

    # Normalize traffic volume columns
    traffic_volume_columns = df.columns[7:]  # Assuming traffic data starts from the 8th column
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[traffic_volume_columns] = scaler.fit_transform(df[traffic_volume_columns])

    # Extract day of week as a categorical feature
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfWeek'] = to_categorical(df['DayOfWeek'])


    traffic_volume_columns_x = df.columns[7:-2]
    traffic_volume_columns_y = df.columns[-2]
    X = df[traffic_volume_columns_x].values
    y = df[traffic_volume_columns_y].values

    #X = df[traffic_volume_columns].values
    #y = df['target_column_name'].values  # Replace 'target_column_name' with the actual column name

    X = X.reshape((X.shape[0], X.shape[1], 1))
    return train_test_split(X, y, test_size=0.2, random_state=42)


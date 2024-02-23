import argparse
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses time series data from a CSV file. This function reads the data,
    normalizes selected columns, extracts day of the week as a categorical feature, and
    prepares the dataset for training by reshaping and splitting into training and testing sets.

    Parameters:
    - filepath: String, path to the CSV file containing the dataset.

    Returns:
    - X_train, X_test, y_train, y_test: Arrays, split and preprocessed data ready for training and testing.
    """
    # Load data
    df = pd.read_csv(filepath)
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    # Fill missing values with 0
    df = df.fillna(0)

    # Normalize specified traffic volume columns
    traffic_volume_columns = df.columns[7:]  # Assuming traffic data starts from the 8th column
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[traffic_volume_columns] = scaler.fit_transform(df[traffic_volume_columns])

    # Extract and encode day of the week as a categorical feature
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfWeek'] = to_categorical(df['DayOfWeek'])

    # Define feature and target columns
    traffic_volume_columns_x = df.columns[7:-2]  # Feature columns
    traffic_volume_columns_y = df.columns[-2]  # Target column
    X = df[traffic_volume_columns_x].values
    y = df[traffic_volume_columns_y].values

    # Reshape features for model input
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split the dataset into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)


import argparse
from datetime import date, datetime, timedelta
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



def create_lstm_model(input_shape):
    """
    Creates and compiles an LSTM model with a Conv1D layer followed by MaxPooling,
    LSTM, and Dense layers for sequence data processing.

    Parameters:
    - input_shape: Tuple of integers, the shape of the input data (time_steps, features).

    Returns:
    - Compiled Keras model ready for training.
    """
    model = Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(50, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def create_conv_lstm_model(n_steps, n_features, filter_num=64, kernel_size=3, pool_size=2, lstm_units=50, dense_units=100, dropout_rate=0.2, learning_rate=0.001):
    """
    Creates and compiles a Convolutional LSTM model for sequence data processing, integrating
    Conv1D and LSTM layers with Dropout for regularization.

    Parameters:
    - n_steps: Integer, number of time steps in each input sample.
    - n_features: Integer, number of features in each input sample.
    - filter_num: Integer, number of filters in the Conv1D layer (default 64).
    - kernel_size: Integer, size of the convolution kernel (default 3).
    - pool_size: Integer, size of the pooling window (default 2).
    - lstm_units: Integer, number of units in the LSTM layer (default 50).
    - dense_units: Integer, number of units in the Dense layer (default 100).
    - dropout_rate: Float between 0 and 1, fraction of the input units to drop (default 0.2).
    - learning_rate: Float, learning rate for the Adam optimizer (default 0.001).

    Returns:
    - Compiled Keras model ready for training.
    """
    model = Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.Conv1D(filters=filter_num, kernel_size=kernel_size, activation='relu'),
        layers.MaxPooling1D(pool_size=pool_size),
        layers.Flatten(),
        layers.Reshape((1, -1)),  # Reshape for LSTM layer
        layers.LSTM(units=lstm_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(units=dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1)  # Output layer
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model


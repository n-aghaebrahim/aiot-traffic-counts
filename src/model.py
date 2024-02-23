from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from math import sqrt
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, Input, LSTM, MaxPooling1D, Reshape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import TimeseriesGenerator



def create_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def create_conv_lstm_model(n_steps, n_features, filter_num=64, kernel_size=3, pool_size=2, lstm_units=50, dense_units=100, dropout_rate=0.2, learning_rate=0.001):


    # Define the input layer for the model
    model = Sequential()
    #inputs = tf.keras.layers.Input(shape=(n_steps, n_features))
    #inputs = Reshape((n_steps,1, n_features, 1))(inputs)

    # Apply Convolutional Neural Network (CNN) layers to the input data
    #model.add(layers.TimeDistributed(tf.keras.layers.Conv1D(filters=filter_num, kernel_size=3, dilation_rate=1, padding=0, activation='relu')))
    #model.add(layers.TimeDistributed(layers.Flatten()))



    # Input shape for Conv1D should be (n_steps, n_features) per sample
    model.add(Input(shape=(n_steps, n_features)))
    # Conv1D layer
    model.add(Conv1D(filters=filter_num, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    # Flatten the output of the Conv layers to feed into the LSTM layer
    model.add(Flatten())
    # Reshape for LSTM layer - converting back to sequence (1, units) as LSTM expects 3D input
    model.add(Reshape((1, -1)))
    # LSTM layer
    model.add(LSTM(units=lstm_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    # Dense layer
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    # Output layer
    model.add(Dense(1))  # Predicting a single value; adjust according to your output

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model



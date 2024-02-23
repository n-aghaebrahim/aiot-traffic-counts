import os
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

from src import data_preprocessing, model, train, predict, plot

def main():
    parser = argparse.ArgumentParser(description='Train a model to predict traffic volume.')
    parser.add_argument('--datafile', type=str, required=True, help='Path to the traffic volume data file')
    args = parser.parse_args()
    
    n_steps=23
    n_features=1
    filter_num=64
    kernel_size=3 
    pool_size=2 
    lstm_units=50 
    dense_units=100
    dropout_rate=0.2
    learning_rate=0.001

    epochs=12 
    batch_size=32 
    validation_split=0.2
    verbose=1

    model_name = "conv_lstm"

    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    X_train, X_test, y_train, y_test = data_preprocessing.load_and_preprocess_data(args.datafile)
    
    if model_name == "lstm":
        model_c = model.create_model((X_train.shape[1], 1))
    
    if model_name == 'conv_lstm':
        model_c = model.create_conv_lstm_model(n_steps, n_features, filter_num=filter_num, kernel_size=kernel_size, pool_size=pool_size, lstm_units=lstm_units, dense_units=dense_units, dropout_rate=dropout_rate, learning_rate=learning_rate)



    if model_name == "lstm":
        history = train.train_model(model_c, X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
        y_pred = predict.predict(model_c, X_test)

        plot.plot_predict(y_test, y_pred, results_dir)
        plot.plot_training_history(history, results_dir)

        rmse = sqrt(mean_squared_error(y_test, y_pred))
        print(f"Root Mean Squared Error: {rmse}")

    if model_name == "conv_lstm":
        history = train.train_model(model_c, X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
        y_pred = predict.predict(model_c, X_test)

        plot.plot_conv_lstm_predict(y_test, y_pred, results_dir)
        plot.plot_conv_lstm_training_history(history, results_dir)

        rmse = sqrt(mean_squared_error(y_test, y_pred))
        print(f"Root Mean Squared Error: {rmse}")



if __name__ == '__main__':
    main()

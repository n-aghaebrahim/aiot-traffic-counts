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


def plot_training_history(history, results_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    #plt.show()

    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, 'training_and_validation_loss.png'))
    plt.close()  # Close the plot to free up memory


def plot_predict(y_test, y_pred, results_dir):
    N = 100  # Number of samples to plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:N], label='Actual', marker='.', linestyle='-', linewidth=1.5)
    plt.plot(y_pred[:N], label='Predicted', marker='o', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title('Predicted vs Actual Traffic Volume')
    plt.xlabel('Sample Index')
    plt.ylabel('Traffic Volume')
    plt.legend()
    #plt.show()

    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, 'predicted_vs_actual_traffic_volume.png'))
    plt.close()  #



def plot_conv_lstm_predict(y_test, y_pred, results_dir):
    N = 100  # Number of samples to plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:N], label='Actual', marker='.', linestyle='-', linewidth=1.5)
    plt.plot(y_pred[:N], label='Predicted', marker='o', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title('Predicted vs Actual Traffic Volume')
    plt.xlabel('Sample Index')
    plt.ylabel('Traffic Volume')
    plt.legend()
    #plt.show()
    
    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, 'predicted_vs_actual_traffic_volume_lstm_conv.png'))
    plt.close()  #



def plot_conv_lstm_training_history(history, results_dir):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot training & validation accuracy values
    # Only if you have 'mean_absolute_error' in your metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, 'training_loss_mean_conv_lstm.png'))
    plt.close()  #



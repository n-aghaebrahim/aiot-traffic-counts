import os
import matplotlib.pyplot as plt




def plot_training_history(history, results_dir):
    """
    Plots the training and validation loss over epochs and saves the plot to a file.

    Parameters:
    - history: History object returned by the fit method of a keras Model.
    - results_dir: String, directory path to save the plot image.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, 'training_and_validation_loss.png'))
    plt.close()


def plot_predict(y_test, y_pred, results_dir):
    """
    Plots the first N actual vs predicted values and saves the plot to a file.

    Parameters:
    - y_test: Array, actual target values.
    - y_pred: Array, predicted target values.
    - results_dir: String, directory path to save the plot image.
    """
    N = 100  # Number of samples to plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:N], label='Actual', marker='.', linestyle='-', linewidth=1.5)
    plt.plot(y_pred[:N], label='Predicted', marker='o', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title('Predicted vs Actual Traffic Volume')
    plt.xlabel('Sample Index')
    plt.ylabel('Traffic Volume')
    plt.legend()

    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, 'predicted_vs_actual_traffic_volume.png'))
    plt.close()


def plot_conv_lstm_predict(y_test, y_pred, results_dir):
    """
    Plots the first N actual vs predicted values for a ConvLSTM model and saves the plot to a file.

    Parameters:
    - y_test: Array, actual target values.
    - y_pred: Array, predicted target values.
    - results_dir: String, directory path to save the plot image.
    """
    N = 100  # Number of samples to plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:N], label='Actual', marker='.', linestyle='-', linewidth=1.5)
    plt.plot(y_pred[:N], label='Predicted', marker='o', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title('Predicted vs Actual Traffic Volume')
    plt.xlabel('Sample Index')
    plt.ylabel('Traffic Volume')
    plt.legend()

    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, 'predicted_vs_actual_traffic_volume_lstm_conv.png'))
    plt.close()


def plot_conv_lstm_training_history(history, results_dir):
    """
    Plots the training and validation loss and Mean Absolute Error (MAE) over epochs for a ConvLSTM model
    and saves the plots to a file.

    Parameters:
    - history: History object returned by the fit method of a keras Model.
    - results_dir: String, directory path to save the plot image.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(os.path.join(results_dir, 'training_loss_mean_conv_lstm.png'))
    plt.close()


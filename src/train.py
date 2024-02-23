from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def train_model(model, X_train, y_train, epochs, batch_size, validation_split, verbose):
    """
    Trains the provided Keras model using the given training data and parameters.

    Parameters:
    - model: The Keras model to train.
    - X_train: Training data features.
    - y_train: Training data labels.
    - epochs: Number of epochs for training.
    - batch_size: Size of the batches of data.
    - validation_split: Fraction of the data to use as validation set.
    - verbose: Verbosity mode.

    Returns:
    - A history object containing the training history.
    """
    return model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )



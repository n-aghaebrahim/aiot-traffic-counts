import os
from tensorflow.keras.models import Sequential

def predict(model, X_test):
    """
    Uses the provided Keras model to make predictions on the given test data.

    Parameters:
    - model: The Keras model to use for making predictions.
    - X_test: Test data features for which predictions are to be made.

    Returns:
    - The predictions made by the model on the given test data.
    """
    return model.predict(X_test)






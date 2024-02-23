import os
import argparse
from math import sqrt
from sklearn.metrics import mean_squared_error
from src import data_preprocessing, model, train, predict, plot
from config import config

def main(datafile):
    """
    Main function to orchestrate the data preprocessing, model training, prediction, and evaluation workflow.
    It dynamically selects the model based on configuration, trains the model, makes predictions, and evaluates the results.
    
    Parameters:
    - datafile: String, path to the dataset file.
    """
    # Ensure results directory exists
    os.makedirs(config["results_dir"], exist_ok=True)

    # Data preprocessing
    X_train, X_test, y_train, y_test = data_preprocessing.load_and_preprocess_data(datafile)

    # Model creation based on configuration
    if config["model_name"] == 'conv_lstm':
        model_c = model.create_conv_lstm_model(config["n_steps"], config["n_features"],
                                               config["filter_num"], config["kernel_size"],
                                               config["pool_size"], config["lstm_units"],
                                               config["dense_units"], config["dropout_rate"],
                                               config["learning_rate"])
    else:
        # Fallback to a default LSTM model if 'conv_lstm' is not specified
        model_c = model.create_lstm_model((X_train.shape[1], 1))

    # Model training and prediction
    history = train.train_model(model_c, X_train, y_train, epochs=config["epochs"],
                                batch_size=config["batch_size"], validation_split=config["validation_split"],
                                verbose=config["verbose"])
    y_pred = predict.predict(model_c, X_test)
    
    # Plotting results and training history
    plot_func = getattr(plot, f"plot_{config['model_name']}_predict", plot.plot_predict)
    plot_func(y_test, y_pred, config["results_dir"])
    
    plot_history_func = getattr(plot, f"plot_{config['model_name']}_training_history", plot.plot_training_history)
    plot_history_func(history, config["results_dir"])

    # Evaluation
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error: {rmse}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model to predict traffic volume.')
    parser.add_argument('--datafile', type=str, required=True, help='Path to the traffic volume data file')
    args = parser.parse_args()

    main(args.datafile)


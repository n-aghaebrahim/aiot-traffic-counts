# config.py

config = {
    "n_steps": 23,
    "n_features": 1,
    "filter_num": 64,
    "kernel_size": 3,
    "pool_size": 2,
    "lstm_units": 50,
    "dense_units": 100,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "epochs": 12,
    "batch_size": 32,
    "validation_split": 0.2,
    "verbose": 1,
    "model_name": "lstm", # lstm or conv_lstm
    "results_dir": "./results",
}


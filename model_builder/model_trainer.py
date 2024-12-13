# Import necessary libraries
import os
import sys
import numpy as np
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
import tensorflow as tf
from tensorflow import keras
from variable_config import LOOK_BACK as look_back, EPOCHS, file_name
from data_wrangler import data_wrangler, split_into_datasets
from model_builder import hb_tuner_lstm, hb_tuner_gru
from numpy.random import seed
from tensorflow import random

def train_model(model_name='lstm', tuner=hb_tuner_lstm):
    
    # Call data_wrangler to create features and label.
    # X, y, data, scaler, raw_data, prep_data = data_wrangler(file_name, look_back)
    # Don't want to unpack all variables, use result[3] from the entire result.
    X, y, _, _, _, _, _ = data_wrangler(file_name, look_back)


    # Hold out validation data
    # X_train, X_val, X_test, y_train, y_val, y_test = split_into_datasets(X=X, y=y, look_back=look_back, get_val_set=True)
    X_train, X_val, _, y_train, y_val, _ = split_into_datasets(X=X, y=y, look_back=look_back, get_val_set=True)
    
    # Create determinism and model reproducibility
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    # This code make sure that each time you run your code, your neural network weights will be initialized equally.
    seed(42)
    random.set_seed(42)

    # Early stoping callbacks for best epoch
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min')
    early_stop_retrain = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, mode='min')

    check_point = keras.callbacks.ModelCheckpoint(
        # 'hyper_model/check_point/{epoch:03d}-{val_loss:.4f}.keras',
        # 'hyper_model/check_point/epoch_{epoch:03d}.keras',
        f'hyper_model/check_point/best_{model_name}_check_point.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'
    )

    # Saves the best model's training history
    # append=True, it appends if file exists (useful for continuing training)
    csv_logger = keras.callbacks.CSVLogger(filename=f'hyper_model/best_model/best_{model_name}_model_history.csv', separator=',', append=False)

    tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stop, check_point])

    print(f'Search Space Summary of {model_name.upper()}:\n', tuner.search_space_summary())

    # Returns the best hyperparameters, as determined by the objective.
    # These hyperparameters can be used to reinstantiate the (untrained) best model found during the search process.
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'Best hyperparameters of {model_name.upper()}:\n', best_hps.values)
    
    # Build the model with the optimal hyperparameters and train it on the data for user defined epochs
    model = tuner.hypermodel.build(best_hps)
    # For best performance, it is recommended to retrain your Mode_lstml on the full dataset using the best hyperparameters found during search
    # Retrain the model with the entire dataset including training and validation sets
    # Fit with the entire dataset.
    X_all = np.concatenate((X_train, X_val))
    y_all = np.concatenate((y_train, y_val))
    model.fit(x=X_all, y=y_all, epochs=EPOCHS, callbacks=[early_stop_retrain, csv_logger])

    # Saves the entire model in new high-level .keras format
    model.save(f'hyper_model/best_model/best_{model_name}_model.keras')
    # best_model.save('hyper_model/best_model/best_model.h5')
    # Saves the entire model as a SavedModel. It places the contents of model in a directory.
    # best_model.export('hyper_model/best_model/best_model')

    # save tuner
    tuner.save()

    return tuner, best_hps, model

tuner, best_hps_lstm, model_lstm = train_model()
hb_tuner_gru, best_hps_gru, model_gru = train_model('gru', hb_tuner_gru)
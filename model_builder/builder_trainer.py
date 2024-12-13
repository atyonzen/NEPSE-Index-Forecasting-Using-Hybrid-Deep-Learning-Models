# Requires variable definitions for LOOK_BACK, MAX_EPOCHS, and DISABLE_RESUME
import os
import sys
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
from variable_config import LOOK_BACK
import numpy as np
from tensorflow import keras
from variable_config import LOOK_BACK as look_back, file_name, EPOCHS
from data_wrangler import data_wrangler, split_into_datasets

# This code make sure that each time you run your code, your neural network weights will be initialized equally.
from numpy.random import seed
seed(42)
from tensorflow import random
random.set_seed(42)

# Call data_wrangler to create features and label.
X, y, scaler, raw_data, prep_data, data, unseen_data = data_wrangler(file_name, look_back, unseen=True)
# Split data sets.
X_train, X_val, _, y_train, y_val, _ = split_into_datasets(X=X, y=y, look_back=look_back, get_val_set=True)

# Saves the best model's training history
# append=True, it appends if file exists (useful for continuing training)
csv_logger_lstm = keras.callbacks.CSVLogger(filename='hyper_model/best_model/best_lstm_model_history.csv', separator=',', append=False)
csv_logger_gru = keras.callbacks.CSVLogger(filename='hyper_model/best_model/best_gru_model_history.csv', separator=',', append=False)
early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, mode='min')

# Fit with the entire dataset.
X_all = np.concatenate((X_train, X_val))
y_all = np.concatenate((y_train, y_val))

def build_train_lstm():

    model_lstm = keras.Sequential()

    # This is the recommended approach to define input shape in keras
    model_lstm.add(keras.layers.Input(shape=(LOOK_BACK, 1)))  # Define input shape here
    model_lstm.add(
        keras.layers.LSTM(
            units=161,
            activation='tanh', 
            return_sequences=False
            # input_shape=(LOOK_BACK, 1)
        )
    )

    model_lstm.add(
        keras.layers.Dropout(
            rate=0.0033810516825470596
        )
    )

    model_lstm.add(keras.layers.Dense(units=161))
    model_lstm.add(keras.layers.Dense(1))
                
    model_lstm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.004787239377037189),
        # Optimizer tries to minimize the loss function when training the model.
        # loss=keras.losses.MeanAbsoluteError(),
        loss=keras.losses.MeanSquaredError(),
        # optimizer=optimizer,
        # loss=loss,
        # https://stackoverflow.com/questions/48280873/what-is-the-difference-between-loss-function-and-metric-in-keras#:~:text=The%20loss%20function%20is%20that,parameters%20passed%20to%20Keras%20model.
        # A metric is used to judge the performance of your model. 
        # This is only for you to look at and has nothing to do with the optimization process.
        # Metric is the model performance parameter that one can see while the model is judging 
        # itself on the validation set after each epoch of training. It is important to note that 
        # the metric is important for few Keras callbacks like EarlyStopping when one wants to stop 
        # training the model in case the metric isn't improving for a certaining no. of epochs.
        # metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'r2_score']
        metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.MeanAbsolutePercentageError(), keras.metrics.R2Score()]
        # metrics=[metrics]
    )

    # Train model with combined data set of training and validation sets
    print('LSTM Training')
    model_lstm.fit(x=X_all, y=y_all, batch_size=24, epochs=EPOCHS, callbacks=[early_stop, csv_logger_lstm])

    # Saves the entire model in new high-level .keras format
    model_lstm.save('hyper_model/best_model/best_lstm_model.keras')

def build_train_gru():

    model_gru = keras.Sequential()

    # This is the recommended approach to define input shape in keras
    model_gru.add(keras.layers.Input(shape=(LOOK_BACK, 1)))  # Define input shape here
    model_gru.add(
        keras.layers.GRU(
            units=153,
            activation='tanh', 
            return_sequences=False
            # input_shape=(LOOK_BACK, 1)
        )
    )

    model_gru.add(
        keras.layers.Dropout(
            rate=0.0036935548740093917
        )
    )

    model_gru.add(keras.layers.Dense(units=193))
    model_gru.add(keras.layers.Dense(1))
                
    model_gru.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0004591527984583266),
        # Optimizer tries to minimize the loss function when training the model.
        # loss=keras.losses.MeanAbsoluteError(),
        loss=keras.losses.MeanSquaredError(),
        # optimizer=optimizer,
        # loss=loss,
        # https://stackoverflow.com/questions/48280873/what-is-the-difference-between-loss-function-and-metric-in-keras#:~:text=The%20loss%20function%20is%20that,parameters%20passed%20to%20Keras%20model.
        # A metric is used to judge the performance of your model. 
        # This is only for you to look at and has nothing to do with the optimization process.
        # Metric is the model performance parameter that one can see while the model is judging 
        # itself on the validation set after each epoch of training. It is important to note that 
        # the metric is important for few Keras callbacks like EarlyStopping when one wants to stop 
        # training the model in case the metric isn't improving for a certaining no. of epochs.
        # metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'r2_score']
        metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.MeanAbsolutePercentageError(), keras.metrics.R2Score()]
        # metrics=[metrics]
    )

    # Train model with combined data set of training and validation sets
    print('GRU Training')
    model_gru.fit(x=X_all, y=y_all, batch_size=16, epochs=EPOCHS, callbacks=[early_stop, csv_logger_gru])

    # Saves the entire model in new high-level .keras format
    model_gru.save('hyper_model/best_model/best_gru_model.keras')

build_train_lstm()
build_train_gru()
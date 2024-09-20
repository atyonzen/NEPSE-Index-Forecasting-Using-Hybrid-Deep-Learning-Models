# Dependencies notes:
# tensorflowjs 4.21.0 depends on tensorflow<3 and >=2.13.0
# tensorflow-decision-forests 1.8.1 depends on tensorflow~=2.15.0
# tensorflow~=2.15.0 requires python 3.9.13
# py -m pip install tensorflow==2.15.0
# py -m pip install tensorflow-decision-forests==1.8.1
# py -m pip install tensorflowjs==4.21.0


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
# from keras.src.models import Sequential
# from keras.src.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow import keras
from keras.src.callbacks import LearningRateScheduler

# reac csv file
# df = pd.read_csv('nepsealpha0c.csv')
df = pd.read_csv('nepsealpha0c.csv')
# remove columns which our neural network will not use
df = df.drop(['Symbol', 'Date', 'Open', 'High', 'Low', 'Percent Change', 'Volume'], axis=1)

EPOCHS = 200
MAX_EPOCHS = 30 # max_epochs = 8 or 10 seems good for large or small sample
DISABLE_RESUME = True
look_back = 7

# Normalize the Close price column
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Prepare the dataset
# look_back is the number of previous days' prices to consider for prediction
def prepare_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + look_back
        # handle last item which may be like 228+1 > 229-1
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_train_test_split(data, look_back = 7, train_test_split = 0.8, val_split = 0.1):

    # Prepare the training data
    X, y = prepare_data(data, look_back)
    
    # Split the data into training and testing sets
    # It is assumed to create validation set from training set
    training_size = int(len(X) * train_test_split)
    validation_size = int(len(X) * val_split)
    X_train, y_train = X[:training_size - validation_size], y[:training_size - validation_size]
    X_val, y_val = X_train[-validation_size:], y_train[-validation_size:]
    X_test, y_test = X[training_size:], y[training_size:]
    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)
    # (155, 7)
    # (22, 7)
    # (45, 7)

    # Reshape the input data for LSTM
    # Reshape input data into [samples, timesteps, features]
    X_train = X_train.reshape(-1, look_back, 1)
    X_val = X_val.reshape(-1, look_back, 1)
    X_test = X_test.reshape(-1, look_back, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Define the stacked LSTM model
class CustomHyperModel(kt.HyperModel):

    # steps = look_back["look_back"]

    def build(self, hp):
        model = keras.Sequential()

        activation = hp.Choice('activation', ['relu', 'tanh', 'sigmoid', 'linear', 'swish'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        num_layers = hp.Int('num_layers', min_value=1, max_value=20)
        optimizers = hp.Choice('optimizer', values =('SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam'))
        loss = hp.Choice('loss', values=('mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'))
        metrics = hp.Choice('metric', values=('mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'))

        for i in range(num_layers):
            model.add(
                keras.layers.LSTM(
                    units=hp.Int(f'units_{i}', min_value=1, max_value=200, step=8),
                    activation=activation, 
                    return_sequences=True if i < num_layers-1 else False,
                    input_shape=(look_back, 1)
                )
            )
            
            if hp.Boolean('dropout'):
                model.add(
                    keras.layers.Dropout(
                        rate=hp.Float(f'dropout_{i}', min_value=1e-4, max_value=0.6, sampling='log')
                    )
                )
        
        model.add(keras.layers.Dense(units=hp.Int('units', min_value=1, max_value=200, step=8)))
        model.add(keras.layers.Dense(1))
        
        # https://stackoverflow.com/questions/67286051/how-can-i-tune-the-optimization-function-with-keras-tuner
        # Match cases
        match optimizers:
            case 'SGD': optimizer = keras.optimizers.SGD(learning_rate=learning_rate) 
            case 'RMSprop': optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate) 
            case 'Adam': optimizer = keras.optimizers.Adam(learning_rate=learning_rate) 
            case 'Adadelta': optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate) 
            case 'Adagrad': optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate) 
            case 'Adamax': optimizer = keras.optimizers.Adamax(learning_rate=learning_rate) 
            case 'Nadam': optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)

            # As a alternative solution
            # https://stackoverflow.com/questions/67286051/how-can-i-tune-the-optimization-function-with-keras-tuner
                    
        model.compile(
                # optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                # Optimizer tries to minimize the loss function when training the model.
                # loss=keras.losses.MeanAbsoluteError(),
                # loss=keras.losses.MeanSquaredError(),
                optimizer=optimizer,
                loss=loss,
                # https://stackoverflow.com/questions/48280873/what-is-the-difference-between-loss-function-and-metric-in-keras#:~:text=The%20loss%20function%20is%20that,parameters%20passed%20to%20Keras%20model.
                # A metric is used to judge the performance of your model. 
                # This is only for you to look at and has nothing to do with the optimization process.
                # Metric is the model performance parameter that one can see while the model is judging 
                # itself on the validation set after each epoch of training. It is important to note that 
                # the metric is important for few Keras callbacks like EarlyStopping when one wants to stop 
                # training the model in case the metric isn't improving for a certaining no. of epochs.
                # metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error']
                # metrics=[keras.metrics.R2Score()]
                metrics=[metrics]
            )

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        # lr = hp.get('learning_rate')
        # epoch = kwargs["epochs"]
        return model.fit(
            *args,
            # validation_data=(X_val, y_val),
            # Tune the appropriate batch size
            batch_size=hp.Choice('batch_size', values=[8, 16, 24, 32, 40, 48, 56, 64, 72]),
            **kwargs,
        )

# Create train and test data
data = df['Close'].values
X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(data=data)

# Create determinism and model reproducibility
# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()

# Keras-Tuner
tuner = kt.Hyperband(
    # create_stacked_lstm_model,
    CustomHyperModel(),
    # objective='val_mean_absolute_error',
    # objective=kt.Objective(name='val_mean_absolute_percentage_error', direction='min'),
    objective=kt.Objective(name='val_loss', direction='min'),
    max_epochs=MAX_EPOCHS, # dafault value is 100.
    factor=3, # dafault value is 3.
    # Disable the usage of the existing state from trail_ logs to resume the search.
    overwrite=DISABLE_RESUME,
    directory='hyper_model',
    project_name='tuning'
)

# Early stoping callbacks for best epoch
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, mode='min')

# Scheduler function for LearningRateScheduler
# def scheduler(epoch, lr):
#     return lr
# lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
# callbacks = [early_stop, lr_scheduler]
tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stop])

# Best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)

# Get the top 1 model.
# best_model = tuner.get_best_models(num_models=1)[0]
# Build the model with the optimal hyperparameters and train it on the data for user defined epochs
model = tuner.hypermodel.build(best_hps)
# history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stop])
history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, validation_data=(X_val, y_val))

val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch))

# exit()

# After finding the best epochs, lets re-instantiate the hypermodel and  retrain it.
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_val, y_val))
best_model.summary()

# Saves the entire model in new high-level .keras format
best_model.save('best_model.keras')
# best_model.save('best_model.h5')
# Saves the entire model as a SavedModel. It places the contents of model in a directory.
# best_model.export('saved_model/best_model')


# Evaluate the hypermodel on the test data.
test_result = best_model.evaluate(X_test, y_test)
print(test_result)
print(
    'Test Evaluation of model:\n',
    test_result,
    'loss(mae): %1f, mape: %2f, r2_score: %3f]:' % (
    scaler.inverse_transform(np.array([test_result[0]]).reshape(-1, 1))[0][0],
    test_result[1],
    test_result[1]
    )
)
# Calculate RMSE performance matrics
# Train data RMSE
# import math
# from sklearn.metrics import mean_squared_error
# math.sqrt(mean_squared_error(y_train, train_predict))

# Test data RMSE
# math.sqrt(mean_squared_error(y_test, test_predict))
# exit()

# Predict using the trained model
# Predict for x_train, X_test
train_predict = best_model.predict(X_train)
val_predict = best_model.predict(X_val)
test_predict = best_model.predict(X_test)
# print(train_predict.shape)
# print(train_predict)
train_predict = scaler.inverse_transform(train_predict)
val_predict = scaler.inverse_transform(val_predict)
test_predict = scaler.inverse_transform(test_predict)

# Plot the predicted prices
# Shift train predictions for plotting
trainPredictPlot = np.empty_like(df)
# Inserts nan values to empty df like dataframe
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back: len(train_predict) + look_back, :] = train_predict

valPredictPlot = np.empty_like(df)
# Inserts nan values to empty df like dataframe
valPredictPlot[:, :] = np.nan
valPredictPlot[len(train_predict) + look_back: len(train_predict) + len(val_predict) + look_back, :] = val_predict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(df)
# Inserts nan values to empty df like dataframe
testPredictPlot[:, :] = np.nan
testPredictPlot[ len(train_predict) + len(val_predict) + look_back : len(df), :] = test_predict

# Predict for coming future
future_steps = 14  # Number of days to predict
future_data = data[-look_back:].reshape(-1, look_back, 1)
predicted_prices = []
for i in range(future_steps):
    prediction = best_model.predict(future_data)
    predicted_prices.append(prediction)
    # Rolls the first item of future_data to last along axis 1
    future_data = np.roll(future_data, -1, axis=1)
    # Predicted value is used for forecasting other values
    # For this, `prediction`` is used to replace the rolled item at the last of future_data
    future_data[0, -1] = prediction

# Inverse transform the predicted prices to original scale
# predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Shift future predictions for plotting
futurePredictPlot = np.empty_like(np.concatenate((df, predicted_prices)))
# Inserts nan values to empty df like dataframe
futurePredictPlot[:, :] = np.nan
futurePredictPlot[ len(df) : len(df) + future_steps, :] = predicted_prices

plt.plot(scaler.inverse_transform(df), label='Actual Prices', linestyle='dashed')
plt.plot(trainPredictPlot, label='Train Predicted Prices')
plt.plot(valPredictPlot, label='Validation Predicted Prices')
plt.plot(testPredictPlot, label='Test Predicted Prices')
plt.plot(futurePredictPlot, label='Future Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Close')
plt.title('NEPSE Index Prediction')
plt.grid(True)
plt.legend(loc='best')
plt.show()
# Import necessary libraries
import os
import sys
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from variable_config import LOOK_BACK as look_back, EPOCHS, file_name
from data_wrangler import data_wrangler, split_into_datasets
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Call data_wrangler to create features and label
X, y, data, scaler = data_wrangler(file_name, look_back)

# Hold out validation data
X_train, X_val, X_test, y_train, y_val, y_test = split_into_datasets(X, y, look_back, get_val_set=True)

# Load saved model
model = keras.models.load_model('hyper_model/best_model/best_model.keras')

def evaluate_model():

    # Model summary
    print(model.summary())

    # print(model.evaluate(X_test, y_test))
    
    
    # Evaluate the hypermodel on the test data.
    # test_loss, test_mae, test_mape, test_r2 = model.evaluate(X_test, y_test)
    test_result = model.evaluate(X_test, y_test)
    print('Evaluation of model:\n')
    print(f'Test Metric: {test_result}')
    print(
        'Test Evaluation of model:\n',
        'mae: %1f, mape: %2f, r2_score: %3f]:' % (
        scaler.inverse_transform(np.array([test_result[1]]).reshape(-1, 1))[0][0],
        test_result[2],
        test_result[3]
        )
    )

    # TimeSeriesSplit for k-fold validation for time series data
    # The data will be split into 5 consecutive folds, where each fold trains on 
    # a progressively larger portion of the dataset and tests on the subsequent time period.
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize an array to store the results of each fold
    fold_results = []

    # Iterate over each split in TimeSeriesSplit
    for train_index, test_index in tscv.split(X):
        # Split data into training and testing sets
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        # Train the model
        # early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, mode='min')
        # model.fit(X_train_k, y_train_k, epochs=EPOCHS, callbacks=[early_stop], verbose=0)

        # Evaluate the model on the test set
        y_pred_k = model.predict(X_test_k)
        test_mse = mean_squared_error(y_test_k, y_pred_k)

        # Store the result
        fold_results.append(test_mse)
        print(f"Test MSE for fold: {test_mse}")

    # Average performance across all folds
    average_mse = np.mean(fold_results)
    print(f"Average Test MSE across all folds: {average_mse}")

    plt.plot(range(1, len(fold_results) + 1), fold_results, marker='o')
    plt.title('MSE on Each Fold')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.show()


def plot_model():

    # Create dataframe from 1D numy array
    df = pd.DataFrame(data, columns=['Close'])

    # Predict using the trained model
    # Predict for x_train, X_test
    # print(X_train.shape)
    # exit()
    train_predict = model.predict(X_train)
    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)
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
        prediction = model.predict(future_data)
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

    # Plot Learning Curves
    # Plot learning curves (loss vs. epochs and metrics vs. epochs) to check for overfitting or underfitting.

    # history = model.history  # Assuming you've stored the history object during training

    # print(history)
    # plt.plot(history.history['loss'], label='train_loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.legend()
    # plt.title('Loss Curve')
    # plt.show()

    # plt.plot(history.history['metric'], label='train_metric')
    # plt.plot(history.history['val_metric'], label='val_metric')
    # plt.legend()
    # plt.title('Metric Curve')
    # plt.show()


evaluate_model()
plot_model()

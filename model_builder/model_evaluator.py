# Import necessary libraries
import os
import sys
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from variable_config import LOOK_BACK as look_back, file_name, EPOCHS
from data_wrangler import data_wrangler, split_into_datasets
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create determinism and model reproducibility
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# Call data_wrangler to create features and label
X, y, data, scaler = data_wrangler(file_name, look_back)

# Hold out validation data
X_train, X_val, X_test, y_train, y_val, y_test = split_into_datasets(X, y, look_back, get_val_set=True)

def evaluate_model():

    # Load saved model
    model = keras.models.load_model('hyper_model/best_model/best_model.keras')

    # Model summary
    print(model.summary())
    
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

    # https://forecastegy.com/posts/time-series-cross-validation-python/
    # So make sure your data is sorted before using this method.
    # This tool automates the expanding window method, that expands the training set while keeping constant the test set.
    # TimeSeriesSplit respects the temporal order of your data, ensuring that the ‘future’ data is not used to train your model.
    # TimeSeriesSplit for k-fold validation for time series data
    # The data will be split into 5 consecutive folds, where each fold trains on 
    # a progressively larger portion of the dataset and tests on the subsequent time period.

    k_fold = 7
    tscv = TimeSeriesSplit(n_splits=k_fold)

    # Initialize an array to store the results of each fold
    fold_results = {
        'r2_score': [],
        'mape': []
    }

    # Iterate over each split in TimeSeriesSplit
    for train_index, test_index in tscv.split(X):
        # Split data into training and testing sets
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        # Train the model
        early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, mode='min')
        model.fit(X_train_k, y_train_k, epochs=EPOCHS, callbacks=[early_stop], verbose=0)

        # Evaluate the model on the test set
        y_pred_k = model.predict(X_test_k)
        test_r2_score = r2_score(y_test_k, y_pred_k)
        test_mape = mean_absolute_percentage_error(y_test_k, y_pred_k)

        # Store the result
        fold_results['r2_score'].append(test_r2_score)
        fold_results['mape'].append(test_mape)
        print(f"Test R2_Score for fold: {test_r2_score}")
        print(f"Test MAPE for fold: {test_mape}")

    # Average performance across all folds
    average_r2_score = np.mean(fold_results['r2_score'])
    average_mape = np.mean(fold_results['mape'])
    print(f"Average Test R2_Score across all folds: {average_r2_score}")
    print(f"Average Test MAPE across all folds: {average_mape}")

    # Max values
    min_r2_score = np.min(fold_results['r2_score'])
    max_mape = np.max(fold_results['mape'])

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(f'{k_fold}-Fold Validation with TimeSeriesSplit')
    x_range = range(1, len(fold_results['r2_score']) + 1)
    # R2 Score
    ax1.plot(x_range, fold_results['r2_score'], marker='o', c='green', linestyle='--')
    # ax1.set_title('R2_Score on Each Fold')
    ax1.set_ylabel('R2_Score')

    # MAPE
    ax2.plot(x_range, fold_results['mape'], marker='o', c='red', linestyle='--')
    # ax2.set_title('MAPE on Each Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('MAPE')
    for i in range(len(fold_results['r2_score'])):
        # ax1.text(i+1, fold_results['r2_score'][i] + 0.01,  # Offset the y-position slightly
        #      f'({i+1}, {fold_results["r2_score"][i]:.2f})', fontsize=9, color='red', ha='center')
        if (fold_results['r2_score'][i] == min_r2_score):
            xytext_value = (0, 10)
        else:
            xytext_value = (0, -15)

        ax1.annotate(f'({fold_results["r2_score"][i]:.3f})', 
            xy=(i+1, fold_results['r2_score'][i]),      # Point being annotated
            xytext= xytext_value,                       # Offset the label slightly
            textcoords='offset points',                 # Use offset points for placement
            fontsize=9, color='red', ha='center')
        
    for i in range(len(fold_results['mape'])):
        if (fold_results['mape'][i] == max_mape):
            xytext_value = (0, -15)
        else:
            xytext_value = (0, 10)
        ax2.annotate(f'({fold_results["mape"][i]:.2f})', 
            xy=(i+1, fold_results['mape'][i]),     # Point being annotated
            xytext=xytext_value,                   # Offset the label slightly
            textcoords='offset points',            # Use offset points for placement
            fontsize=9, color='green', ha='center')

    plt.show()


def plot_model_prediction():

    # model = keras.models.load_model('hyper_model/best_model/best_model.keras')
    model = keras.models.load_model('hyper_model/best_model/best_model.keras')

    # Create dataframe from 1D numy array
    df = pd.DataFrame(data, columns=['Close'])

    # Predict using the trained model
    # Predict for x_train, X_test
    print(X_train.shape, X_val.shape, X_test.shape)
    train_predict = model.predict(X_train)
    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)
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

def plot_training_history():

    # Plot Learning Curves
    
    history = pd.read_csv('hyper_model/best_model/best_model_history')

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Trainig history of the best model')
    x_range = range(1, len(history['epoch']) + 1)
    # R2 Score
    ax1.plot(x_range, history['r2_score'], label='r2_score', marker='o', c='red')
    ax1.plot(x_range, history['val_r2_score'], label='val_r2_score', marker='o', c='green', linestyle='--')
    # ax1.set_title('R2_Score on Each Fold')
    ax1.set_ylabel('R2_Score')
    ax1.legend()

    # Loss
    ax2.plot(x_range, history['loss'], label='loss', marker='o', c='red')
    ax2.plot(x_range, history['val_loss'], label='val_loss', marker='o', c='green', linestyle='--')
    # ax2.set_title('MAPE on Each Fold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.legend()

    plt.show()

plot_model_prediction()
plot_training_history()
evaluate_model()

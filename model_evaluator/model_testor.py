# Import necessary libraries
import os
import sys
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
sys.path.append(os.path.abspath('') + os.path.sep + 'model_builder')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import dataframe_image as dfi
from scipy import stats
from tensorflow import keras
from variable_config import LOOK_BACK as look_back, file_name
from data_wrangler import data_wrangler, split_into_datasets
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from model_builder import hb_tuner_lstm, hb_tuner_gru

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create determinism and model reproducibility
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# Call data_wrangler to create features and label
# `prep_data` is sorted data after duplicate and null removal
# `data` is after unnecessary features removal, necessary features and label creation, and transformtion
X, y, scaler, raw_data, prep_data, data, unseen_data = data_wrangler(file_name, look_back)

# Hold out validation data
X_train, X_val, X_test, y_train, y_val, y_test = split_into_datasets(X, y, look_back, get_val_set=True)

def get_prev_best_trials(tuner):
    best_trial_ids = []
    best_trial_scores = []
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_trial_ids.append(best_trial.trial_id)
    best_trial_scores.append(best_trial.score)
    # Returns None if the key doesn't exist
    prev_best_trial_id = best_trial.hyperparameters.values.get('tuner/trial_id', None)
    
    while prev_best_trial_id is not None:
        best_trial_ids.append(prev_best_trial_id)
        best_trial_scores.append(best_trial.score)
        best_trial = tuner.oracle.get_trial(prev_best_trial_id)
        prev_best_trial_id = best_trial.hyperparameters.values.get('tuner/trial_id', None)
    # sort in ascending order.
    return sorted(best_trial_ids, reverse=False), sorted(best_trial_scores, reverse=False)

def get_prev_training_results(tuner):
    best_trial_ids, _ = get_prev_best_trials(tuner)
    print(best_trial_ids)
    # Dateframe to store training history from TensorBoard
    prev_training_results = pd.DataFrame()
        
    model_type = tuner.hypermodel.layer.__module__.split('.')[-1]
    
    for best_trial_id in best_trial_ids:
    
        best_trail_path = f'hyper_model/history/{model_type}/history_trial_{best_trial_id}.csv'
        new_df = pd.read_csv(best_trail_path)
        prev_training_results = pd.concat([prev_training_results, new_df], ignore_index=True)
    
    best_history_path = f'hyper_model/best_model/best_{model_type}_model_history.csv'
    new_df = pd.read_csv(best_history_path)
    prev_training_results = pd.concat([prev_training_results, new_df], ignore_index=True)
        
    return prev_training_results

def plot_and_test_series():

    # This code make sure that each time you run your code, your neural network weights will be initialized equally.
    from numpy.random import seed
    seed(42)
    from tensorflow import random
    random.set_seed(42)

    # Time series plot of NEPSE closing index
    # Create dataframe from 1D numy array
    df = pd.DataFrame(prep_data[['Date', 'Close']])
    plt.plot(df['Date'], df['Close'], label='Actual NEPSE Index')
    plt.xticks(rotation=45)
    plt.xlabel('Trading Date')
    plt.ylabel('Closing Index')
    plt.title('Time Series Plot of NEPSE Closing Index')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

    # Generate descriptive statistics.
    # dfi.export(prep_data['Close'].describe().to_frame().rename(columns={'Close': 'Value'}).round(2).transpose(), 'statistics.png')
    dfi.export(prep_data['Close'].describe().to_frame().round(2).transpose(), 'statistics.png')

    # Generate box plot to check if outlier exists.
    prep_data.boxplot(column=['Close'])

    # Plot the histogram of closing price.
    prep_data.plot.hist(column=['Close'], bins=10)
    plt.xlabel('NEPSE Closing Index')

    # Saves the dataframe as image, similar to Jupyter notebooks DataFrame format.
    dfi.export(raw_data.head(11), 'raw_data.png')
    plt.show()

def get_test_statistics_on_metric(metric):
    # Loss or MSE series
    lstm_metric = pd.read_csv('hyper_model/best_model/best_lstm_model_history.csv')[metric]
    gru_metric = pd.read_csv('hyper_model/best_model/best_gru_model_history.csv')[metric]
    
    # training_history_lstm = get_prev_training_results(hb_tuner_lstm)
    # training_history_gru = get_prev_training_results(hb_tuner_gru)
    # print(training_history_lstm[['loss', 'mean_squared_error']])
    # print(training_history_gru[['loss', 'mean_squared_error']])
    # lstm_metric=training_history_lstm['loss']
    # gru_metric=training_history_gru['loss']

    # Perform D'Agostino-Pearson's normality test
    lstm_d_p_stat, lstm_p_value = stats.normaltest(lstm_metric)
    gru_d_p_stat, gru_p_value = stats.normaltest(gru_metric)

    print(f"\nD'Agostino-Pearson Normality Test on {metric}:")
    print(f"(LSTM-Dense model, p-values) : ({round(lstm_d_p_stat, 6)}, {round(lstm_p_value, 6)})")
    print(f"(GRU-Dense model, p-values) : ({round(gru_d_p_stat, 6)}, {round(gru_p_value, 6)})")
    
    # Plot the heatmap to see the dependence between variables.
    # Create a dataframe from two ndarray
    df = pd.DataFrame({f'lstm_dense_{metric}': lstm_metric, f'gru_dense_{metric}': gru_metric})
    correlation_matrix = df.corr()
    plt.figure(figsize=(8, 6))  # Adjust size
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Heatmap of {metric}")
    plt.show()

    # The data is not normal. So, Wilcoxon Signed Rank Test, w-statistic, 
    # is used to compare the median difference between two groups.
    # It is a non-parametric version of the paired T-test.
    print(lstm_metric.shape, gru_metric.shape)
    min_of_series = min(len(lstm_metric), len(gru_metric))
    w_statistic, p_value = stats.wilcoxon(lstm_metric.iloc[:min_of_series], gru_metric.iloc[:min_of_series], zero_method="pratt")
    print(f"\nWilcoxon Signed-rank Test on {metric}:\n")
    print(f"(w_statistic, p_values) : ({np.round(w_statistic, 6)}, {np.round(p_value, 6)})")

    # Welch's t-test (Two sample)
    welchs_t_stat, p_value = stats.ttest_ind(lstm_metric, gru_metric, equal_var=False)
    print(f"\nWelch's T-Test on {metric}:")
    print(f"(Welch's t-statistic, p-value): ({round(welchs_t_stat, 6)}, {round(p_value, 6)})")


def get_test_statistics_on_pred_error():

    lstm_model = keras.models.load_model('hyper_model/best_model/best_lstm_model.keras')
    gru_model = keras.models.load_model('hyper_model/best_model/best_gru_model.keras')

    y_pred_lstm = lstm_model.predict(X_test).reshape(-1)
    y_pred_gru = gru_model.predict(X_test).reshape(-1)

    # Generating prediction error
    error_lstm = np.subtract(y_test.reshape(-1), y_pred_lstm)
    error_gru = np.subtract(y_test.reshape(-1), y_pred_gru)
    # error_lstm = np.array(scaler.inverse_transform(y_pred_lstm.reshape(-1,1)))
    # error_gru = np.array(scaler.inverse_transform(y_pred_gru.reshape(-1,1)))

    # Create a dataframe from two ndarray
    # df = pd.DataFrame({'error_lstm': error_lstm.reshape(-1,), 'error_gru': error_gru.reshape(-1,)})
    # correlation_matrix = df.corr()

    # Plot the heatmap
    # plt.figure(figsize=(8, 6))  # Adjust size
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    # plt.title("Correlation Heatmap")
    # plt.show()

    # The data is not normal. So, Wilcoxon Signed Rank Test, w-statistic, 
    # is used to compare the median difference between two groups.
    # It is a non-parametric version of the paired T-test.
    w_statistic, p_value = stats.wilcoxon(error_lstm, error_gru, zero_method="pratt")
    print("\nWilcoxon Signed-rank Test:\n")
    # print(f"(w_statistic, p_values) : ({round(w_statistic, 6)}, {round(p_value, 6)})")
    print(f"(w_statistic, p_values) : ({np.round(w_statistic, 6)}, {np.round(p_value, 6)})")
    
    print('\n\n')
      
    # Get all trials from the tuner
    # trials = hb_tuner_lstm.oracle.trials
    # Print out the details of each trial, focusing on the round and epoch
    # for trial_id, trial in trials.items():
    #     hp = trial.hyperparameters.values
    #     # Returns None if the key doesn't exist
    #     prev_trial_id = hp.get('tuner/trial_id', None)
    #     print(f"  - Trial {trial_id}:")
    #     print(f"  - Hyperparameters {trial.hyperparameters.values}:")
    #     print(f"  - Trial id {trial.trial_id}:")
    #     print(f"  - Best step: {trial.best_step}")
    #     print(f"  - Score: {trial.score}")
    #     print(f"  - Metrics _observation value: {trial.metrics.metrics['val_loss']._observations[0].value[0]}")
    #     print(f"  - Status: {trial.status}")
    #     print("----------")
    
    # exit()
    
    # Perform D'Agostino and Pearson's normality test
    lstm_d_p_stat, lstm_p_value = stats.normaltest(error_lstm)
    gru_d_p_stat, gru_p_value = stats.normaltest(error_gru)

    print("\nD'Agostino-Pearson Normality Test on prediction errors:")
    print(f"(LSTM-Dense model, p-values) : ({np.round(lstm_d_p_stat, 6)}, {np.round(lstm_p_value, 6)})")
    print(f"(GRU-Dense model, p-values) : ({np.round(gru_d_p_stat, 6)}, {np.round(gru_p_value, 6)})")

    # Welch's t-test (Two sample)
    welchs_t_stat, p_value = stats.ttest_ind(y_pred_lstm, y_pred_gru, equal_var=False)
    print("\nWelch's T-Test on prediction error:")
    print(f"(Welch's t-statistic, p-value): ({round(welchs_t_stat, 6)}, {round(p_value, 6)})")
    
# plot_and_test_series()
get_test_statistics_on_metric('mean_squared_error')
get_test_statistics_on_metric('r2_score')
# get_test_statistics_on_pred_error()

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
from tensorflow.python.summary.summary_iterator import summary_iterator

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


def get_initial_epoch(log_dir):
    # Locate the TensorBoard event files
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        raise FileNotFoundError("No event files found in the specified log directory.")
    
    # Initialize variables
    last_epoch = -1
    
    # Read the event files
    for event_file in event_files:
        file_path = os.path.join(log_dir, event_file)
        for event in summary_iterator(file_path):
            for value in event.summary.value:
                if value.tag == 'epoch':  # This assumes 'epoch' is logged
                    last_epoch = max(last_epoch, int(value.simple_value))
    
    return last_epoch + 1  # Start from the next epoch


def get_prev_best_trials(tuner):
    best_trial_ids = []
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_trial_ids.append(best_trial.trial_id)
    # Returns None if the key doesn't exist
    prev_best_trial_id = best_trial.hyperparameters.values.get('tuner/trial_id', None)
    
    while prev_best_trial_id is not None:
        best_trial_ids.append(prev_best_trial_id)
        best_trial = tuner.oracle.get_trial(prev_best_trial_id)
        prev_best_trial_id = best_trial.hyperparameters.values.get('tuner/trial_id', None)
    # sort in ascending order.
    return sorted(best_trial_ids, reverse=False)

def get_test_statistics_on_pred_error():

    lstm_model = keras.models.load_model('hyper_model/best_model/best_lstm_model.keras')
    gru_model = keras.models.load_model('hyper_model/best_model/best_gru_model.keras')

    y_pred_lstm = lstm_model.predict(X_test).reshape(-1)
    y_pred_gru = gru_model.predict(X_test).reshape(-1)

    # Generating prediction error
    # error_lstm = np.subtract(y_test.reshape(-1), y_pred_lstm)
    # error_gru = np.subtract(y_test.reshape(-1), y_pred_gru)
    error_lstm = np.array(scaler.inverse_transform(y_pred_lstm.reshape(-1,1)))
    error_gru = np.array(scaler.inverse_transform(y_pred_gru.reshape(-1,1)))

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

    # Extract training history from TensorBoard
    # Best trials from previous    
    best_trial_ids = get_prev_best_trials(hb_tuner_lstm)
    print('best_trial_ids: ', best_trial_ids)
    # List variable for training loss
    training_loss = []
    
    model_type = hb_tuner_lstm.hypermodel.layer.__module__.split('.')[-1]
    
    for best_trial_id in best_trial_ids:
    
        best_trail_path = f'hyper_model/tensor_board/{model_type}_logs/{best_trial_id}'
        
        # Find the .tfevents file
        for root, dirs, files in os.walk(best_trail_path, topdown=True):
            # for file in files:
            if not files:
                # files is empty.
                pass
            else:
                file = files[0] # for training loss
                # file = files[1] # for validation loss
        
                if 'train' in root:
                # if 'validation' in root:
                    # print(root)
                    # print(os.path.basename(os.path.dirname(os.path.normpath(root))))
                    execution_dirname = os.path.basename(os.path.dirname(os.path.normpath(root)))
                    if file.startswith("events.out.tfevents"):
                        event_file = os.path.join(root, file)
                        
                        for event in summary_iterator(event_file):
                             for value in event.summary.value:
                                print('value_tag:', value.tag)
                                #  if value.tag == 'epoch':  # This assumes 'epoch' is logged
                                #     last_epoch = max(last_epoch, int(value.simple_value))
                                #     print(last_epoch)
                                #     exit()
                        
                        # data = {'epoch': [], 'loss': []}
                        # # Iterate over events in the file
                        # for event in tf.compat.v1.train.summary_iterator(event_file):
                        #     for value in event.summary.value:
                        #         if value.tag == 'epoch':
                        #             data['epoch'].append(value.simple_value)
                        #         elif value.tag == 'loss':
                        #             data['loss'].append(value.simple_value)
                                    
                        # Load the .tfevents file
                        event_acc = EventAccumulator(event_file)
                        event_acc.Reload()  # Load the data

                        # Retrieve all scalar tags
                        all_tensors = event_acc.Tags()['tensors']
                        # print('Available all tensors:', all_tensors)
                        # Print the data
                        for tag in all_tensors:
                            # writing the loss
                            if tag == 'epoch_loss':
                                loss_data = event_acc.Tensors(tag)
                                trial_specific_loss = []
                                for event in loss_data:
                                    tensor_data = event.tensor_proto
                                    tensor_content = np.frombuffer(tensor_data.tensor_content, dtype=np.float32)
                                    # print(vars(event), tensor_content)
                                    # print(f'Tag: {tag}, Step: {event.step}, Tensor_Data:{tensor_content[0]}')
                                    # Convert training loss to DataFrame
                                    epoch =event.step+1
                                    trial_specific_loss.append((epoch, tensor_content[0]))
                                trial_specific_loss_df = pd.DataFrame(trial_specific_loss, columns=['Epoch', 'Training Loss'])
                                # trial_specific_loss_df = trial_specific_loss_df.groupby(by='Epoch', as_index=False).mean()
                                print(trial_specific_loss_df)
            # print (root)
            # print (dirs)
            # print (files)
            # print ('----------------------')
    # Convert training loss to DataFrame
    # training_loss_df = pd.DataFrame(training_loss, columns=['Execution', 'Epoch', 'Training Loss'])
    # training_loss_df_avgs = training_loss_df.groupby(by='Execution', as_index=False).mean()
    # print(training_loss_df)
    # print(training_loss_df_avgs)
    exit()
    
    # Perform D'Agostino and Pearson's normality test
    lstm_d_p_stat, lstm_p_value = stats.normaltest(error_lstm)
    gru_d_p_stat, gru_p_value = stats.normaltest(error_gru)

    print("\nD'Agostino-Pearson Normality Test on prediction errors:")
    print(f"(LSTM-Dense model, p-values) : ({round(lstm_d_p_stat, 6)}, {round(lstm_p_value, 6)})")
    print(f"(GRU-Dense model, p-values) : ({round(gru_d_p_stat, 6)}, {round(gru_p_value, 6)})")

    # Welch's t-test (Two sample)
    welchs_t_stat, p_value = stats.ttest_ind(y_pred_lstm, y_pred_gru, equal_var=False)
    print("\nWelch's T-Test on prediction error:")
    print(f"(Welch's t-statistic, p-value): ({round(welchs_t_stat, 6)}, {round(p_value, 6)})")

def get_test_statistics_on_mse():
    # MSE series
    lstm_metrics = pd.read_csv('hyper_model/best_model/best_lstm_model_history.csv')['loss']
    gru_metrics = pd.read_csv('hyper_model/best_model/best_gru_model_history.csv')['loss']
    # print(lstm_metrics)
    # print(gru_metrics)

    # Perform D'Agostino-Pearson's normality test
    lstm_d_p_stat, lstm_p_value = stats.normaltest(lstm_metrics)
    gru_d_p_stat, gru_p_value = stats.normaltest(gru_metrics)

    print("\nD'Agostino-Pearson Normality Test on MSE:")
    print(f"(LSTM-Dense model, p-values) : ({round(lstm_d_p_stat, 6)}, {round(lstm_p_value, 6)})")
    print(f"(GRU-Dense model, p-values) : ({round(gru_d_p_stat, 6)}, {round(gru_p_value, 6)})")
    
    # Plot the heatmap to see the dependence between variables.
    # Create a dataframe from two ndarray
    df = pd.DataFrame({'error_lstm': lstm_metrics, 'error_gru': gru_metrics})
    correlation_matrix = df.corr()
    plt.figure(figsize=(8, 6))  # Adjust size
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    # The data is not normal. So, Wilcoxon Signed Rank Test, w-statistic, 
    # is used to compare the median difference between two groups.
    # It is a non-parametric version of the paired T-test.
    print(lstm_metrics.shape, gru_metrics.shape)
    min_of_series = min(len(lstm_metrics), len(gru_metrics))
    w_statistic, p_value = stats.wilcoxon(lstm_metrics.iloc[:min_of_series], gru_metrics.iloc[:min_of_series], zero_method="pratt")
    print("\nWilcoxon Signed-rank Test:\n")
    print(f"(w_statistic, p_values) : ({np.round(w_statistic, 6)}, {np.round(p_value, 6)})")

    # Welch's t-test (Two sample)
    welchs_t_stat, p_value = stats.ttest_ind(lstm_metrics, gru_metrics, equal_var=False)
    print("\nWelch's T-Test on MSE:")
    print(f"(Welch's t-statistic, p-value): ({round(welchs_t_stat, 6)}, {round(p_value, 6)})")

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

# plot_and_test_series()
get_test_statistics_on_pred_error()
# get_test_statistics_on_mse()

# Requires variable definitions for LOOK_BACK, MAX_EPOCHS, and DISABLE_RESUME
import os
import sys
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
import keras_tuner as kt
from variable_config import MAX_EPOCHS, DISABLE_RESUME, LOOK_BACK, executions_per_trial
from tensorflow import keras
from keras.api.layers import LSTM, GRU
import pandas as pd
import numpy as np
# import pickle
import csv


# This code make sure that each time you run your code, your neural network weights will be initialized equally.
from numpy.random import seed
seed(42)
from tensorflow import random
random.set_seed(42)

# Define the custom tuner.
class CustomTuner(kt.Hyperband):
    
    # Check if run_trail() is running for the first time.
    first_run = True
    
    # Create dictionary values of matrics per index.
    def create_dict_keys_values(self, history):    
        # List container for dictionary
        dict_list = []
        # Create dictionary keys
        dict_keys = ['epoch'] + list(history.history.keys())
        
        for i in range(len(history.epoch)):
            epoch = history.epoch[i] + 1
            dict_values_per_index = [
                epoch,
                history.history['loss'][i],
                history.history['mean_absolute_error'][i],
                history.history['mean_absolute_percentage_error'][i],
                history.history['mean_squared_error'][i],
                history.history['r2_score'][i],
                history.history['val_loss'][i],
                history.history['val_mean_absolute_error'][i],
                history.history['val_mean_absolute_percentage_error'][i],
                history.history['val_mean_squared_error'][i],
                history.history['val_r2_score'][i],
            ]
            new_dict = dict(zip(dict_keys, dict_values_per_index))
            dict_list.append(new_dict)
        return dict_list
    
    def run_trial(self, trial, *args, **kwargs):
        
        # List to hold dictionaries
        list_of_dicts = []
        trial_id = trial.trial_id
        # Unpack hyperparameters
        hp = trial.hyperparameters
        # Extract model type
        model_type = self.hypermodel.layer.__module__.split('.')[-1]
        csv_file = f'hyper_model/history/{model_type}/history_trial_{trial_id}.csv'
        # Create the directory if it doesn't exist
        # The exist_ok=True ensures no error is raised if the directory already exists.
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
                
        for _ in range(self.executions_per_trial):
            # Customize kwargs with successive epoch and initial_epoch
            kwargs['epochs']=hp['tuner/epochs']
            kwargs['initial_epoch']=hp['tuner/initial_epoch']
            
            # Build and train the model
            model = self.hypermodel.build(hp)
            history = model.fit(
                *args,
                batch_size=hp.Choice('batch_size', values=[8, 16, 24, 32, 40, 48, 56]),
                **kwargs
            )
        
            # # Save the history for this trial
            # trial.history = history.history
        
            # print('history:', history.history)
            print('epoch:', history.epoch)

            # Report the final result
            # Pass the min validation metrics to the tuner
            self.oracle.update_trial(trial_id, {'val_loss': min(history.history['val_loss'])})
            self.oracle._save_trial(trial)
            # self.oracle.save()
            
            dicts = self.create_dict_keys_values(history)
            # Concantenate the dicts on each loop.
            list_of_dicts = list_of_dicts + dicts
            
            # If resume is disable, then delete the .csv files.
            if self.first_run and DISABLE_RESUME:
                for root, _, files in os.walk(os.path.dirname(csv_file), topdown=True):
                    for file in files:
                        if '.csv' in file:
                            os.remove(os.path.join(root, file))
                # Make first_run false only if it is True for furthur execution of run_trail().
                if self.first_run == True:
                    self.first_run = False
        
        
        # Calculate average across execution per trials.
        # Create a DataFrame
        df = pd.DataFrame(list_of_dicts)
        # Replace inf value with nan
        # df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Group by 'epoch' to calculate averages and preserve 'epoch' as column header
        df_avgs = df.groupby(by='epoch', as_index=False, dropna=True).mean()
        # print(df_avgs[['mean_squared_error','val_mean_squared_error']])
        # Convert the result to dictionary object as records.
        list_of_dicts_avgs = df_avgs.to_dict(orient='records')
        dict_keys = list(df_avgs.columns)
          
        # Save history to csv file
        # if not file exits, then open with 'w' mode
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as file:
                
                writer = csv.DictWriter(
                    file,
                    fieldnames=dict_keys
                )
                # Write header (column names)
                writer.writeheader()
                for dict in list_of_dicts_avgs:
                    # Write rows
                    writer.writerows([dict])
        
        # Save the history for this trial
        # trial.history = history.history
        
           
        # Save the training history to a file for each trial
        # history_file = f'hyper_model/history/history_trial_{trial_id}.pkl'
        
        # with open(history_file, 'wb') as f:
        #     pickle.dump(history.history, f)


# Define the Hybrid model structure.       
class CustomHyperModel(kt.HyperModel):
    
    def __init__(self, layer=LSTM):
        self.layer = layer
        
    def build(self, hp):
        model = keras.Sequential()
        # This is the recommended approach to define input shape in keras
        model.add(keras.layers.Input(shape=(LOOK_BACK, 1)))  # Define input shape here

        activation = hp.Choice('activation', ['relu', 'tanh', 'sigmoid', 'linear', 'swish'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        num_layers = hp.Int('num_layers', default=1, min_value=1, max_value=5)
        optimizers = hp.Choice('optimizer', values =('SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam'))
        loss = hp.Choice('loss', values=('mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'))
        # metrics = hp.Choice('metric', values=('mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'))

        for i in range(num_layers):
            model.add(
                self.layer(
                    units=hp.Int(f'units_{i}', min_value=1, max_value=200, step=8),
                    activation=activation, 
                    return_sequences=True if i < num_layers-1 else False
                    # input_shape=(LOOK_BACK, 1)
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
        # As a alternative solution
        # https://stackoverflow.com/questions/67286051/how-can-i-tune-the-optimization-function-with-keras-tuner

        # Optimizer matching
        optimizers_dict = {
            'SGD': keras.optimizers.SGD(learning_rate=learning_rate),
            'RMSprop': keras.optimizers.RMSprop(learning_rate=learning_rate),
            'Adam': keras.optimizers.Adam(learning_rate=learning_rate),
            'Adadelta': keras.optimizers.Adadelta(learning_rate=learning_rate),
            'Adagrad': keras.optimizers.Adagrad(learning_rate=learning_rate),
            'Adamax': keras.optimizers.Adamax(learning_rate=learning_rate),
            'Nadam': keras.optimizers.Nadam(learning_rate=learning_rate)
        }

        optimizer = optimizers_dict[optimizers]
                    
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
            # metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'r2_score']
            metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError(), keras.metrics.MeanAbsolutePercentageError(), keras.metrics.R2Score()]
            # metrics=[metrics]
        )

        return model


# Hyperband Tuner for LSTM-Dense Model
hb_tuner_lstm = CustomTuner(
    hypermodel=CustomHyperModel(LSTM),
    objective=kt.Objective(name='val_loss', direction='min'),
    # objective=kt.Objective(name='val_mean_absolute_error', direction='min'),
    
    # This is the maximum possible number of epochs per model. Sets the maximum number of epochs for each trial.
    # The maximum resources (epochs) that can be allocated to a model.
    max_epochs=MAX_EPOCHS, # dafault value is 100.
    # The reduction factor.
    # Only the top one-third of models are retained in a bracket, and the rest are pruned in case of default value.
    factor=2, # dafault value is 3.
    seed=42,
    # Disable the usage of the existing state from trail_ logs to resume the search.
    overwrite=DISABLE_RESUME,
    # Trail contrains a set of hyperparameters, run a single trail multiple times is to reduce results variance 
    # and therefore be able to more accurately assess the performance of a model.
    # It is number of times each set of hyperparameters is used to train and evaluate the model for 
    # reducing the impact of randomness and ensure the robust results.
    executions_per_trial=executions_per_trial,
    directory='hyper_model',
    project_name='tuning_lstm'
)

# Hyperband Tuner for GRU-Dense Model
hb_tuner_gru = CustomTuner(
    hypermodel=CustomHyperModel(GRU),
    objective=kt.Objective(name='val_loss', direction='min'),
    # objective=kt.Objective(name='val_mean_absolute_error', direction='min'),
    
    # This is the maximum possible number of epochs per model. Sets the maximum number of epochs for each trial.
    # The maximum resources (epochs) that can be allocated to a model.
    max_epochs=MAX_EPOCHS, # dafault value is 100.
    # The reduction factor.
    # Only the top one-third of models are retained in a bracket, and the rest are pruned in case of default value.
    factor=2, # dafault value is 3.
    seed=42,
    # Disable the usage of the existing state from trail_ logs to resume the search.
    overwrite=DISABLE_RESUME,
    # Trail contrains a set of hyperparameters, run a single trail multiple times is to to reduce results variance 
    # and therefore be able to more accurately assess the performance of a model.
    executions_per_trial=executions_per_trial,
    directory='hyper_model',
    project_name='tuning_gru'
)
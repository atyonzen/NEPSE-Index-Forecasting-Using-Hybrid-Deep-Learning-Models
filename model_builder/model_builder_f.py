# Requires variable definitions for LOOK_BACK, MAX_EPOCHS, and DISABLE_RESUME
import os
import sys
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
from variable_config import MAX_EPOCHS, DISABLE_RESUME, LOOK_BACK, executions_per_trial
import keras_tuner as kt
from tensorflow import keras
from keras.api.layers import LSTM, GRU


# This code make sure that each time you run your code, your neural network weights will be initialized equally.
from numpy.random import seed
seed(42)
from tensorflow import random
random.set_seed(42)

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
    
    def fit(self, hp, model, *args, **kwargs):
        # lr = hp.get('learning_rate')
        print('epochs:', hp['tuner/epochs'],
              'initial_epoch:', hp['tuner/initial_epoch'], 
              'bracket:', hp['tuner/bracket'], 
              'round:', hp['tuner/round']
              )
        if 'tuner/trial_id' in hp:
            # past_trial = hp['tuner/trial_id'] # self.oracle.get_trial(hp['tuner/trial_id'])
            print('trail_id:', hp['tuner/trial_id'])
        print('\n\n')
        # print(vars(model))
        # print(kwargs.keys())
        print('Epochs from kwargs:\n', kwargs['epochs'])
        # print(args[0].shape, args[1].shape)
        # print(kwargs.keys())
        # kwargs["epochs"] value is set by max_epochs in HyperBand tuner object
        # max_epochs is causing duplicate values when hypertuning epochs in model.fit() function.
        # Therefore, epochs value is revomed from kwargs.
        # if 'epochs' in kwargs:
        #     kwargs.pop('epochs')

        # Ensure validation data is passed correctly
        if 'validation_data' not in kwargs:
            raise ValueError("Validation data must be provided in the fit method for validation metrics to be logged.")
        
        # history = model.fit(
        return model.fit(
            *args,
            # validation_data=kwargs['validation_data'],
            # Tune the appropriate batch size
            batch_size=hp.Choice('batch_size', values=[8, 16, 24, 32, 40, 48, 56]),
            # epochs=hp.Int('epochs', min_value=1, max_value=500, sampling='log'),
            **kwargs,
        )
        # print(vars(history))
        # print(history.params)
        # print(history.epoch)
        # print(history.history)
        # return history


# Hyperband Tuner for LSTM-Dense Model
hb_tuner_lstm = kt.Hyperband(
    CustomHyperModel(LSTM),
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
hb_tuner_gru = kt.Hyperband(
    CustomHyperModel(GRU),
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
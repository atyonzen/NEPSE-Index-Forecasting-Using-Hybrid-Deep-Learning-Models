# Setup the environment varibles
import os
# import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Add path to custom modules
# sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')

# Define custom variables
MAX_EPOCHS = 200  # max_epochs = 8 or 10 seems good for large or small sample
EPOCHS = 200
LOOK_BACK = 5 # window size for time series data
DISABLE_RESUME = False
# DISABLE_RESUME = True
# csv file name without file extension
file_name = 'nepsealpha'
# file_name = 'nepse_indices'
executions_per_trial=5
# Number of future days to predict
future_steps = 15
# Number of folds for time series split
k_fold = 5
# UNSEEN = True
UNSEEN = False
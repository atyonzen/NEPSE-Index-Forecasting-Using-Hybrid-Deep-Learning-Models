# Setup the environment varibles
import os
# import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Add path to custom modules
# sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')

# Define custom variables
EPOCHS = 500
MAX_EPOCHS = 42  # max_epochs = 8 or 10 seems good for large or small sample
LOOK_BACK = 5 # window size for time series data
DISABLE_RESUME = False
# DISABLE_RESUME = True
# csv file name without file extension
file_name = 'nepsealpha0c'
executions_per_trial=5
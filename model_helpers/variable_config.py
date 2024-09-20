# Setup the environment varibles
import os
# import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Add path to custom modules
# sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')

# Define custom variables
EPOCHS = 200
MAX_EPOCHS = 3 # max_epochs = 8 or 10 seems good for large or small sample
LOOK_BACK = 7 # window size for time series data
DISABLE_RESUME = False
# csv file name without file extension
file_name = 'nepsealpha0c'
executions_per_trial=10
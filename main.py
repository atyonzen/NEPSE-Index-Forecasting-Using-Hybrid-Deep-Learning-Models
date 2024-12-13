# Data collection with web scraper
# Data processing cleaning, normalizing, spliting
# Model selection
# Training the model
# Evaluating the model
# Hyperparameter tuning and optimization

import os
import sys
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
from variable_config import file_name
os.system('python model_helpers/variable_config.py')
os.system('python model_helpers/data_wrangler.py')

# If file_name does not exit, then execute web_scrapper.py
file_path = f'./data_src/{file_name}.csv'
if(not os.path.isfile(file_path)):
    os.system('python model_helpers/web_scrapper.py')

# os.system('python model_builder/model_builder.py')
# os.system('python model_builder/model_trainer.py')
os.system('python model_evaluator/lstm_model_evaluator.py')
os.system('python model_evaluator/gru_model_evaluator.py')
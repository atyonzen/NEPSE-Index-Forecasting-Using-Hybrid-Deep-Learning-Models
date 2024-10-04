# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from variable_config import LOOK_BACK

# Prepares feature and lebel according to the window size
# LOOK_BACK is the number of previous days' prices (window size) to consider for prediction
def data_wrangler(file_name, look_back=LOOK_BACK, transform_data = True):

    # read csv file
    df = pd.read_csv(f'data_src/{file_name}.csv')

    # remove columns which the neural network will not use
    df = df.drop(['Symbol', 'Date', 'Open', 'High', 'Low', 'Percent Change', 'Volume'], axis=1)

    # Transform data
    if(transform_data):
        scaler = MinMaxScaler()
        # In reshape(-1, 1), -1 means to infer the number of rows automatically and column 1. 
        # This reshaping is needed because the MinMaxScaler expects the input data to be 2D 
        # (with rows as individual data points and columns as features).
        df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    # Numpy ndarray
    data = df['Close'].values

    X, y = [], []
    for i in range(len(data)):
        end_ix = i + look_back
        # handle last item which may be like 228+1 > 229-1
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    if(transform_data):
        return np.array(X), np.array(y), data, scaler
    else:
        return np.array(X), np.array(y), data


# Splits dataset into training, validation and testing sets
def split_into_datasets(X, y, look_back = LOOK_BACK, test_size = 0.2, val_size = 0.125, get_val_set = False):
    
    # Test_size in fractional number between 0 and 1 segregates the proportion of data. 
    # If integer value is provided at test_size, it seggregates that numbers of data as test set.
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42, shuffle=False)
    y_train, y_test = train_test_split(y, test_size=test_size, random_state=42, shuffle=False)


    # It is assumed to create validation set from the training set.
    if(get_val_set):
        X_train, X_val = train_test_split(X_train, test_size=val_size, random_state=42, shuffle=False)
        y_train, y_val = train_test_split(y_train, test_size=val_size, random_state=42, shuffle=False)
        # print(X_train.shape)
        # print(X_val.shape)
        # print(X_test.shape)
        # print(X.shape)

        # Reshape datasets into [samples, timesteps, features] to satisfy LSTM requirement
        X_train = X_train.reshape(-1, look_back, 1)
        X_val = X_val.reshape(-1, look_back, 1)
        X_test = X_test.reshape(-1, look_back, 1)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train = X_train.reshape(-1, look_back, 1)
        X_test = X_test.reshape(-1, look_back, 1)
        return X_train, X_test, y_train, y_test


# Creates train, validation and test datasets
def train_val_test_split(X, y, look_back = LOOK_BACK, test_split = 0.8, val_split = 0.2):
    
    # Split the data into training, validation and testing sets
    # It is assumed to create validation set from training set
    training_size = int(len(X) * test_split)
    validation_size = int(len(X) * val_split)
    X_train, y_train = X[:training_size - validation_size], y[:training_size - validation_size]
    X_val, y_val = X_train[-validation_size:], y_train[-validation_size:]
    X_test, y_test = X[training_size:], y[training_size:]
    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)
    # (155, 7)
    # (22, 7)
    # (45, 7)

    # Reshape the input data for LSTM
    # Reshape input data into [samples, timesteps, features]
    X_train = X_train.reshape(-1, look_back, 1)
    X_val = X_val.reshape(-1, look_back, 1)
    X_test = X_test.reshape(-1, look_back, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test

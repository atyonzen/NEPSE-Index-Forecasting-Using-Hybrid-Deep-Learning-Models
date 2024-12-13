# Import necessary libraries
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from variable_config import LOOK_BACK, future_steps

# Prepares feature and lebel according to the window size
# LOOK_BACK is the number of previous days' prices (window size) to consider for prediction
def data_wrangler(file_name, look_back=LOOK_BACK, transform_data = True, unseen=False):

    # read csv file
    raw_data = pd.read_csv(f'data_src/{file_name}.csv')

    # Drop last duplicate on Date column.
    # print(raw_data.duplicated(subset=['Date'], keep='last').loc[lambda x : x == True])
    prep_data = raw_data.drop_duplicates(subset=['Date'], keep='last', inplace=False)

    # Drop the rows where at least one element is missing.
    # print(raw_data.isnull())
    prep_data = prep_data.dropna()

    # Convert the string date to python datatime object.
    prep_data['Date'] = pd.to_datetime(arg=prep_data['Date'], format='%d-%m-%y')

    # Usage of assignment operator along with inplace parameter results in NoneType data.
    # Avoid using assignment operator when you use inplace parameter.
    prep_data.sort_values(by=['Date'], inplace=True)

    # remove columns which the neural network will not use
    # df = df.drop(['Symbol', 'Date', 'Open', 'High', 'Low', 'Percent Change', 'Volume'], axis=1)
    df = prep_data[['Close']]

    # Transform data
    if(transform_data):
        scaler = MinMaxScaler()
        # In reshape(-1, 1), -1 means to infer the number of rows automatically and column 1. 
        # This reshaping is needed because the MinMaxScaler expects the input data to be 2D 
        # (with rows as individual data points and columns as features).
        df.loc[:, 'Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1)).flatten().astype('float64')
    
    # Numpy ndarray
    data = df['Close'].values

    # Spare out-of-sample data to visually evaluate the prediction made by walk forward validation.
    if(unseen):
        unseen_data = data[-future_steps:]
        data = data[: len(data) - future_steps]
    else:
        unseen_data = None

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
        return np.array(X), np.array(y), scaler, raw_data, prep_data, data, unseen_data
    else:
        return np.array(X), np.array(y), None, raw_data, prep_data, data, unseen_data


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

    # Reshape the input data for LSTM
    # Reshape input data into [samples, timesteps, features]
    X_train = X_train.reshape(-1, look_back, 1)
    X_val = X_val.reshape(-1, look_back, 1)
    X_test = X_test.reshape(-1, look_back, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test

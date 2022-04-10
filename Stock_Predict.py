# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:04:32 2022

@author: alfre
"""

import math
import pandas_datareader as pdr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime as dt
from tkinter import *

root = Tk()

tick = Entry(root, width=50)
tick.pack()


def click():
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2021, 11, 30)
    df = pdr.get_data_yahoo(tick.get(), start, end)

    # Scale the data
    data = df.filter(['Close'])
    dataset = data.values
    training_data_length = math.ceil(len(dataset)*.8)

    # Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training data set
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_length, :]
    # Split the data into x_train and y_train
    xtrain = []
    ytrain = []

    for i in range(60, len(train_data)):
        xtrain.append(train_data[i-60:i, 0])
        ytrain.append(train_data[i, 0])

    # Convert the xtrain and ytrain to numpy arrays
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)

    # Reshape the data
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
              input_shape=(xtrain.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False,))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(xtrain, ytrain, batch_size=1, epochs=1)

    # Create the testing data set
    # Create a new array containing scaled values from index 1548 to 2003
    test_data = scaled_data[training_data_length - 60:, :]
    # Create the data sets xtest and ytest
    xtest = []
    ytest = dataset[training_data_length:, :]
    for i in range(60, len(test_data)):
        xtest.append(test_data[i-60:i, 0])

    # Convert the data to a numpy array
    xtest = np.array(xtest)

    # Reshape the data
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

    # Get models predicted price values
    predictions = model.predict(xtest)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    RMSE = rmse = np.sqrt(np.mean((predictions - ytest)**2))

    # Plot data
    train = data[:training_data_length]
    valid = data[training_data_length:]
    valid['Predictions'] = predictions

    # Visualization

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Closed Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


btn = Button(root, text="Enter ticker", command=click)
btn.pack()

root.mainloop()

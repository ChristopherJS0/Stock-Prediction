import os
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras import layers
TF_ENABLE_ONEDNN_OPTS = 0

def createAiModel(xSet, ySet, xVals, yVals, epochs):
    '''
    param: set of X values to train on.
    param: set of Y values to train on.
    param: set of x validation values.
    param: set of y validation values.
    param: Number of times to train the model

    rtype: Model created by tensorflow using the LSTM learning technique.
    '''
    
    model = tf.keras.models.Sequential([tf.keras.layers.Input((xSet.shape,ySet.shape)),
                        tf.keras.layers.LSTM(64),
                        tf.keras.layers.Dense(32, activation='relu'),
                        tf.keras.layers.Dense(32, activation='relu'),
                        tf.keras.layers.Dense(1)])
    
    model.compile(loss='mse',
              optimizer='Adam',
              metrics=['mean_absolute_error'])
 
    model.fit(x=xSet, y=ySet, validation_data=(xVals, yVals), epochs=epochs) 

    return model

def predictFutureVals(dateVals, dateTests, xSet):
    recursivePredictions = []
    recursiveDates = np.concatenate([dateVals, dates_test])

    for targetDate in recursiveDates:
        lastWindow = deepcopy(X_train[-1])
        next_pred = model.predict(np.array([lastWindow])).flatten()
        recursivePredictions.append(next_pred)
        lastWindow[-1] = next_pred
    
    return recursiveDates, recursivePredictions

if __name__ == '__main__':
    # Make sure there's a csv file filled with the company's stock history.
    StockDF = pd.read_csv('AppleStock.csv')
    
    '''plt.plot(StockDF['Date'], StockDF['Close'])
    plt.title('Dates and Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Value')
    '''

    # Preparing data to use for the training.
    xVals = StockDF['Date'].to_numpy()
    yVals = StockDF['Close'].to_numpy()

    print(xVals.shape)
    
    q_80 = int(len(yVals) * .8) # 80% of allDates
    q_90 = int(len(yVals) * .9) # 90% of allDates

    # Training sets. First 80% of data used to train.
    dates_train, X_train, y_train = xVals[:q_80], xVals[:q_80], yVals[:q_80]
    # Validation data. Next 10% of data used to validate the training.
    dates_val, X_val, y_val = xVals[q_80:q_90], xVals[q_80:q_90], yVals[q_80:q_90]
    # Testing data.
    dates_test, X_test, y_test = xVals[q_90:], xVals[q_90:], yVals[q_90:]

    # Get the AI model to train, val, and test.
    model = createAiModel(X_train, y_train, X_val, y_val, 100)
    
    # Plotting the training graph.
    trainPreds = model.predict(X_train).flatten()
    plt.plot(dates_train, trainPreds)
    plt.plot(dates_train, y_train)
    plt.legend(['Training Predictions', 'Training Observations'])

    # Plotting the validation prediction graph.
    val_predictions = model.predict(X_val).flatten()
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.legend(['Validation Predictions', 'Validation Observations'])

    # Plotting the testing prediction graph.
    test_predictions = model.predict(X_test).flatten()
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Testing Predictions', 'Testing Observations'])

    #rDates, rPreds = predictFutureVals(dates_val, dates_test, X_train)

    plt.plot(dates_train, trainPreds)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    #plt.plot(rDates, rPreds)
    plt.legend(['Training Predictions', 
                'Training Observations',
                'Validation Predictions', 
                'Validation Observations',
                'Testing Predictions', 
                'Testing Observations'])


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime as dt

def createAiModel(xSet, ySet, xVal, yVal, epochs):
    '''
    param: set of X values to train on.
    param: set of Y values to train on.
    param: set of x validation values.
    param: set of y validation values.
    param: Number of times to train the model

    rtype: Model created by tensorflow using the LSTM learning technique.
    '''
    model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(2,1)),
                        tf.keras.layers.LSTM(128),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(1)])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mean_absolute_error'])
 
    model.fit(x=xSet, y=ySet, validation_data=(xVal, yVal), epochs=epochs) 

    return model

def createFuturePreds(model, lastY, futureDays):
    fullPreds = []

    for _ in range(futureDays):
        predY = model.predict(lastY).flatten()
        lastY = np.vstack((lastY, predY[-1]))
        fullPreds.append(predY[-1])

    return fullPreds

if __name__ == '__main__':
    # Make sure there's a csv file filled with the company's stock history.
    StockDF = pd.read_csv('AppleStock.csv')

    # Preparing data to use for the training.
    dates = StockDF['Date'].to_numpy()
    StockDF['Date'] = pd.to_datetime(StockDF['Date'])
    StockDF.set_index('Date', inplace=True)

    xVals = StockDF[['Close']]
    yVals = StockDF['Close']

    y_scaled = yVals.to_numpy().reshape(-1,1)

    q_80 = int(len(StockDF) * .8) # 80% of allDates
    q_90 = int(len(StockDF) * .9) # 90% of allDates

    # Training sets. First 80% of data used to train.
    X_train, y_train = xVals[:q_80], y_scaled[:q_80]
    # Validation data. Next 10% of data used to validate the training.
    X_val, y_val = xVals[q_80:q_90], y_scaled[q_80:q_90]
    # Testing data.
    X_test, y_test = xVals[q_90:], y_scaled[q_90:]

    datesTrain = dates[:q_80]
    datesVal = dates[q_80:q_90]
    datesTest = dates[q_90:]

    # Get the AI model to train, val, and test.
    model = createAiModel(X_train, y_train, X_val, y_val, 100)

    # Plotting the training graph.
    trainPreds = model.predict(y_train).flatten()
    plt.plot(datesTrain, trainPreds)
    plt.plot(datesTrain, y_train)
    plt.legend(['Training Predictions', 'Training Observations'])
    plt.show()

    # Plotting the testing prediction graph.
    test_predictions = model.predict(y_test).flatten()
    plt.plot(datesTest, test_predictions)
    plt.plot(datesTest, y_test)
    plt.legend(['Testing Predictions', 'Testing Observations'])
    plt.show()

    # Predicting future values based off training from final value.
    numOfNextDays = 50 # Generate dates from today until 30 days from now
    futurePreds = createFuturePreds(model, y_test, numOfNextDays)
    nextDays = np.concatenate((test_predictions, futurePreds))
    days = list(range(1,len(nextDays)+1))
    plt.plot(days, nextDays)
    plt.legend(['Future Predictions for stock Values.'])
    plt.show()
    
    model.summary()
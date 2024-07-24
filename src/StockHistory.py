import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import NewsSentimentRate as NSR
from datetime import datetime, timedelta

def prepareData(StockDF):
    '''
    stockDF : Datatframe that will be prepared for analyzing
    '''

    # Preparing data to use for the training.
    dates = StockDF['Date'].to_numpy()
    StockDF['Date'] = pd.to_datetime(StockDF['Date'])
    StockDF.set_index('Date', inplace=True)

    xVals = StockDF[['Close']]
    yVals = StockDF['Close']

    yVals = yVals.to_numpy().reshape(-1,1)

    q_80 = int(len(StockDF) * .8) # 80% of allDates
    q_90 = int(len(StockDF) * .9) # 90% of allDates

    X_train, y_train = xVals[:q_80], yVals[:q_80] # Training sets. First 80% of data used to train.   
    X_val, y_val = xVals[q_80:q_90], yVals[q_80:q_90] # Validation data. Next 10% of data used to validate the training.   
    X_test, y_test = xVals[q_90:], yVals[q_90:] # Testing data.

    datesTrain = dates[:q_80]
    datesVal = dates[q_80:q_90]
    datesTest = dates[q_90:]

    return X_train, y_train, X_val, y_val, X_test, y_test, datesTrain, datesVal, datesTest

def createAiModel(xSet, ySet, xVal, yVal, epochs):
    '''
    xSet: set of X values to train on.
    ySet: set of Y values to train on.
    xVal: set of x validation values.
    yVal: set of y validation values.
    epochs: Number of times to train the model

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

def makeTicker():
    '''
    rtype: Returns a CSV file that will be used to analyze stock data.
    ''' 
    end_date = datetime.today()
    start_date = end_date - timedelta(days=1825)
    userTicker = input('Please enter a valid stock ticker to predict stock from: ')
    stockData = yf.download(userTicker,start=start_date,end=end_date)

    if stockData.empty:
        print("No data found for the given ticker, please enter a valid ticker!")
        makeTicker()
    else:
        #Saving the stock data.
        stockCSV = stockData.to_csv('Stock.csv')
        return stockCSV, userTicker

def evaluateModel(predictions, values):
    '''
    @predictions: The values predicted by the model.
    @values: The official values found in the stock history.

    rtype: List of mae, mse and rmse
    '''

    mae = tf.keras.losses.MeanAbsoluteError()(values, predictions).numpy()
    mse = tf.keras.losses.MeanSquaredError()(values, predictions).numpy()
    rmse = np.sqrt(mse)

    return { mae, mse, rmse }

def drawGraphs(x, y1: tuple, y2: tuple):
    '''
    @x : The x axis values, such as dates.
    @y1 : The y axis values such as stock values and the label of values.
    @y2 : The y axis values such as prediction stock values and the label of values.
    '''

    plt.plot(x, y1[0])
    plt.plot(x, y2[0])
    plt.legend([y1[1], y2[1]])
    plt.show()

if __name__ == '__main__':
    _, Company = makeTicker()
    StockDF = pd.read_csv('Stock.csv')

    # Preparing data to use for the training.
    X_train, y_train, X_val, y_val, X_test, y_test, \
        datesTrain, datesVal, datesTest = prepareData(StockDF)

    # Get the AI model to train, val, and test.
    model = createAiModel(X_train, y_train, X_val, y_val, 100)

    # Plotting the training graph.
    trainPreds = model.predict(y_train).flatten()
    trainPredTuple = (trainPreds, 'Training Predictions')
    trainDataTuple = (y_train, 'Training Data')
    drawGraphs(datesTrain, trainPredTuple, trainDataTuple)
    trainMAE, trainMSE, trainRMSE = evaluateModel(trainPreds, y_train)

    # Plotting the testing prediction graph.
    test_predictions = model.predict(y_test).flatten()
    testPredTuple = (test_predictions, 'Test Predictions')
    testDataTuple = (y_test, 'Test Data')
    drawGraphs(datesTest, testPredTuple, testDataTuple)
    testMAE, testMSE, testRMSE = evaluateModel(test_predictions, y_test)

    # Predicting future values based off training from final value.
    numOfNextDays = 50 # Generate dates from today until 30 days from now
    futurePreds = createFuturePreds(model, y_test, numOfNextDays)
    nextDays = np.concatenate((test_predictions, futurePreds))
    days = list(range(1,len(nextDays)+1))
    plt.plot(days, nextDays)
    plt.legend(['Future Predictions for stock Values.'])
    plt.show()

    print('Training evaluation values:')
    print(f'Training MAE: {trainMAE}')
    print(f'Training MSE: {trainMSE}')
    print(f'Training RMSE: {trainRMSE}\n')
    print(f'Testing MAE: {testMAE}')
    print(f'Testing MSE: {testMSE}')
    print(f'Testing RMSE: {testRMSE}')
    NSR.rate_all_news(Company)
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras import Adam
from tensorflow.python.keras import layers

def makeValidDates(rawDate):
    SplitDate = rawDate.split('-')
    year,month,day = int(SplitDate[0]), int(SplitDate[1]), int(SplitDate[2])
    return datetime.datetime(year=year, month=month, day=day)

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    firstDate = makeValidDates(first_date_str)
    lastDate = makeValidDates(last_date_str)

    target_date = firstDate

    dates = []
    X, Y = [], []
    last_time = False

    while True:
        # View until target date, the last n+1 elemns of
        df_subset = dataframe.loc[:target_date].tail(n+1)
        if len(df_subset) != n+1: # If subset doesn't have n+1 elems.
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy() # Make subset into array.
        x, y = values[:-1], values[-1] # x = all vals until last, y = last val.

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date: target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]

        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

    target_date = next_date

    if target_date == lastDate:
      last_time = True

    ret_df = pd.DateFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
      X[:, i] # Until the ith element of the real X list into the array.
      ret_df[f'Target-{n-i}'] = X[:, i]

    ret_df['Target'] = Y
    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:,0]
    middle_matrix = df_as_np[:, 1: -1] # Get all rows, get the 2nd to the 2nd to last cols.
    #   Tuple of amount of vecs (len(dates)), and 3 vecs, for 1 elem of vec
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

if __name__ == '__main__':
# Make sure there's a csv file filled with the company's stock history.
    StockDF = pd.read_csv('Stock.csv')
    StockDF['Date'] = StockDF['Date'].apply(makeValidDates)
    StockDF.index = StockDF.pop('Date')
    plt.plot(StockDF.index, StockDF['Close'])

    #Take a year of stock data up to now.
    today = datetime.now().date()
    today = str(today)
    lastYear = list(today)
    lastYear[3] = str(int(lastYear[3]) - 1)
    lastYear = ''.join(lastYear)

    yearDF = df_to_windowed_df(StockDF, lastYear, today, n=3)
    allDates,bigX, y = windowed_df_to_date_X_y(yearDF)


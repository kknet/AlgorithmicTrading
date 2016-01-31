__author__ = 'kmanda1'

import numpy as np
import pandas as pd
import os
import statistics as stats

def symbol_to_path(symbol, base_dir=os.path.join(".", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_adj_closing(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""

    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        df = df.dropna()
    return df


def getData(symbols, dates):
    price_df = get_adj_closing(symbols, dates)
    Y_df = stats.Y(price_df).fillna(method='ffill')
    BB_df = stats.BollingerBand(price_df).fillna(0)
    V_df = stats.Volatility(price_df).fillna(0)
    M_df = stats.Momentum(price_df).fillna(0)
    total_rows = len(Y_df)
    data = np.zeros(shape=(total_rows,4))
    price_array = np.zeros(shape=(total_rows,1))
    #prices = price_df.fillna(method='ffill')

    for i in xrange(total_rows):
       data[i,0]=BB_df.ix[i]
       data[i,1]=V_df.ix[i]
       data[i,2]=M_df.ix[i]
       data[i,3]=Y_df.ix[i]

       price_array[i,0] = price_df.ix[i]

    return data , price_array , price_df

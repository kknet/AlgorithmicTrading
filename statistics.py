__author__ = 'kknet'

import pandas as pd

def Volatility(prices):
    return pd.rolling_std(prices, window=20)

def Momentum(prices):
    momentum = (prices/prices.shift(-20))-1
    return momentum

def BollingerBand(prices):
    sma = pd.rolling_mean(prices, window=20)
    std = pd.rolling_std(prices, window=20)
    df = (prices - sma)/(2*std)
    return df

def Y(prices):
    return (prices.shift(-5)/prices) - 1.0

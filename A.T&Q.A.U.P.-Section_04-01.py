# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:03:07 2019

@author: neelkanth mehta
"""
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
from mpl_finance import candlestick2_ohlc
from matplotlib.dates import MONDAY, DateFormatter, MonthLocator, WeekdayLocator, DayLocator

# Setting dates and other constants
start = dt.date.today() - dt.timedelta(1825)
end   = dt.date.today()

ticker = 'MSFT'

# Fetching data and obtaining a DF
ohlcv = pdr.get_data_yahoo(ticker, start, end)

df = ohlcv.copy()

# Generating MACD technical indicator variable
def MACD(df, a, b, c):
    df['MAF'] = df['Adj Close'].ewm(span=a, min_periods=a).mean()
    df['MAS'] = df['Adj Close'].ewm(span=b, min_periods=b).mean()
    df['MACD']= df['MAF'] - df['MAS']
    df['Sig'] = df['MACD'].ewm(span=c, min_periods=c).mean()
    df.dropna(axis=0, inplace=True)
    return df


def ATR(Df,n):
    """Function to calculate True Range and Average True Range"""
    df = Df.copy()
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC']= abs(df['High'] - df['Adj Close'].shift(1))
    df['L-PC']= abs(df['Low']  - df['Adj Close'].shift(1))
    df['TR']  = df[['H-L','H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df2 = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    df2.dropna(axis=0, inplace=True)
    return df2


def bollinger_bands(Df,n):
    """A function to calculate Bollinger Band"""
    df = Df.copy()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['BBup'] = df['MA20'] + (2 * df['Adj Close'].rolling(window=20).std())
    df['BBdn'] = df['MA20'] - (2 * df['Adj Close'].rolling(window=20).std())
    df.dropna(inplace=True)
    return df


def RSI(Df, n):
    """Function to calculate RSI"""
    df = Df.copy()
    df['delta'] = df['Adj Close'].diff()
    df['gain']  = np.where(df['delta']>=0,df['delta'],0)
    df['loss']  = np.where(df['delta']<0, abs(df['delta']),0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(window=n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(window=n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n-i)*avg_gain[i-1]+ gain[i])/n)
            avg_loss.append(((n-i)*avg_gain[i-1]+ loss[i])/n)
    df['avg_gain'] = np.array(avg_gain)
    df['avg_loss'] = np.array(avg_loss)
    df['RS'] = df['avg_gain']/ df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    df.drop(labels=['delta', 'gain', 'loss', 'avg_gain', 'avg_loss'], inplace=True, axis=1)
    return df


# demo
df2 = MACD(ohlcv.copy(), 10, 24, 8)
df2 = ATR(Df=df2.copy(), n=20)
df2 = bollinger_bands(Df=df2.copy(), n=20)
df2 = RSI(Df=df2.copy(), n=20)


df2.loc['2019-06','MA20':].plot();
plt.show()


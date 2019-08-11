# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 07:14:56 2019

@author: neelkanth mehta
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import statsmodels.api as sm

# For plotting 
#import matplotlib.dates as mdates
#from mpl_finance import candlestick_ohlc
#from matplotlib.dates import MONDAY, DateFormatter, DayLocator, WeekdayLocator

# Start and end date
beg = dt.date.today() - dt.timedelta(days=1825)
end = dt.date.today()

# Tickers
ticker = 'MSFT'

# Downloading the date and creating a dataframe
ohlcv = web.get_data_yahoo(ticker, beg, end)

'''Create an MACD indicator'''
def MACD(Df, a, b, c):    
    df = Df.copy()
    df['MAfast'] = df['Adj Close'].ewm(span=a, min_periods=a).mean()
    df['MAslow'] = df['Adj Close'].ewm(span=b, min_periods=b).mean()
    df['MACD']= df['MAfast'] - df['MAslow']
    df['MACDsig']= df['MACD'].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    df.drop(labels=['MAfast','MAslow'], inplace=True, axis=1)
    return df

macddf = MACD(ohlcv, 12, 26, 9)

# Plotting output
macddf.iloc[:, -3:].plot()
plt.legend(frameon=False, title=None)
plt.show()


'''Average True Range'''
# Defining ATR function
def ATR(Df, n, ewm=False):
    '''Calculates and returns Average True Range'''
    df = Df.copy()
    df['HML'] = abs(df['High'] - df['Low'])
    df['HMPC']= abs(df['High'] - df['Adj Close'].shift(1))
    df['LMPC']= abs(df['Low'] - df['Adj Close'].shift(1))
    df['TR']  = df[['HML','HMPC','LMPC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean() if ewm == False else df['TR'].ewm(span=n, min_periods=n).mean()
    df2 = df.drop(labels=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'HML', 'HMPC', 'LMPC'], axis=1)
    return df2

# Generating ATR DataFrame
atrdf = ATR(ohlcv, 20, ewm=True)

# Plotting the output
atrdf['ATR'].plot()
plt.legend(title=None, frameon=False)
plt.show()

'''Bollinger Bands'''
# Defining the function
def bollinger_bands(Df, n, BBrange=False):
    '''returns DataFrame with Bollinger Bands'''
    df = Df.copy()
    df['MA'] = df['Adj Close'].rolling(n).mean()
    df['BBup'] = df['MA'] + df['MA'].rolling(n).std() * 2
    df['BBdn'] = df['MA'] - df['MA'].rolling(n).std() * 2
    if BBrange == True:
        df['BBrange'] = df['BBup'] - df['BBdn']
    df.dropna(inplace=True)
    df.drop(labels=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], axis=1, inplace=True)
    return df

# Generating Bollinger Bands
bbdf = bollinger_bands(ohlcv, 20)

# Plotting the output
plt.figure(figsize=(10,6))
ohlcv.loc['2019':,'Adj Close'].plot()
bbdf.loc['2019':, 'MA'].plot(linestyle='-.', alpha=2)
bbdf.loc['2019':,'BBup'].plot(linestyle=':', alpha=0.7)
bbdf.loc['2019':,'BBdn'].plot(linestyle=':', alpha=0.7)
plt.legend(frameon=False, title=None)
plt.title('Bollinger Bands')
plt.grid()
plt.show()

'''Relative Strength Index'''
# Defining the function
def RSI(Df, n):
    '''Function to calculate RSI'''
    df = Df.copy()
    df['Delta'] = df['Adj Close'].diff().dropna()
    df['gain']  = np.where(df['Delta'] >= 0, df['Delta'], 0)
    df['loss']  = np.where(df['Delta'] < 0, abs(df['Delta']), 0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n-1)*avg_gain[i-1] + gain[i])/ n)
            avg_loss.append(((n-1)*avg_gain[i-1] + loss[i])/ n)
    df['avg_gain'] = np.array(avg_gain)
    df['avg_loss'] = np.array(avg_loss)
    df['RS'] = df['avg_gain']/ df['avg_loss']
    df['RSI']= 100 - (100/(1+df['RS']))
    return df['RSI']

# Fetching the RSI
rsidf = RSI(ohlcv, 14)

'''Average Directional Index'''
def ADX(Df, n):
    '''Function to calculate ADX'''
    df = Df.copy()
    df['TR'] = ATR(Df, n)['TR']
    df['DMplu'] = np.where((df['High']-df['High'].shift(1))>(df['Low']-df['Low'].shift(1)), (df['High']-df['High'].shift(1)), 0)
    df['DMplu'] = np.where(df['DMplu']<0,0, df['DMplu'])
    df['DMmin'] = np.where((df['Low']-df['Low'].shift(1))>(df['High']-df['High'].shift(1)), (df['Low']-df['Low'].shift(1)), 0)
    df['DMmin'] = np.where(df['DMmin']<0,0, df['DMmin'])
    TRn = []
    DMpluN = []
    DMminN = []
    TR = df['TR'].tolist()
    DMplu = df['DMplu'].tolist()
    DMmin = df['DMmin'].tolist()
    for i in range(len(df)):
        if i < n:
            TRn.append(np.NaN)
            DMpluN.append(np.NaN)
            DMminN.append(np.NaN)
        elif i == n:
            TRn.append(df['TR'].rolling(n).sum().tolist()[n])
            DMpluN.append(df['DMplu'].rolling(n).sum().tolist()[n])
            DMminN.append(df['DMmin'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1]-(TRn[i-1]/ 14) + TR[i])
            DMpluN.append(DMpluN[i-1] - (DMpluN[i-1]/ 14) + DMplu[i])
            DMminN.append(DMminN[i-1] - (DMminN[i-1]/ 14) + DMmin[i])
    df['TRn'] = np.array(TRn)
    df['DMpluN'] = np.array(DMpluN)
    df['DMminN'] = np.array(DMminN)
    df['DIpluN'] = 100 * (df['DMpluN']/ df['TRn'])
    df['DIminN'] = 100 * (df['DMminN']/ df['TRn'])
    df['DIdiff'] = abs(df['DIpluN'] - df['DIminN'])
    df['DIsum']  = df['DIpluN'] + df['DIminN']
    df['DX']     = 100 * (df['DIdiff']/ df['DIsum'])
    ADX = []
    DX = df['DX'].tolist()
    for j in range(len(df)):
        if j < 2 * (n-1):
            ADX.append(np.NaN)
        elif j == 2 * (n - 1):
            ADX.append(df['DX'][j-n+1:j+1].mean())
        elif j > 2 * (n - 1):
            ADX.append(((n-1)*ADX[j-1] + DX[j])/n)
    df['ADX'] = np.array(ADX)
    return df['ADX']

# Fetching ADX values
adxdf = ADX(ohlcv, 14)

'''On Balance Volume'''
def OBV(Df):
    '''Returns OBV Series'''
    df = Df.copy()
    df['ret'] = df['Adj Close'].pct_change()
    df['dir'] = np.where(df['ret'] >=0, 1, -1)
    df['dir'][0] = 0
    df['dirvol'] = df['dir'] * df['Volume']
    df['OBV'] = df['dirvol'].cumsum()
    return df['OBV']

# Fetching OBV form the dataset
obvdf = OBV(ohlcv)

'''Slope'''
def slope(ser, n):
    '''Function to calculate the slope of n consecutive points on a plot'''
    slopes = [i * 0 for i in range(n-1)]
    for i in range(n, len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/ (y.max() - y.min())
        x_scaled = (x - x.min())/ (x.max() - x.min())
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

# Fetching the data
ser = slope(ohlcv['Adj Close'].copy(), 10)

# Plotting the output
plt.plot(ser[:50])
plt.title('Slope')
plt.ylabel('Level')
plt.xlabel('Time')
plt.show()


'''Renko Chart'''
df = ohlcv.copy()
df.index = df.index.date
df.reset_index(inplace=True)
df.drop(labels=['Close'], axis=1, inplace=True)
df.rename(columns={'Adj Close': 'Close', 'index': 'Date'}, inplace=True)
df = df.iloc[:,[0,3,1,2,-1, -2]]
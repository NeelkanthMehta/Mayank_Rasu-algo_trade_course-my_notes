# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:05:26 2019

@author: neelkanth mehta
"""

import copy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from alpha_vantage.timeseries import TimeSeries
from stocktrends import Renko


'''Defining requisite technical indicators and performance metrics'''

def ATR(Df, n):
    '''
    Computes average true range of a price series
    
    Parameters
    ==========
    df: pd.Series(): high
    l: pd.Series(): low
    c: pd.Series(): close
    n: int: number of days of which to compute average
    '''
    # Copying original price series so as not to edit the original
    high, low, close = copy.deepcopy(h), copy.deepcopy(l), copy.deepcopy(c)
    
    # Computing High (-) Low
    hml = abs(high - low)
    
    # Computing High (-) Previous Close
    # Computing Low  (-) Previous close
    hmpc= abs(high - close.shift(1))
    lmpc= abs(low  - close.shift(1))
    
    # True Range
    tr  = pd.concat([hml, hmpc, lmpc], axis=1).max(axis=1, skipna=False)
    
    # Returns Rolling Mean Series of True Eange
    return tr.rolling(n).mean()


def slope(Ser, n):
    '''
    Computes Slope of n consecutive points on a plot
    
    Parameters
    ==========
    Ser: pandas.Series(): enter the price series for which you want to calculate slope
    n  : int            : number of days
    '''
    # Copying the original price series so as not to edit it
    ser = copy.deepcopy(Ser)
    
    # Instantiating the zeroes array later to be filled by the following loop
    slopes = [i * 0 for i in range(n-1)]
    
    # Looping through series to standardize series and generate slope
    for i in range(n, len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/ (y.max() - y.min())
        x_scaled = (x - x.min())/ (x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    
    # Converting radians to angle
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    
    # Returns an array of slope values
    return np.array(slope_angle)


def renko(o, h, l, c, v):
    '''
    Computes Ranko values for the given price series
    
    Parameters
    ==========
    o: pandas.Series(): Open price
    h: pandas.Series(): High price
    l: pandas.Series(): Low  price
    c: pandas.Series(): Closeprice
    v: pandas.Series(): Volume
    
    Returns
    =======
    renko_df: pandas.DataFrame()
    '''
    df = pd.concat([copy.deepcopy(o), copy.deepcopy(h), copy.deepcopy(l), copy.deepcopy(c), copy.deepcopy(v)], axis=1)
    df.reset_index(inplace=True)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df2 = Renko(df)
    renko_df = df2.get_bricks()
    renko_df['bar_num'] = np.where(renko_df['uptrend']==True, 1,np.where(renko_df['uptrend']==False,-1, 0))
    for i in range(1, len(renko_df['bar_num'])):
        if renko_df['bar_num'][i]>0 & renko_df['bar_num'][i-1] > 0:
            renko_df['bar_num'][i] += renko_df['bar_num'][i-1]
        elif renko_df['bar_num'][i] < 0 & renko_df['bar_num'][i-1] < 0:
            renko_df['bar_num'][i] += renko_df['bar_num'][i-1]
    renko_df.drop_duplicates(subset='Date', keep='last', inplace=True)
    return renko_df


def OBV(c, v):
    '''
    Function to compute on balance volume of a give price series
    
    Parameters
    ==========
    c: pandas.Series(): close price of the price series
    v: pandas.Series(): volume of the price series
    
    Returns
    =======
    on balance volume: pandas.Series()
    '''
    # Copying the series for convenience
    df = pd.concat([copy.deepcopy(c), copy.deepcopy(v)], axis=1)
    
    # Defining parameters
    df.columne = ['Close', 'Volume']
    df['ret'] = df['Close'].pct_change().fillna(0)
    df['dir'] = np.where(df['ret']>=0, 1, -1)
    df['dir'][0] = 0
    df['vol'] = df['Volume'] - df['dir']
    
    # Returns on balance volume of a series
    return df['vol'].cumsum()


def CAGR(Ser) -> float:
    '''
    Computes compounded average growth rate of asecurity in question
    
    Parameters
    ==========
    Ser: pandas.Series: enter price series of a security you wish to compute CAGR
    logscale:  boolean: select True if you want log returns
    
    Returns
    =======
    CAGR: float
    '''
    ser = Ser.copy()
    ret = ser.pct_change().fillna(0)
    cumret = ret.add(1).cumprod()
    n = len(cumret)/ 252
    CAGR = (cumret[-1])**(1/n) - 1
    return CAGR


def ann_vol(Ser) -> float:
    '''
    Computes annual volatility of a price series of a security
    
    Parameters
    ==========
    Ser: pandas.Series(): enter price series of a security you wish to compute annual volatility
    logscale: bool value: If selected True, will use volatility on log-scale returns to compute volatility
        
    Returns
    =======
    annual volatiliy: float
    '''
    ser = Ser.copy()
    ret = ser.pct_change().fillna(0)
    vol = ret.std()
    return vol * np.sqrt(252)


def sharpe(Ser, rf:float = 0)-> float:
    '''
    Computes Sharpe Ratio of a price series
    
    Parameters
    ==========
    Ser: pandas.Series(): enter price series of a security you wish to compute Sharpe
    logscale: bool value: If selected True, will use log-scale returns to compute Sharpe ratio
    rf:     string value: should be the risk-free rate against which you want to compute Sharpe
    
    Returns
    =======
    Sharpe Ratio: string
    '''
    cagr = CAGR(Ser=Ser)
    vol  = ann_vol(Ser=Ser)
    return (cagr - rf)/ vol


def max_dd(Ser) -> float:
    '''
    Computes Max  Drawdown of a price series
    
    Parameters
    ==========
    Ser: pandas.Series(): enter price series of a security you wish to compute Max Drawdown
    logscale: bool value: If selected True, will use log-scale returns to compute Max Drawdown
    rf:      float value: should be the risk-free rate against which you want to compute Sortino
    
    Returns
    =======
    Max Drawdown: float
    '''
    ser = Ser.copy()
    ret = ser.pct_change().fillna(0)
    cumret = ret.add(1).cumprod()
    cumrollmax = cumret.cummax()
    dd = cumrollmax - cumret
    dd_pct = dd/ cumrollmax
    return dd_pct.max()

'''Downloading historical data for tech stocks'''

tickers = ['MSFT', 'AAPL', 'FB', 'AMZN', 'INTC', 'CSCO', 'VZ', 'IBM', 'QCOM', 'LYFT']
ohlcvintraday = {}
vantage_api_key = 'OL0DOX9N42V2C5T3'
ts = TimeSeries(key=vantage_api_key, output_format='pandas')

attempt = 0
drop = {}
while len(tickers) > 0 and attempt <= 5:
    tickers = [j for j in tickers if j not in drop]
    for i in range(len(tickers)):
        try:
            ohlcvintraday[tickers[i]] = ts.get_intraday(symbol=tickers[i], interval='5min', outputsize='full')[0]
            ohlcvintraday[tickers[i]].columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            drop.append(tickers[i])
        except:
            print('%s failed to fetch data....retrying' % tickers[i])
            continue
    attempt += 1

tickers = ohlcvintraday.keys() # redefine tickers variable after removing any NaN tickers


'''Backtesting'''
# Merging Renko DF with original ohlcv df
ohlcvrenko = {}
df = copy.deepcopy(ohlcvintraday)
ticker_signal = {}
ticker_ret    = {}
for ticker in tickers:
    print('Merging for %s' % ticker)
    renkodf = renko(df[ticker]['Open'], df[ticker]['High'], df[ticker]['Low'], df[ticker]['Close'], df[ticker]['Volume'])
    df[ticker]['Date'] = df[ticker].index
    ohlcvrenko[ticker] = df[ticker].merge(renkodf.loc[:,['Date', 'bar_num']], how='outer', on='Date')
    ohlcvrenko[ticker]['bar_num'].fillna(method='ffill', inplace=True)
    ohlcvrenko[ticker]['OBV'] = OBV(df[ticker]['Close'], df[ticker]['Volume'])
    ohlcvrenko[ticker]['OBV_slope'] = slope(ohlcvrenko[ticker]['OBV'], 5)
    ticker_signal[ticker] = ""
    ticker_ret[ticker] = []


# Identifying signals and calculating daily returns
    for ticker in tickers:
        print('Calculating daily returns for %s' % ticker)
        for i in range(len(ohlcvintraday[ticker])):
            if ticker_signal[ticker] == "":
                ticker_ret[ticker].append(0)
                if ohlcvrenko[ticker]['bar_num'][i] >= 2 and ohlcvrenko[ticker]['OBV_slope'][i] > 30:
                    ticker_signal[ticker] = "Buy"
                elif ohlcvrenko[ticker]['bar_num'][i] <= -2 and ohlcvrenko[ticker]['OBV_slope'][i] <= -30:
                    ticker_signal[ticker] = "Sell"
            
            elif ticker_signal[ticker] == "Buy":
                ticker_ret[ticker].append((ohlcvrenko[ticker]['Close'][i]/ ohlcvrenko[ticker]['Close'][i-1])-1)
                if ohlcvrenko[ticker]['bar_num'][i] <= -2 and ohlcvrenko[ticker]['OBV_shape'][i] < -30:
                    ticker_signal[ticker] = "Sell"
                elif ohlcvrenko[ticker]['bar_num'][i] < 2:
                    ticker_signal[ticker] = ""
            
            elif ticker_signal[ticker] == "Sell":
                ticker_ret[ticker].append((ohlcvrenko[ticker]['Close'][i-1]/ ohlcvrenko[ticker]['Close'][i])-1)
                if ohlcvrenko[ticker]['bar_num'][i] >= 2 and ohlcvrenko[ticker]['OBV_slope'][i] > 30:
                    ticker_signal[ticker] = "Buy"
                elif ohlcvrenko[ticker]['bar_num'][i] > -2:
                    ticker_signal[ticker] = ""
        ohlcvrenko[ticker]['ret'] = np.array(ticker_ret[ticker])


# Calculating overall strategy API
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlcvrenko[ticker]['ret']

strategy_df['ret'] = strategy_df.mean(axis=1)
CAGR(strategy_df)
sharpe(strategy_df, 0.025)
max_dd(strategy_df)

# Visualizing strategy returns
(1 + strategy_df['ret']).cumprod().plot()


# Calculating strategy returns
cagr = {}
sharpe_ratio = {}
max_drawdown = {}
for ticker in tickers:
    print('Calculating KPIs for %s' % ticker)
    cagr[ticker] = CAGR(ohlcvrenko[ticker])
    sharpe_ratio[ticker] = sharpe(ohlcvrenko[ticker], 0.025)
    max_drawdown[ticker] = max_dd(ohlcvrenko[ticker])

KPI_df = pd.DataFrame([cagr, sharpe_ratio, max_drawdown], index=['Return', 'Sharpe', 'Max_DD'])
KPI_df.T
# -*- coding: utf-8 -*-
"""
Backtesting Strategy 3: Combining Renko Charts and MACD indicator
The features of the strategy are as follows:-

1. Choose high volume, high activity stocks for this strategy (pre market movers, historically high volume stocks etc.)

2. Buy Signal:
    * Renko bar GEQ 2.
    * MACD line is above signal line
    * MACD line's slope (over last 5 periods) is GEQ signal line's slope (over last 5 periods)
    * Exit when MACD line goes below the signal line and MACD line's slope is lower than signal line's slope
    
3. Sell Signals:
    * Renko bar LEQ -2
    * MACD line is below the signal line
    * MACD line's slope (over past 5 periods) is LEQ signal line's slope (over last 5 periods)
    * Exit when MACD line goes above the signal line and MACD line's slope is greater than signal line's slope

Created on Sat Aug 10 21:04:26 2019

@author: neelkanth mehta
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from stocktrends import Renko
from alpha_vantage.timeseries import TimeSeries


'''Defining various technical indicators and performance metrices implemented in this backtest'''
def MACD(Df, a, b, c):
    '''
    Computes moving average convergence-diversion
    
    Parameters
    ==========
    Df: pd.DataFrame():
    a: int: fast moving average
    b: int: slow moving average
    c: int: signal moving average
    
    Returns
    =======
    MACD: pd.Series()
    Signal: pd.Series()
    '''
    df = Df.copy()
    df['MAf'] = df['Adj Close'].ewm(span=a, min_periods=a).mean()
    df['MAs'] = df['Adj Close'].ewm(span=b, min_periods=b).mean()
    df['MACD']= df['MAf'] - df['MAs']
    df['Sig'] = df['MACD'].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    return df['MACD'], df['Sig']


def ATR(Df, n):
    '''
    Computes average true range of a ticker
    
    Parameter
    =========
    Df: pandas.DataFrame(): ohlcv dataframe
    n: int: number of periods over which average true range is to be computed
    
    Returns
    =======
    ATR: pandas.DataFrame(): 
    '''
    df = Df.copy()
    df['HML'] = abs(df['High'] - df['Low'])
    df['HMPC']= abs(df['High'] - df['Adj Close'].shift(1))
    df['LMPC']= abs(df['Low']  - df['Adj Close'].shift(1))
    df['TR']  = df[['HML', 'HMPC', 'LMPC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    return df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'TR', 'ATR']]


def slope(Ser, n):
    '''
    Computes slope of n consecutive points on a plot
    
    Parameters
    ==========
    Ser: pandas.Series(): enter the price series which you intend to find slope for
    n: int: number of consecutive points
    
    Returns
    =======
    '''
    ser = Ser.copy()
    slopes = [i*0 for i in range(n-1)]
    for i in range(n, len(Ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/ (y.max() - y.min())
        x_scaled = (x - x.min())/ (x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)


def renko(Df):
    '''
    Converts ohlc data to Renko bricks
    
    Parameters
    ==========
    Df: pandas.DataFrame(): enter a regular OHLCv dataframe with pandas.Date index
    
    Returns
    =======
    Renko_df: pandas.DataFrame(): Renko bric indicators
    '''
    df = Df.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0, 1, 2, 3, 4, 5]]
    df.rename(columns = {'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Adj Close':'close', 'Volume':'volume'}, inplace=True)
    df2 = Renko(df)
    df2.brick_size = max(0.5, round(ATR(Df, 120)['ATR'][-1],0))
    renko_df = df2.get_ohlc_data()
    renko_df['bar_num'] = np.where(renko_df['uptrend']==True, 1, np.where(renko_df['uptrend']==False,-1, 0))
    for i in range(1, len(renko_df['bar_num'])):
        if renko_df['bar_num'][i] > 0 and renko_df['bar_num'][i-1] > 0:
            renko_df['bar_num'][i] += renko_df['bar_num'][i-1]
        elif renko_df['bar_num'][i]<0 and renko_df['bar_num'][i-1] < 0:
            renko_df['bar_num'][i] += renko_df['bar_num'][i-1]
    renko_df.drop_duplicates(subset='date', keep='last', inplace=True)
    return renko_df


def CAGR(Df):
    '''
    Computes compounded average annual growth rate for a give series
    
    Parameter
    =========
    Df: pandas.DataFrame(): standard dataset
    
    Returns
    =======
    CAGR: float: compounded average annual growth rate
    '''
    df = Df.copy()
    df['CumRet'] = df['ret'].add(1).cumprod()
    n = len(df)/ (252 * 78)
    return (df['CumRet'].tolist()[-1])**(1/n) - 1


def volatility(Df):
    '''
    Computes annualized volatiliy of a given series
    
    Parameters
    ==========
    **Same as that of CAGR**
    
    Returns
    =======
    Annual volatility: float
    '''
    df = Df.copy()
    return df['ret'].std() * np.sqrt(252 * 78)


def sharpe(Df, rf):
    '''
    Computes Sharpe ratio for a given series
    
    Parameters
    ==========
    Df: pandas.DataFrame(): a dataframe containing returns
    rf: float: risk-free rate
    '''
    return (CAGR(Df) - rf)/ volatility(Df)


def max_dd(Df):
    '''
    Computes Max Drawdown of a given series
    
    Parameters
    ==========
    Df: pandas.DataFrame(): a pandas DataFrame containing returns 'ret' 
    
    Returns
    =======
    Max Drawdown: float
        
    '''
    df = Df.copy()
    df['CumRet'] = df['ret'].add(1).cumprod()
    df['CumRollMax'] = df['CumRet'].cummax()
    df['Drawdown'] = df['CumRollMax'] - df['CumRet']
    df['Drawdown_pct'] = df['Drawdown']/ df['CumRollMax']
    return df['Drawdown_pct'].max()


'''Downloading historical data for tech stocks'''
tickers = ['MSFT', 'AAPL', 'FB', 'AMZN', 'INTC', 'CSCO', 'VZ', 'IBM', 'QCOM', 'LYFT']

ohlcvintraday = {}
key_path = "C:\\Users\\neelkanth mehta\\Documents\\Algo_Trading_Strategies\\key.txt"
ts = TimeSeries(key=open(key_path, 'r').read(), output_format='pandas')

attempt = 0
drop = {}
while len(tickers) > 0 and attempt <= 5:
    tickers = [j for j in tickers if j not in drop]
    for i in range(len(tickers)):
        try:
            ohlcvintraday[tickers[i]] = ts.get_intraday(symbol=tickers[i], interval='5min', outputsize='full')[0]
            ohlcvintraday[tickers[i]].columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
            drop.append(tickers[i])
        except:
            print('%s failed to fetch data....retrying' % tickers[i])
            continue
    attempt += 1

tickers = ohlcvintraday.keys() # redefine tickers variable after removing any NaN tickers


'''Backtesting'''
# Merging Renko Df with original OHLCV 
ohlcvrenko = {}
ohlcv_dict = copy.deepcopy(ohlcvintraday)
ticker_signal = {}
ticker_return = {}
for ticker in tickers:
    print('Merging for %s' % ticker)
    renkodf = renko(ohlcv_dict[ticker])
    ohlcv_dict[ticker]['Date'] = ohlcv_dict[ticker].index
    ohlcvrenko[ticker] = ohlcv_dict[ticker].merge(renkodf.loc[:,['date', 'bar_num']], how='outer', on='date')
    ohlcvrenko[ticker]['bar_num'].fillna(method='ffill', inplace=True)
    ohlcvrenko[ticker]['MACD'] = MACD(ohlcvrenko[ticker], 12, 26, 9)[0]
    ohlcvrenko[ticker]['MACD_sig'] = MACD(ohlcvrenko[ticker], 12, 26, 9)[1]
    ohlcvrenko[ticker]['MACD_slo'] = slope(ohlcvrenko[ticker]['MACD'], 5)
    ohlcvrenko[ticker]['MACD_sig_slo'] = slope(ohlcvrenko[ticker]['MACD_sig'], 5)
    ticker_signal[ticker] = ""
    ticker_return[ticker] = []

# Identifying signals and calculating daily returns
for ticker in tickers:
    print('Calculating daily returns for %s' % ticker)
    for i in range(len(ohlcvrenko[ticker])):
        if ticker_signal[ticker] == "":
            ticker_return[ticker].append(0)
            if i > 0:
                if ohlcvrenko[ticker]['bar_num'][i] >= 2 and ohlcvrenko[ticker]['MACD'][i] > ohlcvrenko[ticker]['MACD_sig'][i] and ohlcvrenko[ticker]['MACD_slo'][i] > ohlcvrenko[ticker]['MACD_sig_slo'][i]:
                    ticker_signal[ticker] = "Buy"
                elif ohlcvrenko[ticker]['bar_num'][i] <= -2 and ohlcvrenko[ticker]['MACD'][i] < ohlcvrenko[ticker]['MACD_sig'][i] and ohlcvrenko[ticker]['MACD_slo'][i] < ohlcvrenko[ticker]['MACD_sig_slo'][i]:
                    ticker_signal[ticker] = "Sell"
            
        elif ticker_signal[ticker] == "Buy":
            ticker_return[ticker].append((ohlcvrenko[ticker]["Adj Close"][i]/ ohlcvrenko[ticker]["Adj Close"][i-1])-1)
            if i > 0:
                if ohlcvrenko[ticker]['bar_num'][i] <= -2 and ohlcvrenko[ticker]['MACD'][i] < ohlcvrenko[ticker]['MACD_sig'][i] and ohlcvrenko[ticker]['MACD_slo'][i] < ohlcvrenko[ticker]['MACD_sig_slo'][i]:
                    ticker_signal[ticker] = "Sell"
                elif ohlcvrenko[ticker]['MACD'][i] < ohlcvrenko[ticker]['MACD_sig'][i] and ohlcvrenko[ticker]['MACD_slo'][i] < ohlcvrenko[ticker]['MACD_sig_slo']:
                    ticker_signal[ticker] = ""
        
        elif ticker_signal[ticker] == "Sell":
            ticker_return[ticker].append((ohlcvrenko[ticker]['Adj Close'][i-1]/ ohlcvrenko[ticker]['Adj Close'][i])-1)
            if i > 0:
                if ohlcvrenko[ticker]["bar_num"][i] >=2 and ohlcvrenko[ticker]["MACD"][i] > ohlcvrenko[ticker]["MACD_sig"][i] and ohlcvrenko[ticker]["MACD_slo"][i] > ohlcvrenko[ticker]["MACD_sig_slo"][i]:
                    ticker_signal[ticker] = "Buy"
                elif ohlcvrenko[ticker]["MACD"][i] > ohlcvrenko[ticker]["MACD_sig"][i] and ohlcvrenko[ticker]["MACD_slope"][i] > ohlcvrenko[ticker]["MACD_sig_slo"][i]:
                    ticker_signal[ticker] = ""
    ohlcvrenko[ticker]['ret'] = np.array(ticker_return[ticker])


'''Evaluating performance'''
# Calculating overall strategy performance
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlcvrenko[ticker]['ret']
strategy_df['ret'] = strategy_df.mean(axis=1)
CAGR(strategy_df)
sharpe(strategy_df, 0.025)
max_dd(strategy_df)

# Plotting the output
strategy_df['ret'].add(1).cumprod().plot();
plt.show()

# Calculating individual stock's KPIs
cagr = {}
sharpe_ratios = {}
max_drawdown = {}
for ticker in tickers:
    print('Calculating KPIs for %s' % ticker)
    cagr[ticker] = CAGR(ohlcvrenko[ticker])
    sharpe_ratios[ticker] = sharpe(ohlcvrenko[ticker], 0.025)
    max_drawdown[ticker] = max_dd(ohlcvrenko[ticker])


KPIdf = pd.DataFrame([cagr, sharpe_ratios, max_drawdown], index=['Return', 'Sharpe', 'Max-DD'])
KPIdf.T

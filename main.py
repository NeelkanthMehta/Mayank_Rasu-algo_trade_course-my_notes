'''
backtesting strategy 2

for installing alphavantage library: conda install alpha_vantage -c hoishing
'''
import copy
import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader as web

begdt = dt.date.today() - dt.timedelta(1825)
enddt = dt.date.today()

tickers = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'BA', 'CVX', 'CSCO', 'KO', 'DIS', 'DOW', 'XOM', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'V', 'WMT', 'WBA']

stock_op = pd.DataFrame()
stock_hp = pd.DataFrame()
stock_lp = pd.DataFrame()
stock_cp = pd.DataFrame()
stock_vol= pd.DataFrame()
drop = []
attempt = 0
while len(tickers) >= 0 and attempt <=5:
    tickers = [j for j in tickers if j not in drop]
    for i in range(0, len(tickers)):
        try:
            temp = web.get_data_yahoo(tickers[i], begdt, enddt)
            temp.dropna(inplace=True)
            stock_op[tickers[i]] = temp['Open']
            stock_hp[tickers[i]] = temp['High']
            stock_lp[tickers[i]] = temp['Low']
            stock_cp[tickers[i]] = temp['Adj Close']
            stock_vol[tickers[i]]= temp['Volume']
            drop.append(tickers[i])
        except:
            print('%s failed to fetch data...retrying' % tickers[i])
            continue
    attempt += 1


def ATR(h, l, c, n):
    '''Computes Average True Range  of a price series'''
    high = h.copy()
    low = l.copy()
    close = c.copy()
    HML = abs(high - low)
    HMPC= abs(high - close.shift(1))
    LMPC= abs(low  - close.shift(1))
    TR  = pd.concat([HML, HMPC, LMPC], axis=1).max(axis=1, skipna=False)
    return TR.rolling(n).mean()


def CAGR(c):
    '''Computes compounded average annual aveage returns of a price series'''
    close = c.copy()
    cumret = close.pct_change().fillna(0).add(1).cumprod()
    n = len(close)/ 252
    return cumret[-1] ** (1/ n)


def volatility(c):
    '''Computes annual volatility of a trading company'''
    close = c.copy()
    return close.pct_change().fillna(0).std() * np.sqrt(252)


def sharpe(c, rf=0):
    '''Computes Sharpe ratio of a given price series'''
    return (CAGR(c) - rf)/ volatility(c)


def max_dd(c):
    '''Computes Max Drawdown on a price series'''
    close = c.copy()
    cumret = close.pct_change().fillna(0).add(1).cumprod()
    cumrolmax = cumret.cummax()
    dd = cumrolmax - cumret
    ddpct = dd/ cumrolmax
    return ddpct.max()




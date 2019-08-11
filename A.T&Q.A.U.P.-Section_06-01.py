# -*- coding: utf-8 -*-
"""
Backtesting: Monthly Portfolio Rebalancing
1. Test the strategy by applying the rules and trading signal criteria on historical data mimicing actual trading conditions.
2. Factors in slippage, trading/ brokerage cost when assessing the performance
3. Be conservative - err on the side of caution
4. Backtesting is of critical importance in assessing the merit in trading system/ strategy
5. Don't deploy strategy in live market unless backtest generates satisfactory results
6. Limitations - since it is based on historic data, it has little predictive power

Created on Wed Aug  7 14:20:25 2019

@author: neelkanth mehta
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import StandardScaler
import investment_performance as ip
import copy

# Defining variables
beg = dt.date.today() - dt.timedelta(days=1825)
end = dt.date.today()
tickers = ['MMM', 'AXP', 'T', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'V', 'WMT', 'DIS']

'''Downloading data from the web'''
# Downloading DJIA constituent stock price data
df = pd.DataFrame()
attempt = 0
drop = []
while len(tickers) >  0 and attempt <= 5:
    tickers = [j for j in tickers if j not in drop]
    for i in range(0, len(tickers)):
        try:
            temp = web.get_data_yahoo(tickers[i], beg, end, interval='m')['Adj Close']
            temp.dropna(inplace=True)
            temp.name = str(tickers[i])
            df[tickers[i]] = temp
            drop.append(tickers[i])
        except:
            print('%s failed to fetch...retrying.' % tickers[i])
            continue
    attempt += 1

tickers = df.columns.tolist()

# Downloading Benchmark price data: DIA ETF
DIA = web.get_data_yahoo('DIA', beg, end, interval='m')['Adj Close']
bmrk = ip.return_metrics(Ser=DIA, rf=0.02)

# Copying the data and generating a monthly return dataframe
ohlc = copy.deepcopy(df)
rets = ohlc.pct_change().fillna(0)

'''
Strategy:-

Our strategy comprises of taking long position in all the constituent DJIA stocks in equal proportion.
We hold the stock for a month and rebalance the same at the beginning of every month,
The performance of this is comapred with the benchmark
'''
def pfolio(Df, m, x):
    '''
    Returns Cumulative Portfolio Returns
    
    Parameters
    ==========
    Df: Dataframe with monthly return info for all stocks
     m: number of stocks in the portfolio
     x: number of underperforming stocks to be removed from the portfolio every month
    
    Returns
    =======
    Monthly Portfolio returns: pandas.DataFrame()
    '''
    # Creating a copy of dataframe
    df = Df.copy()
    portfolio = []
    monthly_ret = [0]
    
    # 
    for i in df.iterrows():
        if len(portfolio) > 0:
            monthly_ret.append(df[portfolio].iloc[i,:].mean())
            bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
            portfolio = [t for t in portfolio if t not in bad_stocks]
        fill = m - len(portfolio)
        new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
        portfolio += new_picks
        print(portfolio)
    return pd.DataFrame(np.array(monthly_ret), columns=['returns'])
    

# -*- coding: utf-8 -*-
"""
Investment Performance Metrics :-
This code is a part of Performance Measurement KPIs section of Algorithmic Trading & Quantitative Analysis using Python course by Mayank Rasu

Created on Wed Aug  7 07:04:07 2019

@author: neelkanth mehta
"""

import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader as web

beg = dt.date.today() - dt.timedelta(1825)
end = dt.date.today()

tickers = ['^GSPC', 'SPY']

df = pd.DataFrame()
attempt = 0
drop = []
while len(tickers) != 0 and attempt <= 5:
    tickers = [j for j in tickers if j not in drop]
    for i in range(0, len(tickers)):
        try:
            temp = web.get_data_yahoo(tickers[i], beg, end)['Adj Close']
            temp.dropna(inplace=True)
            temp.name = str(tickers[i])
            df[tickers[i]] = temp
            drop.append(tickers[i])
        except:
            print('%s failed to fetch...retrying' % tickers[i])
            continue
    attempt += 1


def CAGR(Ser, logscale=False) -> float:
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
    ret = ser.pct_change().fillna(0) if logscale == False else np.log(ser.pct_change().fillna(0).add(1))
    cumret = ret.add(1).cumprod()
    n = len(cumret)/ 252
    CAGR = (cumret[-1])**(1/n) - 1
    return CAGR


def ann_vol(Ser, logscale=False) -> float:
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
    ret = ser.pct_change().fillna(0) if logscale == False else np.log(ser.pct_change().fillna(0).add(1))
    vol = ret.std()
    return vol * np.sqrt(252)


def sharpe(Ser, logscale=False, rf:float = 0)-> float:
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
    cagr = CAGR(Ser=Ser, logscale=logscale)
    vol  = ann_vol(Ser=Ser, logscale=logscale)
    return (cagr - rf)/ vol


def sortino(Ser, logscale=False, rf:float = 0) -> float:
    '''
    Computes Sortino Ratio of a given price series
    
    Parameters
    ==========
    Ser: pandas.Series(): enter price series of a security you wish to compute Sortino
    logscale: bool value: If selected True, will use log-scale returns to compute Sortino ratio
    rf:      float value: should be the risk-free rate against which you want to compute Sortino
    
    Returns
    =======
    Sharpe Ratio: string
    '''
    ser = Ser.copy()
    ret = ser.pct_change().fillna(0) if logscale == False else np.log(ser.pct_change().fillna(0).add(1))
    neg_ann_vol = ret.loc[ret < 0].std() * np.sqrt(252)
    cagr = CAGR(Ser=Ser, logscale=logscale)
    return (cagr - rf)/ neg_ann_vol


def max_dd(Ser, logscale=False) -> float:
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
    ret = ser.pct_change().fillna(0) if logscale == False else np.log(ser.pct_change().fillna(0).add(1))
    cumret = ret.add(1).cumprod()
    cumrollmax = cumret.cummax()
    dd = cumrollmax - cumret
    dd_pct = dd/ cumrollmax
    return dd_pct.max()


def calmar(Ser, logscale=False) -> float:
    '''
    Computes Calmar Ratio for the price series
    
    Parameters
    ==========
    Ser: pandas.Series(): enter price series of a security you wish to compute Calmar Ratio
    logscale: bool value: If selected True, will use log-scale returns to compute Calmar Ratio
    
    Returns
    =======
    Calmar Ratio: float
    '''
    cagr = CAGR(Ser=Ser, logscale=logscale)
    mdd = max_dd(Ser=Ser, logscale=logscale)
    return cagr/ mdd


def return_metrics(Ser, logscale=False, rf: float = 0.0):
    '''
    Generates a performance metrics table for the selected price series
    
    Parameters
    ==========
    Ser: pandas.Series(): enter price series of a security you wish to compute Return Metrics
    logscale: bool value: If selected True, will use log-scale returns to compute Return Metrics
    rf:      float value: should be the risk-free rate against which you want to compute Sharpe & Sortino ratios
    
    Returns
    =======
    Returns Metrics: pandas.DataFrame()
    '''
    results = pd.Series(
            {'CAGR': CAGR(Ser=Ser, logscale=logscale), 
             'Annual Volatility': ann_vol(Ser=Ser, logscale=logscale), 
             'Sharpe Ratio': sharpe(Ser=Ser, logscale=logscale, rf=rf), 
             'Sortino Ratio': sortino(Ser=Ser, logscale=logscale, rf=rf), 
             'Max Drawdown': max_dd(Ser=Ser, logscale=logscale), 
             'Calmar Ratio': calmar(Ser=Ser)}
            )
    return results.T

mets = return_metrics(Ser=df['SPY'], logscale=True, rf=0.02)
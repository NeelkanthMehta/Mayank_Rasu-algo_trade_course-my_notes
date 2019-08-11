# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:22:12 2019

@author: neelkanth mehta
"""
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import StandardScaler

# Defining variables
start_dt = dt.date.today() - dt.timedelta(days=365)
end_dt = dt.date.today()
tickers = ['AMZN', 'AAPL', 'MSFT', 'GOOG', 'IQ', 'RAD', 'ADPT', 'CETX']

# Downloading data
df = pd.DataFrame()
attempt = 0
drop = []
while len(tickers) != 0 and attempt <= 5:
    tickers = [j for j in tickers if j not in drop]
    for i in range(0, len(tickers)):
        try:
            temp = web.get_data_yahoo(tickers[i], start_dt, end_dt)['Adj Close']
            temp.dropna(inplace=True)
            temp.name = str(tickers[i])
            df[tickers[i]] = temp
            drop.append(tickers[i])
        except:
            print("%s failed to fetch...retrying." % tickers[i])
            continue
    attempt += 1


# Standardizing the dataframe
normalize = StandardScaler()
stddf = normalize.fit_transform(X=df)
stddf = pd.DataFrame(data=stddf, index=df.index.date, columns=df.columns.tolist())



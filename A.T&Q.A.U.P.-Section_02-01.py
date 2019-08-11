# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:18:49 2019

@author: neelkanth mehta
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

#url = 'https://finance.yahoo.com/quote/MSFT/balance-sheet?p=MSFT'
#page = requests.get(url)
#
#page_content = page.content
#
#soup = BeautifulSoup(page_content, 'html.parser')
#
#tabl = soup.find_all('table', {'class': 'Lh(1.7) W(100%) M(0)'})
#
#for t in tabl:
#    rows = t.find_all('tr')
#    for row in rows:
#        print(row.get_text())

tickers = ['AAPL', 'MSFT']
financial_dir = {}

for ticker in tickers:
    temp_dir = {}
    # Getting balance-sheet data from yahoo finance for given ticker
    url  = 'https://finance.yahoo.com/quote/'+ticker+'/balance-sheet?p='+ticker
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    table= soup.find_all('table', {'class': 'Lh(1.7) W(100%) M(0)'})
    for t in table:
        rows = t.find_all('tr')
        for row in rows:
            if len(row.get_text(separator='|').split('|')[0:2])>1:
                temp_dir[row.get_text(separator='|').split('|')[0]] = row.get_text(separator='|').split('|')[1]
    
    # Getting income statement data from yahoo finance for the given ticker
    url = 'https://finance.yahoo.com/quote/'+ticker+'/financials?p='+ticker
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    table= soup.find_all('table', {'class': 'Lh(1.7) W(100%) M(0)'})
    for t in table:
        rows = t.find_all('tr')
        for row in rows:
            if len(row.get_text(separator='|').split('|')[0:2])>1:
                temp_dir[row.get_text(separator='|').split('|')[0]] = row.get_text(separator='|').split('|')[1]
    
    # Getting cash flow data from yahoo finance for the given ticker
    url = 'https://finance.yahoo.com/quote/'+ticker+'/cash-flow?p='+ticker
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    table= soup.find_all('table', {'class': 'Lh(1.7) W(100%) M(0)'})
    for t in table:
        rows = t.find_all('tr')
        for row in rows:
            if len(row.get_text(separator='|').split('|')[0:2])>1:
                temp_dir[row.get_text(separator='|').split('|')[0]] = row.get_text(separator='|').split('|')[1]
    
    # Getting key statistics data from yahoo finance for the given ticker
    url = 'https://finance.yahoo.com/quote/'+ticker+'/key-statistics?p='+ticker
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    table= soup.find_all('table', {'class': 'table-qsp-stats Mt(10px)'})
    for t in table:
        rows = t.find_all('tr')
        for row in rows:
            if len(row.get_text(separator='|').split('|')[0:2])>1:
                temp_dir[row.get_text(separator='|').split('|')[0]] = row.get_text(separator='|').split('|')[-1]

    # Combining all extracted information with the corrosponding ticker
    financial_dir[ticker] = temp_dir

# storing information in pandas dataframe
combined_financials = pd.DataFrame(financial_dir)
combined_financials.dropna(axis=1, inplace=True)
tickers = combined_financials.columns
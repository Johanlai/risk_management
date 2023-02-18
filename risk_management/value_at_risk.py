# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 08:36:02 2023

@author: JHL
"""
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import scipy.stats as stats

def portfolio_return(weights, log_returns, trading_days=None):
    if trading_days != None:
        return (np.sum(weights * log_returns.mean()) * trading_days)
    else: return (np.sum(weights * log_returns,axis=1))
def portfolio_volatility(weights, log_returns, trading_days=None):
    if trading_days != None:
        return (np.sqrt(np.dot(weights.T,np.dot(log_returns.cov() * trading_days, weights))))
    else: return (np.sqrt(np.dot(weights.T,np.dot(log_returns.cov(), weights))))
def column_na(df, threshold=0.8, row_na=True):
    """
    Parameters
    ----------
    df : Any dataframe where some columns have substantial observations missing
    threshold : TYPE, optional
        The default is set to remove columns with less than 80% of total oberservations
    row_na: also removes any rows with missing data for completeness
    
    Returns complete data for analysis
    -------

    """
    names = [x for x in df if df[x].count()<len(df)*threshold]
    if len(set([x[1] for x in names]))>0:
        print(set([x[1] for x in names]))
        print('{} columns were removed because there were less observations than the threshold ({}):'.format(len(set([x[1] for x in names])),threshold))
        print(df[names].count()['Adj Close'])
    else:
        print('No NAs in data')
    if row_na==True:
        return df.dropna(thresh=len(df)*threshold, axis=1).dropna()
    else: 
        return df.dropna(thresh=len(df)*threshold, axis=1)

class Portfolio:
    def __init__(self, tickers=None, start=None, end=None, trading_days = 250, dropnan=True, threshold=0.8):
        """
        Generate a portfolio from a list of tickers.
        .rawdata: {'Adj Close','Close','High','Low','Open','Volume'}
        -------------------
        tickers = []
        {start, end} = datetime
        trading_days = number of trading days in the year
        -------------------
        Defaults:
        Ticker: ^FTSE, Vodafone
        Start: 52 weeks from current date
        End: Current date
        -------------------
        Uses yahoo_finance
        -------------------
        example:
        tickers = tickers.indexes
        start = dt.datetime(2018,1,1)
        end = dt.datetime.today()
        threshold = 0.9
        
        x = var.Portfolio(tickers=tickers, start=start, end=end, threshold=threshold)
        """
# Setting default values to generate quick test instances
    # Use FTSE index if no ticker is provided
        if tickers==None:
            self.tickers = ['^FTSE','^GSPC']
            print ('No ticker provided, FTSE and S&P 500 was used')
        else: self.tickers = tickers
    # If no dates specified, use the range from 52 weeks ago till today
        if start==None:
            start = (dt.datetime.today()-dt.timedelta(weeks=52))
            print ('Default start date: {}'.format((dt.datetime.today()-dt.timedelta(weeks=52)).strftime('%d-%m-%y')))
        if end==None:
            end = (dt.datetime.today())
            print ('Default end date: {}'.format((dt.datetime.today()).strftime('%d-%m-%y')))
# Retieve the data from YahooFinance        
        self.raw_data = yf.download(self.tickers, start=start, end=end)
        if dropnan ==True:
            self.raw_data = column_na(self.raw_data, threshold=threshold)
            self.tickers = set([x[1] for x in self.raw_data.columns])
        self.risk_free_rate = yf.download('^TNX')['Adj Close']
# Quick indication of missing date
        print('The data spans {} working days, but has {} observations.'.format(np.busday_count(start.date(),end.date()),len(self.raw_data)))
        self.log_returns = np.log(self.raw_data['Adj Close'] / self.raw_data['Adj Close'].shift(1))
# Functions for creating portfolio returns and volatilities
    def Efficient_Frontier(self, n=1000, s=10):
        portfolio_returns = []
        portfolio_volatilities = []
        for x in range (n):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            portfolio_returns.append(np.sum(weights * self.log_returns.mean())*250)
            portfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(self.log_returns.cov() * 250, weights))))
        self.portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility':portfolio_volatilities})
        plt.figure(figsize=(10,4))
        plt.scatter(x=self.portfolios['Volatility'],y=self.portfolios['Return'],s=s)
        plt.xlabel("Volatility")
        plt.ylabel("Return")
    def equally_weighted(self):
        self.weights = np.ones(len(self.tickers))/len(self.tickers)
        self.portfolio_prices = portfolio_return(self.weights, self.raw_data['Adj Close'])
        self.portfolio_return = portfolio_return(self.weights, self.log_returns)
        self.portfolio_vol = portfolio_volatility(self.weights, self.log_returns)
    def randomly_weighted(self):
        self.weights = np.random.random(len(tickers))
        self.weights /= np.sum(self.weights)
        self.portfolio_prices = portfolio_return(self.weights, self.raw_data['Adj Close'])
        self.portfolio_return = portfolio_return(self.weights, self.log_returns)
        self.portfolio_vol = portfolio_volatility(self.weights, self.log_returns)
        

    
    

def compute_var(returns, alpha, window=None):
    """
    Input: pandas series of returns
    Output: percentile of return distribution at a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        if window==None:
            return np.percentile(returns, alpha)
        else:
            return returns.rolling(window).apply(lambda x: np.percentile(x,alpha), raw=True).dropna()
            # ALT: returns.rolling(window).quantile(alpha/100)
    else:
        raise TypeError("Expected a pandas data series")

def compute_es(returns, alpha, window=None):
    """
    Input: pandas series of returns
    Output: Expected shortfall for a given confidence interval alpha
    """
    if isinstance(returns, pd.Series):
        if window==None:
            belowVaR = returns <= compute_var(returns, alpha=alpha)
            return returns[belowVaR].mean()
        else:
            return returns.rolling(window).apply(lambda x: x[x<= compute_var(x, alpha)].mean()).dropna()
    else:
        raise TypeError("Expected a pandas data series")
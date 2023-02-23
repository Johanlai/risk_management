# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:21:54 2023

@author: JHL
"""

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
from scipy import stats

def dropNaNs(df, threshold=0.8, row_na=True, drop_extremes=True):
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
    df = df.apply(lambda x: x.replace(0.0,np.nan))
    names = [x for x in df if df[x].count()<len(df)*threshold]
    if len(set([x[1] for x in names]))>0:
        print(set([x[1] for x in names]))
        print('{} columns were removed because there were less observations than the threshold ({}):'.format(len(set([x[1] for x in names])),threshold))
        print(df[names].count()['Adj Close'])
    else:
        print('No NAs in data')
    if row_na==True:
        cleaned = df.dropna(thresh=len(df)*threshold, axis=1).dropna()
    else: 
        cleaned = df.dropna(thresh=len(df)*threshold, axis=1)
    if drop_extremes==True:
        df = cleaned.pct_change().shift(-1).dropna()
        extremes = df[(np.abs(stats.zscore(df)) > 25).any(axis=1)].index
        cleaned = cleaned.drop(index=extremes)
    return cleaned
    
    
class Portfolio:
    def __init__(self, tickers=None, start=None, end=None, dropnan=True, na_threshold=0.8, log_returns=True):
        """
        Generate a portfolio from a list of tickers.
        .rawdata: {'Adj Close','Close','High','Low','Open','Volume'}
        -------------------
        tickers = []
        {start, end} = datetime
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
            self.raw_data = dropNaNs(self.raw_data, threshold=na_threshold)
            self.tickers = set([x[1] for x in self.raw_data.columns])
        self.risk_free_rate = yf.download('^TNX')['Adj Close']
# Quick indication of missing date
        print('The data spans {} working days, but has {} observations.'.format(np.busday_count(start.date(),end.date()),len(self.raw_data)))
        self.returns = self.raw_data['Adj Close'].pct_change().dropna()
        self.log_returns = np.log(self.raw_data['Adj Close']/self.raw_data['Adj Close'].shift(1)).dropna()
        if log_returns==True:
            self.covMatrix = self.log_returns.cov()
        else:
            self.covMatrix = self.returns.cov()
            
def portfolioPerformance(weights, meanReturns, covMatrix, T=252):
    """
    Parameters
    ----------
    weights : list of floats
        DESCRIPTION. Weight of each security in the portfolio.
    meanReturns : series of floats
        DESCRIPTION. Expected return of each security in the portfolio.
    covMatrix : df?
        DESCRIPTION. Covariance matrix of the portfolio securities.
    T : TYPE, int
        DESCRIPTION. The default is 252 trading days in a year.

    Returns
    -------
    port_returns : float
        DESCRIPTION. The portfolio return
        .. math:: \mu = w^{T} \cdot e .
    port_stdev : float
        DESCRIPTION. The portfolio standard deviation.
        .. math:: \sigma = \sqrt{w^{T}Vw} .
    """
    port_returns = np.sum(meanReturns*weights)*T
    port_stdev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))*np.sqrt(T)
    return port_returns, port_stdev

##### Equally weighted
def equallyWeighted(meanReturns, covMatrix):
    weights = np.ones(len(meanReturns))/len(meanReturns)
    return portfolioPerformance(weights, meanReturns, covMatrix)[0], [meanReturns.index.values.tolist(),weights]

##### Sharpe ratio
def negativeSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate = 0):
    portRetuns, portStdev = portfolioPerformance(weights, meanReturns, covMatrix)
    return - (portRetuns - riskFreeRate)/portStdev

def maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=0, contraintSet=(0,1)):
    """
    Parameters
    ----------
    meanReturns : TYPE
        DESCRIPTION.
    covMatrix : TYPE
        DESCRIPTION.
    riskFreeRate : TYPE, optional
        DESCRIPTION. The default is 0.
    contraintSet : TYPE, optional
        DESCRIPTION. The default is (0,1).

    Returns
    -------
    Results : tuple
        DESCRIPTION. NEGATIVE of max sharpe ratio and lists of ALPHABETICALLY ordered tickers with associated weights
    """
    np.set_printoptions(suppress=True)
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) -1})
    bound = contraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(negativeSharpeRatio, numAssets*[1./numAssets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return -result['fun'], [meanReturns.index.values.tolist(), result['x']]

##### Variance

def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    """
    Minimize the portfolio variance by altering the 
     weights/allocation of assets in the portfolio

    Parameters
    ----------
    meanReturns : TYPE, Series of floats
        DESCRIPTION. expected returns for each security
    covMatrix : TYPE, Pandas df
        DESCRIPTION. Covarriance matrix generated by df.cov()
    constraintSet : TYPE, tuple
        DESCRIPTION. Bounded limits of weights. The default is (0,1).

    Returns
    -------
    TYPE
        DESCRIPTION.
    list
        DESCRIPTION. Tickers (alphabetical order) and respective weights.

    """
     
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result['fun'], [meanReturns.index.values.tolist(), result['x']]

###### Efficient frontier
def portfolioReturn(weights, meanReturns, covMatrix):
        return portfolioPerformance(weights, meanReturns, covMatrix)[0]
def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    """For each returnTarget, we want to optimise the portfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return effOpt


##### Results
def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSharpeRatio_Portfolio = maxSharpeRatio(meanReturns, covMatrix)
    maxSharpeRatio_returns, maxSharpeRatio_std = portfolioPerformance(maxSharpeRatio_Portfolio[1][1], meanReturns, covMatrix)
    maxSharpeRatio_returns, maxSharpeRatio_std = round(maxSharpeRatio_returns,3), round(maxSharpeRatio_std,4)
    maxSharpeRatio_allocation = pd.DataFrame(maxSharpeRatio_Portfolio[1][1], index=meanReturns.index, columns=['allocation'])
    maxSharpeRatio_allocation.allocation = [round(i,4) for i in maxSharpeRatio_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio[1][1], meanReturns, covMatrix)
    minVol_returns, minVol_std = round(minVol_returns,4), round(minVol_std,4)
    minVol_allocation = pd.DataFrame(minVol_Portfolio[1][1], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i,4) for i in minVol_allocation.allocation]
    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSharpeRatio_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])
    return maxSharpeRatio_returns, maxSharpeRatio_std, maxSharpeRatio_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns

##### EF visualisation
def EF_graph(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Return a graph ploting the min vol, max sr and efficient frontier"""
    maxSharpeRatio_returns, maxSharpeRatio_std, maxSharpeRatio_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)
    #Max SR
    MaxSharpeRatio = go.Scatter(
        name='Maximium Sharpe Ratio',
        mode='markers',
        x=[maxSharpeRatio_std*100],
        y=[maxSharpeRatio_returns*100],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))
    )
    #Min Vol
    MinVol = go.Scatter(
        name='Mininium Volatility',
        mode='markers',
        x=[minVol_std*100],
        y=[minVol_returns*100],
        marker=dict(color='green',size=14,line=dict(width=3, color='black'))
    )
    #Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100, 2) for ef_std in efficientList],
        y=[round(target*100, 2) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )
    data = [MaxSharpeRatio, MinVol, EF_curve]
    layout = go.Layout(
        title = 'Portfolio Optimisation with the Efficient Frontier',
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Volatility (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)
    
    fig = go.Figure(data=data, layout=layout)
    return fig.show()
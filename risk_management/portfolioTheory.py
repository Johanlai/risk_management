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

class Portfolio:
    def __init__(self, tickers=None, start=None, end=None):
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
        
        x = var.Portfolio(tickers=tickers, start=start, end=end)
        """
# Setting default values to generate quick test instances
    # Use FTSE index if no ticker is provided
        if tickers==None:
            tickers = ['^FTSE','^GSPC']
            print ('No ticker provided, FTSE and S&P 500 was used')
    # If no dates specified, use the range from 52 weeks ago till today
        if start==None:
            start = (dt.datetime.today()-dt.timedelta(weeks=52))
            print ('Default start date: {}'.format((dt.datetime.today()-dt.timedelta(weeks=52)).strftime('%d-%m-%y')))
        if end==None:
            end = (dt.datetime.today())
            print ('Default end date: {}'.format((dt.datetime.today()).strftime('%d-%m-%y')))
        self.tickers = tickers
        self.start = start
        self.end = end
# Retieve the data from YahooFinance 
    def getData(self):
        self.raw_data = yf.download(self.tickers, self.start, self.end)
        self.tickers = self.raw_data['Adj Close'].columns.values
# Clean data for false extremes and NAs
    def cleanData(self, threshold=0.8, drop_extremes=True, excess=5, dateRange=None):
        df = self.raw_data.apply(lambda x: x.replace(0.0,np.nan))
        names = [x for x in df if df[x].count()<len(df)*threshold]
        if len(df[names].count()['Adj Close'].keys())>0:
            print('{} columns were removed because there were less observations than the threshold ({}):'.format(len(df[names].count()['Adj Close'].keys()),threshold))
            print(((df[names].count()['Adj Close'])/len(df)).map('{:.2%}'.format))
        else:
            print('No NAs found')
        self.raw_data = df.dropna(axis=1,thresh=len(df)*threshold)
        if drop_extremes==True:
            df = self.raw_data['Adj Close'].pct_change().shift(-1).dropna()
            extremes = df[df>excess].dropna(how='all').index
            self.raw_data = self.raw_data.drop(index=extremes)
        if dateRange:
            self.raw_data = self.raw_data[(dateRange[0]<self.raw_data.index)&(self.raw_data.index<dateRange[1])]
        self.tickers = self.raw_data['Adj Close'].columns.values
    def calculate_stats(self, logReturns=True):
        self.tickers = self.raw_data['Adj Close'].columns.values
        self.returns = self.raw_data['Adj Close'].pct_change().dropna()
        self.logReturns = np.log(self.raw_data['Adj Close']/self.raw_data['Adj Close'].shift(1)).dropna()
        if logReturns==True:
            self.covMatrix = self.logReturns.cov()
        else:
            self.covMatrix = self.returns.cov()
    def calculate_PortPerformance(self, weights, T=252):
        self.port_return_annual = np.sum(self.logReturns.mean()*weights)*T
        self.port_stdev = np.sqrt(weights.T @ (self.covMatrix @ weights))*np.sqrt(T)
        self.portReturns = self.logReturns @ weights.T

            
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
    np.set_printoptions(suppress=True)
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
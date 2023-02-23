# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:21:54 2023

@author: JHL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go

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
    result : blob
        DESCRIPTION. Returns the NEGATIVE of max sharpe ratio and ALPHABETICALLY ordered associated weights
        THIS IS UPDATED
    """
    np.set_printoptions(suppress=True)
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) -1})
    bound = contraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(negativeSharpeRatio, numAssets*[1./numAssets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return -result['fun'], [meanReturns.index.values, result['x']]

##### Variance

def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    """Minimize the portfolio variance by altering the 
     weights/allocation of assets in the portfolio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result['fun'], [meanReturns.index.values, result['x']]

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
    maxSharpeRatio_returns, maxSharpeRatio_std = round(maxSharpeRatio_returns*100,2), round(maxSharpeRatio_std*100,2)
    maxSharpeRatio_allocation = pd.DataFrame(maxSharpeRatio_Portfolio[1][1], index=meanReturns.index, columns=['allocation'])
    maxSharpeRatio_allocation.allocation = [round(i*100,0) for i in maxSharpeRatio_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio[1][1], meanReturns, covMatrix)
    minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    minVol_allocation = pd.DataFrame(minVol_Portfolio[1][1], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]
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
        x=[maxSharpeRatio_std],
        y=[maxSharpeRatio_returns],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))
    )
    #Min Vol
    MinVol = go.Scatter(
        name='Mininium Volatility',
        mode='markers',
        x=[minVol_std],
        y=[minVol_returns],
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
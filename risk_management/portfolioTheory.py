# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:21:54 2023

@author: JHL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def maxShapreRatio(meanReturns, covMatrix, riskFreeRate=0, contraintSet=(0,1)):
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
    print(result)
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
    print(result)
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
    effOpt = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return effOpt


##### Results
def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]
    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList

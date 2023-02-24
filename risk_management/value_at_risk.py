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
    
def breaches(returns, est_risk):
    df = pd.concat([returns,est_risk],axis=1).dropna()
    df = df[df[0]<df[1]]
    return df[1].rename('breaches')

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
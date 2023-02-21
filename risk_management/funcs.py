# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:15:12 2023

@author: JHL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def mc_sim(data, T=500, sims_count = 100, initial=None, returns_data=False, show_plot=True):
    if isinstance(data, pd.Series)==True:
        if returns_data==False:
            returns = data.pct_change()
            if initial == None:
                initial = data[-1]
        else:
            returns = data
            if initial == None:
                initial = 100
        mean = returns.mean()
        stdev = returns.std()
        means_arr = np.full(shape=(sims_count,T), fill_value=mean)
        portfolio_sims = np.full(shape=(T,sims_count),fill_value=0.0)
        Volatility = np.random.normal(size=(T, 1))
        drift = means_arr - (0.5*stdev**2)
        Z = Z = norm.ppf(np.random.rand(T, sims_count))
        daily_returns = np.exp(drift.T + stdev * Z)
        sims = np.full(shape=(sims_count,T),fill_value=0.0)
        for i in range(0,sims_count):
            sims[i]= np.cumprod(daily_returns.T[i])*initial
    if isinstance(data, pd.DataFrame)==True:
        weights = np.ones(len(data.T))/len(data.T)
        if returns_data==False:
            returns = data.pct_change()
            if initial== None:
                initial = np.dot(data.iloc[-1],weights)
        else:
            returns = data
            if initial == None:
                initial = 100
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        means_arr = np.full(shape=(T,len(data.T)), fill_value=meanReturns)
        sims = np.full(shape=(T,sims_count),fill_value=0.0)
        sims = np.full(shape=(T,sims_count),fill_value=0.0)
        for m in range(0, sims_count):
            Z = np.random.normal(size=(T, len(data.T)))
            L = np.linalg.cholesky(covMatrix)
            dailyReturns = means_arr.T + np.inner(L, Z)
            sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initial
        sims = sims.T
    print('Initial set to {}'.format(initial))
    if show_plot==True:
        plt.plot(sims.T)
        plt.show()
    return sims
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 23:03:53 2023

@author: JHL
"""

from risk_management import value_at_risk as var
from risk_management import yftickers
from risk_management import funcs
from risk_management import portfolioTheory as pt
import timeit
import datetime as dt
import matplotlib.pyplot as plt
from dateutil import relativedelta as timerd
import pandas as pd
from scipy import stats
import numpy as np

tickers = yftickers.simple
start=dt.datetime(2004,1,1)
end=dt.datetime(2023,1,1)
threshold=0.9

port1 = var.Portfolio(tickers=tickers, start=start, end=end, threshold=threshold)

port1.equally_weighted()

weights = port1.weights
meanReturns = port1.log_returns.mean()
covMatrix = port1.log_returns.cov()

pt.portfolioPerformance(weights, meanReturns, covMatrix, T=252)

pt.efficientOpt(meanReturns,covMatrix,returnTarget=0.01)
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:15:12 2023

@author: JHL
"""

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
    print('{} columns were removed because there were less observations than the threshold:'.format(len(df[names].count().T)))
    print(df[names].count())
    if row_na==True:
        return df.dropna(thresh=len(df)*threshold, axis=1).dropna()
    else: 
        return df.dropna(thresh=len(df)*threshold, axis=1)
    
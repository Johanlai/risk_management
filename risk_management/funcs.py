# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:15:12 2023

@author: JHL
"""
tickers = {
    'ftse100':['HWDN.L', 'III', 'SPX.L', 'ICP.L', 'ITV.L', 'HLMA.L', 
               'TW.L', 'ITRK.L', 'EXPN.L', 'RS1.L', 'BDEV.L', 'CRDA.L', 
               'JMAT.L', 'AUTO', 'STJ.L', 'RMV.L', 'HL.L', 'SN.L', 
               'HIK.L', 'AVV.L', 'JD.L', 'BKG.L', 'WPP', 'BNZL.L', 
               'SGE.L', 'ABDN.L', 'SMIN', 'LGEN.L', 'SDR.L', 'AHT', 
               'REL.L', 'SMT.L', 'SGRO.L', 'LAND', 'SKG.L', 'BRBY.L', 
               'LSEG.L', 'NXT.L', 'AV.L', 'HSBA.L', 'DGE.L', 'BME', 
               'UTG', 'AAL', 'PHNX.L', 'BARC.L', 'BT-A.L', 'ABF.L', 
               'LIN', 'MNDI.L', 'NWG', 'CCH.L', 'RTO.L', 'KGF.L', 
               'PSON.L', 'CRH', 'TSCO', 'SVT', 'INVR.L', 'BLND.L', 
               'STAN.L', 'SMDS.L', 'AON', 'RR.L', 'INF.L', 'UU.L', 
               'IMB.L', 'VOD', 'SBRY.L', 'RIO', 'FLTR', 'EDV', 'MRO', 
               'ULVR.L', 'DCC.L', 'CPG', 'ANTO.L', 'ADM', 'MGGT.L', 
               'LLOY.L', 'PRU', 'ENT.L', 'BATS.L', 'RKT.L', 'CNA', 
               'FRES.L', 'IHG', 'WTB.L', 'SHEL', 'AZN', 'BP.L', 
               'GLEN.L', 'GSK', 'NG.L', 'IAG', 'SSE.L', 'BA.L', 
               'OCDO.L', 'PNR', 'DPH.L', 'RMG.L']
    }

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
    
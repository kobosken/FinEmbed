# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:22:42 2019

@author: kenne
"""
#import csv
import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def clean_df(m):
    """
    Take the master dataframe with all of the information, and extract a
    sub-dataframe which consists of columns m and m+1. Then drop NaN values.
    Label rows with the dates (column m), and convert string values to
    numeric.
    """
    a = pd.to_numeric(df.iloc[:,[m,m+1]].set_index('Date').dropna().iloc[:,0])
    return a

def df_merge(lodf):
    """
    Take a list of data frames (lodf), and merge all of them into one 
    single dataframe. This new dataframe will contain rows that are labelled
    by dates that appear in every dataframe in the lodf.
    The column labels of this new dataframe will be names of the original
    dataframes in lodf.
    Note: this also works for a list of Series.
    Haven't tested a mix of series and dataframes...
    """
    n = len(lodf)
    i=2
    labels = [lodf[0].name, lodf[1].name]
    newdf = pd.merge(lodf[0], lodf[1], left_index=True, right_index=True)
    while i < n:
        newdf = pd.merge(newdf, lodf[i], left_index=True, right_index=True)
        labels.append(lodf[i].name)
        i+=1
    newdf.columns = labels
    return newdf


#Import the data into a DataFrame
df = pd.read_csv(r'C:\Users\kenne\Documents\UofT - MSc Mathematics\MAT4000 - Summer Research\Bloomberg Data.csv')
df.columns = df.iloc[0]
df = df.iloc[1:,:]

#Put each separate index into its own DataFrame.
#Sections are separated based on asset class.
#Set the dates as the row labels
#Drop all NaN values

#CASH
BXIIU3MC = pd.to_numeric(df.iloc[:,[0,1]].set_index('Date').dropna().iloc[:,0])
BXIIU3MC.name = 'BXIIU3MC'
list_cash = [BXIIU3MC]
cash = BXIIU3MC
cash.columns = ['BXIIU3MC']
#Since this asset class only has one index, we do it manually.
#We have also set it up so that we can easily include this data frame in
#any other list, if we want to use this data


#EQUITIES
SPX = clean_df(2)
SVX = clean_df(4)
SGX = clean_df(6)
M1USSC = clean_df(8)
MZUSSV = clean_df(10)
MZUSSG = clean_df(12)
S5FINL = clean_df(14)
S5ENRS = clean_df(16)
S5MATR = clean_df(18)
S5INDU = clean_df(20)
S5INFT = clean_df(22)
S5COND = clean_df(24)
S5HLTH = clean_df(26)
S5TELS = clean_df(28)
S5CONS = clean_df(30)
S5RLST = clean_df(32)
S5UTIL = clean_df(34)
#name the dataframes
SPX.name = 'SPX'
SVX.name = 'SVX'
SGX.name = 'SGX'
M1USSC.name = 'M1USSC'
MZUSSV.name = 'MZUSSV'
MZUSSG.name = 'MZUSSG'
S5FINL.name = 'S5FINL'
S5ENRS.name = 'S53NRS'
S5MATR.name = 'S5MATR'
S5INDU.name = 'S5INDU'
S5INFT.name = 'S5INFT'
S5COND.name = 'S5COND'
S5HLTH.name = 'S5HLTH'
S5TELS.name = 'S5TELS'
S5CONS.name = 'S5CONS'
S5RLST.name = 'S5RLST'
S5UTIL.name = 'S5UTIL'
#list of equity dataframes
list_equity = [SPX, SVX, SGX, M1USSC, MZUSSV, MZUSSG, S5FINL, S5ENRS, S5MATR, 
               S5INDU, S5INFT, S5COND, S5HLTH, S5TELS, S5CONS, S5RLST, S5UTIL]
#merge them
equities = df_merge(list_equity)


#FIXED INCOME
LBUSTRUU = clean_df(36)
LUATTRUU = clean_df(38)
LUACTRUU = clean_df(40)
LF98TRUU = clean_df(42)
LGTRTRUU = clean_df(44)
LEGATRUU = clean_df(46)
LECPTREU = clean_df(48)
LBEATREU = clean_df(50)
LACHTRUU = clean_df(52)
LAPCTRJU = clean_df(54)
LP06TREU = clean_df(56)
LP01TREU = clean_df(58)
#name the dataframes
LBUSTRUU.name = 'LBUSTRUU'
LUATTRUU.name = 'LUATTRUU'
LUACTRUU.name = 'LUACTRUU'
LF98TRUU.name = 'LF98TRUU'
LGTRTRUU.name = 'LGTRTRUU'
LEGATRUU.name = 'LEGATRUU'
LECPTREU.name = 'LECPTREU'
LBEATREU.name = 'LBEATREU'
LACHTRUU.name = 'LACHTRUU'
LAPCTRJU.name = 'LAPCTRJU'
LP06TREU.name = 'LP06TREU'
LP01TREU.name = 'LP01TREU'
#list of dataframes
list_fixedinc = [LBUSTRUU, LUATTRUU, LUACTRUU, LF98TRUU, LGTRTRUU, LEGATRUU, 
                 LECPTREU, LBEATREU, LACHTRUU, LAPCTRJU, LP06TREU, LP01TREU]
#merge them
fixed_income = df_merge(list_fixedinc)


#COMMODITIES
CRY = clean_df(60)
BCOM = clean_df(62)
CLN9 = clean_df(64)
NGN19 = clean_df(66)
HGN9 = clean_df(68)
GCQ9 = clean_df(70)
C_N9 = clean_df(72)
COU9 = clean_df(74)
S_N9 = clean_df(76)
LAN19 = clean_df(78)
SIN9 = clean_df(80)
CON9 = clean_df(82)
LXN9 = clean_df(84)
XBN9 = clean_df(86)
SMN9 = clean_df(88)
QSN9 = clean_df(90)
SBN9 = clean_df(92)
BON9 = clean_df(94)
LNN9 = clean_df(96)
W_N9 = clean_df(98)
HON9 = clean_df(100)
KCN9 = clean_df(102)
LCQ9 = clean_df(104)
LHN9 = clean_df(106)
CTN9 = clean_df(108)
LHM9 = clean_df(110)
KWN9 = clean_df(112)
#name the dataframes
CRY.name = 'CRY'
BCOM.name = 'BCOM'
CLN9.name = 'CLN9'
NGN19.name = 'NGN19'
HGN9.name = 'HGN9'
GCQ9.name = 'GCQ9'
C_N9.name = 'C_N9'
COU9.name = 'COU9'
S_N9.name = 'S_N9'
LAN19.name = 'LAN19'
SIN9.name = 'SIN9'
CON9.name = 'CON9'
LXN9.name = 'LXN9'
XBN9.name = 'XBN9'
SMN9.name = 'SMN9'
QSN9.name = 'QSN9'
SBN9.name = 'SBN9'
BON9.name = 'BON9'
LNN9.name = 'LNN9'
W_N9.name = 'W_N9'
HON9.name = 'HON9'
KCN9.name = 'KCN9'
LCQ9.name = 'LCQ9'
LHN9.name = 'LHN9'
CTN9.name = 'CTN9'
LHM9.name = 'LHM9'
KWN9.name = 'KWN9'
#list of dataframes
list_commodity = [CRY, BCOM, CLN9, NGN19, HGN9, GCQ9, C_N9, COU9, S_N9, 
                  LAN19, SIN9, CON9, LXN9, XBN9, SMN9, QSN9, SBN9, BON9, 
                  LNN9, W_N9, HON9, KCN9, LCQ9, LHN9, CTN9, LHM9, KWN9]
#merge them
commodities = df_merge(list_commodity)


#HEDGE FUNDS
HFRXGL = clean_df(114)
HFRXEH = clean_df(116)
HFRXM = clean_df(118)
HFRXED = clean_df(120)
HFRXFIC = clean_df(122)
HFRXDS = clean_df(124)
HFRXAR = clean_df(126)
HFRXRVA = clean_df(128)
#name the dataframes
HFRXGL.name = 'HFRXGL'
HFRXEH.name = 'HFRXEH'
HFRXM.name = 'HFRXM'
HFRXED.name = 'HFRXED'
HFRXFIC.name = 'HFRXFIC'
HFRXDS.name = 'HFRXDS'
HFRXAR.name = 'HFRXAR'
HFRXRVA.name = 'HFRXRVA'
#list of dataframes
list_hedgefund = [HFRXGL, HFRXEH, HFRXM, HFRXED, HFRXFIC, HFRXDS, HFRXAR, 
                  HFRXRVA]
#merge them
hedgefunds = df_merge(list_hedgefund)


#CRYPTOCURRENCIES
BGCI = clean_df(130)
XBTUSD_BGN = clean_df(132)
XETUSD_BGN = clean_df(134)
XRPUSD_BGN = clean_df(136)
XBNUSD_BGN = clean_df(138)
XLCUSD_BGN = clean_df(140)
XEOUSD_BGN = clean_df(142)
XMRUSD_BGN = clean_df(144)
#name the dataframes
BGCI.name = 'BGCI'
XBTUSD_BGN.name = 'XBTUSD_BGN'
XETUSD_BGN.name = 'XETUSD_BGN'
XRPUSD_BGN.name = 'XRPUSD_BGN'
XBNUSD_BGN.name = 'XBNUSD_BGN'
XLCUSD_BGN.name = 'XLCUSD_BGN'
XEOUSD_BGN.name = 'XEOUSD_BGN'
XMRUSD_BGN.name = 'XMRUSD_BGN'
#list of dataframes
list_crypto = [XBTUSD_BGN, XETUSD_BGN, XRPUSD_BGN, XBNUSD_BGN, 
               XLCUSD_BGN, XEOUSD_BGN, XMRUSD_BGN]
#merge them
cryptocurrencies = df_merge(list_crypto)


#COMMENTS

#Idea: What if we drop nan in the original df? This would end up removing
#some non-NaN values since df.dropna() removes entires rows at a time.

#I opted to do df.dropna() on each index manually. I could do this in a loop,
#but I'm not sure how to do it without using up more memory. (I would assign
#a temporary variable to be the list of indicies, taking up more memory). I"m
#not sure how this would affect runtime.

#In the original data set, the first date for each index was, for some unknown
#reason, in the wrong format. I just fixed them directly on the .csv file.

#Is it possible to take a list of dataframes, and turn it into a list of
#strings, which each string being the name of the dataframe? Well, actually
#DataFrames do not have names! We can give them names first!




#PCA
#We want to have a rolling window, and do a PCA at each instance of this
#window. We just have to create a function which runs a PCA at some window,
#and then loop this throughout a given data set.

def rollingPCA(df, weights, pc=1):
    """
    Given a dataframe, window size (ws), and list of weights which is the same
    length as the window size, run a PCA. Output list of pc-tuples of
    eigenvectors for each day
    """
    ws = len(weights)
    newdf = logreturns(df)
    i=0
    eigenvalues = [[] for i in range(pc)]
    while (i+ws) <= len(newdf):
        subdf = newdf.iloc[i:(i+ws),:]
        results = weightedPCA(subdf,weights, pc)
        #if pc==1:
        for j in range(pc):
            eigenvalues[j].append(float(results[1][j]))
        #else:
        #    eigenvalues[j].append(float(results[1][j])) for j in range(pc)
        i+=1
    f = pd.DataFrame(data = eigenvalues).T
    f.plot(figsize=(20,7))
#    plt.plot(f,figsize=(20,7))
#    plt.show()
#    for eivals in eigenvalues:
#        plt.plot(eivals)
#    plt.figure(figsize=(20,7))
#    plt.show()
#    return eigenvalues
    
def weightedPCA(df, w, c):
    """
    Given a data frame (n=windowsize x m=numAssets), rescale the dataframe 
    entries by using the weights (list of n scalars), and then perform regular
    PCA on this new scaled dataframe.
    """
    indx = range(len(w))
    newdf = pd.concat([w[i]*pd.Series.to_frame(df.iloc[i]).T for i in indx])
    results = dfPCA(newdf, c)
    return results
    
def dfPCA(df, n):
    """
    Given a dataframe, perform a PCA.
    Output [df_eigenvectors, eigenvalues].
    """
    pca = PCA(n_components = n)
    principalComponents = pca.fit(df)
    eivals = pca.explained_variance_ratio_
    return [principalComponents, eivals]
    
def logreturns(df):
    """
    Take a dataframe of price values, and output a dataframe of
    the daily returns.
    """
    logdf = df.applymap(math.log)
    differences = pd.DataFrame(data = np.diff(logdf,n=1,axis=0))
#    newdf = np.divide(differences, logdf.drop(logdf.index.values[-1]))
#    newdf.columns = logdf.columns.values
#    newdf.index = logdf.index.values[:-1]
#    return newdf
    differences.columns = logdf.columns.values
    differences.index = logdf.index.values[:-1]
    return differences
    
def expweight(lamda, n):
    """
    Create a list of weights, of length n. Each weight will be given by the
    exponential function with parameter lamda, but rescaled so that they
    sum to 1.
    """
    tempw = []
    total = 0
    i=0
    while i < n:
        w = math.exp(lamda*i)
        tempw.append(w)
        total += w
        i+=1
    weights = [x/total for x in tempw]
    return weights


fi2 = [LUATTRUU, LUACTRUU, LF98TRUU, LGTRTRUU, LBEATREU, LACHTRUU, LAPCTRJU,
       LP06TREU]
plt.bar([x for x in range(len(fi2))], dfPCA(logreturns(df_merge(fi2)), 
                          len(fi2))[1])
#COMMENTS
#Can successfully do a rolling window weighted PCA! Yay!
#Think about: log returns? I think we should actually be doing PCA on the daily
    #log returns
#Think about plotting data to have visuals. Perhaps plot (1st eigenvalue) vs.
    #date? How would I plot the date axis? Well, we could just plot each day
    #with an integer, and then label the first date.
#maybe take 
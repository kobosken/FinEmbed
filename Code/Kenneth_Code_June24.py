# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:30:07 2019

@author: kenne
"""

#IMPORT MODULES
import pandas as pd
import numpy as np
import math
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

class Rolling_Window:
    """
    Take a pandas dataframe and apply a function on each (rolling) window. 
    Depending on the type of Rolling_Window (variable vs fixed), the window 
    size and stride length may or may not change, respectively. Variable means 
    that the number of days per window and the number of days between windows 
    is not constant. Fixed means that each window has a fixed size and the 
    stride length is also fixed.
    """
    def __init__(self, data, function, style = 'fixed'):
        self.data = data
        self.style = style
        self.func = function
        self.windowsize = self.windowsize()
        self.stride = self.stride()
        self.endpoints = self.endpoints()
        self.results = self.applyfunc()
        
    def windowsize(self):
        if self.style == 'fixed':
            ws = int(input('Window size (days) = '))
        elif self.style == 'variable':
            ws = int(input('# of months per window = '))
        else:
            print('The rolling window style is not valid.')
        return ws
    
    def stride(self):
        if self.style == 'fixed':
            sl = int(input('Stride length (days) = '))
        elif self.style == 'variable':
            sl = int(input('# of months between windows = '))
        else:
            print('The rolling window style is not valid.')
        return sl
        
    def endpoints(self):
        i=0
        endpts = [0]
        
        if self.style == 'fixed':
            while i+self.windowsize < len(self.data):
                endpts.append(i+self.windowsize)
                i += self.stride
        elif self.style == 'variable':
            a = self.data.index
            j = 0
            cur_mth = a[0][5:7]
            while j < len(a):
                if a[j][5:7] == cur_mth:
                    j+=1
                else:
                    endpts.append(j)
                    cur_mth = a[j][5:7]
                    j+=1
        else:
            print('The rolling window style is not valid.')
        return endpts
        
    def applyfunc(self):
        i=0
        output = []
        
        if self.style == 'fixed':
            sl = 1
        elif self.style == 'variable':
            sl = self.stride
            
        while i+sl < len(self.endpoints):
            window = self.data.iloc[self.endpoints[i]:self.endpoints[i+sl]]
            results = self.func(window)
            output.append(results)
            i+=1
        return output
        
    
    

#DEFINE CLASS
class Assets:
    """
    Take a list of pd.series (loa), each corresponding to an asset, and obtain
    information about this group of assets.
    """
    def __init__(self, loa):
        self.rawdata = loa
        self.assets = [asset.name for asset in loa] 
        self.data = self.df_merge()
        self.logreturns = self.logreturns()        
        #self.results = self.dfPCA()
        #self.eigenvalues = self.results[0]  #eigenvalues vs. eigenvalue ratios
        #self.eigenvectors = self.results[1]
        #self.imagedata = self.imagedata()
        #self.image = plt.scatter(self.imagedata[0], self.imagedata[1])
        
    #def __eq__(self, obj):
    #    return (self.rawdata == obj.rawdata) and ...
    #SEARCH THIS UP
        
    def df_merge(self):
        """
        Take a list of pd.series (loa), each corresponding to an asset, and 
        merge all of them into one single dataframe. This new dataframe will 
        contain rows that are labelled by dates that appear in every series in 
        the loa. The column labels of this new dataframe will be names of the 
        assets in the loa.
        """
        loa = self.rawdata
        n = len(loa)
        i=2
        newdf = pd.merge(loa[0], loa[1], left_index=True, right_index=True)
        while i < n:
            newdf = pd.merge(newdf, loa[i], left_index=True, right_index=True)
            i+=1
        newdf.columns = self.assets
        return newdf

    def logreturns(self):
        """
        Take a dataframe of price values, and output a dataframe of
        the daily log returns.
        """
        logdf = self.data.applymap(math.log)
        differences = pd.DataFrame(data = np.diff(logdf,n=1,axis=0))
        differences.columns = logdf.columns.values
        differences.index = logdf.index.values[:-1]
        return differences
        
    def dfPCA(self):
        """
        Given a dataframe, perform a PCA.
        Output (list of the first n eigenvalues, array of eigenvectors).
        """
        pca= PCA()
        results = pca.fit(self.logreturns)
        eivals = results.explained_variance_ratio_
        eivecs = results.components_
        return [eivals, eivecs]
    
    def imagedata(self):
        """
        Take the log returns and collapse it into a 2D image, while preserving
        similarities. This uses the SMACOF algorithm.
        """
        embedding = MDS(n_components=2)
        collapse = embedding.fit_transform(self.logreturns.iloc[:90])
        x = [c[0] for c in collapse]
        y = [c[1] for c in collapse]
        return [x,y]
        
    
def dfPCA(df):
        """
        Given a dataframe, perform a PCA.
        Output a list of the eigenvalues.
        """
        pca= PCA()
        results = pca.fit(df)
        eivals = results.explained_variance_ratio_
        return eivals
    

#DEFINE USEFUL FUNCTIONS
def clean_df(m):
    """
    Take the master dataframe with all of the information, and extract a
    sub-dataframe which consists of columns m and m+1. Then drop NaN values.
    Label rows with the dates (column m), and convert string values to
    numeric.
    """
    a = pd.to_numeric(df.iloc[:,[m,m+1]].set_index('Date').dropna().iloc[:,0])
    return a


#def rollingPCA(df, weights, pc=1):
#    """
#    Given a dataframe, window size (ws), and list of weights which is the same
#    length as the window size, run a PCA. Output list of pc-tuples of
#    eigenvectors for each day
#    """
#    ws = len(weights)
#    newdf = logreturns(df)
#    i=0
#    eigenvalues = [[] for i in range(pc)]
#    while (i+ws) <= len(newdf):
#        subdf = newdf.iloc[i:(i+ws),:]
#        results = weightedPCA(subdf,weights, pc)
#        for j in range(pc):
#            eigenvalues[j].append(float(results[1][j]))
#        i+=1
#    f = pd.DataFrame(data = eigenvalues).T
#    f.plot(figsize=(20,7))
    
#def weightedPCA(df, w, c):
#    """
#    Given a data frame (n=windowsize x m=numAssets), rescale the dataframe 
#    entries by using the weights (list of n scalars), and then perform regular
#    PCA on this new scaled dataframe.
#    """
#    indx = range(len(w))
#    newdf = pd.concat([w[i]*pd.Series.to_frame(df.iloc[i]).T for i in indx])
#    results = dfPCA(newdf, c)
#    return results
    
#def expweight(lamda, n):
#    """
#    Create a list of weights, of length n. Each weight will be given by the
#    exponential function with parameter lamda, but rescaled so that they
#    sum to 1.
#  """
#    tempw = []
#    total = 0
#    i=0
#    while i < n:
#        w = math.exp(lamda*i)
#        tempw.append(w)
#        total += w
#        i+=1
#    weights = [x/total for x in tempw]
#    return weights





#Import all of the data into a master DataFrame.
#In the original data set, the first date for each index was, for some unknown
#reason, in the wrong format. I just fixed them directly on the .csv file.
df = pd.read_csv(r'C:\Users\kenne\Documents\UofT - MSc Mathematics\MAT4000 - Summer Research\Bloomberg Data V2.0.csv')
df.columns = df.iloc[0]
df = df.iloc[1:,:]




#Put each separate index into its own DataFrame.
#These DataFrames will be cleaned using the function clean_df(m)
#Sections are separated based on asset class.


#CASH
BXIIU3MC = pd.to_numeric(df.iloc[:,[0,1]].set_index('Date').dropna().iloc[:,0])
BXIIU3MC.name = 'Barclays 3 month USD LIBOR'
list_cash = [BXIIU3MC]
cash = BXIIU3MC
cash.columns = ['Barclays 3 month USD LIBOR']
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
SPX.name = 'S&P 500'
SVX.name = 'S&P 500 Value'
SGX.name = 'S&P 500 Growth'
M1USSC.name = 'MSCI US Small Cap'
MZUSSV.name = 'MSCI US Small Cap Value'
MZUSSG.name = 'MSCI US Small Cap Growth'
S5FINL.name = 'S&P Financials'
S5ENRS.name = 'S&P Energy'
S5MATR.name = 'S&P Materials'
S5INDU.name = 'S&P Industrials'
S5INFT.name = 'S&P Info Tech'
S5COND.name = 'S&P Cons. Discretionary'
S5HLTH.name = 'S&P Health Care'
S5TELS.name = 'S&P Telecom'
S5CONS.name = 'S&P Cons. Staples'
S5RLST.name = 'S&P Real Estate'
S5UTIL.name = 'S&P Utilities'
#list of equity dataframes
list_equity = [SPX, SVX, SGX, M1USSC, MZUSSV, MZUSSG, S5FINL, S5ENRS, S5MATR, 
               S5INDU, S5INFT, S5COND, S5HLTH, S5TELS, S5CONS, S5RLST, S5UTIL]
#merge them
#equities = Assets(list_equity)



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
LBUSTRUU.name = 'U.S. Aggregate'
LUATTRUU.name = 'U.S. Treasury'
LUACTRUU.name = 'Corporate'
LF98TRUU.name = 'U.S. Corporate High Yield'
LGTRTRUU.name = 'Treasuries'
LEGATRUU.name = 'Global Aggregate'
LECPTREU.name = 'Corporates'
LBEATREU.name = 'Euro-Aggregate'
LACHTRUU.name = 'China Aggregate'
LAPCTRJU.name = 'Asian-Pacific Aggregate'
LP06TREU.name = 'Pan-Euro Aggregate'
LP01TREU.name = 'Pan-European High Yield'
#list of dataframes
list_fixedinc = [LBUSTRUU, LUATTRUU, LUACTRUU, LF98TRUU, LGTRTRUU, LEGATRUU, 
                 LECPTREU, LBEATREU, LACHTRUU, LAPCTRJU, LP06TREU, LP01TREU]
#merge them
#fixedincome = Assets(list_fixedinc)



#COMMODITIES
CRY = clean_df(60)
BCOM = clean_df(62)
CL1 = clean_df(64)
NG1 = clean_df(66)
HG1 = clean_df(68)
GC1 = clean_df(70)
C_1 = clean_df(72)
S_1 = clean_df(74)
W_1 = clean_df(76)
SI1 = clean_df(78)
SB1 = clean_df(80)
#name the dataframes
CRY.name = 'TR/CC CRB ER Index'
BCOM.name = 'BBG Commodity'
CL1.name = 'Crude Oil'
NG1.name = 'Natural Gas'
HG1.name = 'Copper'
GC1.name = 'Gold'
C_1.name = 'Corn'
S_1.name = 'Soybean'
W_1.name = 'Wheat'
SI1.name = 'Silver'
SB1.name = 'Raw Sugar'
#list of dataframes
list_commodity = [CRY, BCOM, CL1, NG1, HG1, GC1, C_1, S_1, W_1, SI1, SB1]
#merge them
#commodities = Assets(list_commodity)



#HEDGE FUNDS
HFRXGL = clean_df(82)
HFRXEH = clean_df(84)
HFRXM = clean_df(86)
HFRXED = clean_df(88)
HFRXFIC = clean_df(90)
HFRXDS = clean_df(92)
HFRXAR = clean_df(94)
HFRXRVA = clean_df(96)
#name the dataframes
HFRXGL.name = 'Hedge Fund Research HFRX Globa'
HFRXEH.name = 'Hedge Fund Research HFRX Equit'
HFRXM.name = 'Hedge Fund Research HFRX Macro'
HFRXED.name = 'Hedge Fund Research HFRX Event'
HFRXFIC.name = 'Hedge Fund Research HFRX Fixed'
HFRXDS.name = 'Hedge Fund Research HFRX ED Di'
HFRXAR.name = 'Hedge Fund Research HFRX Absol'
HFRXRVA.name = 'Hedge Fund Research HFRX Relat'
#list of dataframes
list_hedgefund = [HFRXGL, HFRXEH, HFRXM, HFRXED, HFRXFIC, HFRXDS, HFRXAR, 
                  HFRXRVA]
#merge them
#hedgefunds = Assets(list_hedgefund)



#CRYPTOCURRENCIES
BGCI = clean_df(98)
XBTUSD_BGN = clean_df(100)
XETUSD_BGN = clean_df(102)
XRPUSD_BGN = clean_df(104)
XBNUSD_BGN = clean_df(106)
XLCUSD_BGN = clean_df(108)
XEOUSD_BGN = clean_df(110)
XMRUSD_BGN = clean_df(112)
#name the dataframes
BGCI.name = 'Fund'
XBTUSD_BGN.name = 'Bitcoin'
XETUSD_BGN.name = 'Ethereium'
XRPUSD_BGN.name = 'Ripple'
XBNUSD_BGN.name = 'Bitcoin Cash'
XLCUSD_BGN.name = 'Litecoin'
XEOUSD_BGN.name = 'EOS'
XMRUSD_BGN.name = 'Monero'
#list of dataframes
list_crypto = [XBTUSD_BGN, XETUSD_BGN, XRPUSD_BGN, XBNUSD_BGN, 
               XLCUSD_BGN, XEOUSD_BGN, XMRUSD_BGN]
#merge them
#cryptocurrencies = Assets(list_crypto)







#The way to use this code is to create a list of the indices that we are
#interested in, and then create a bar graph of the eigenvalues for this subset
#of indices. An example is shown below. We have taken a subset of the fixed
#income class of assets, and then plotted the eigenvalues for this set as a 
#bar graph.
#In this example we have not done a rolling window, nor a weighted PCA.
#We have simply done a regular PCA using all of the given data, where each
#day's data is weighted equally.

#if __name__ == "__main__":
#    fi2 = [LUATTRUU, LUACTRUU, LF98TRUU, LGTRTRUU, LBEATREU, LACHTRUU, 
#           LAPCTRJU, LP06TREU]
#    plt.bar([x for x in range(len(fi2))], dfPCA(logreturns(df_merge(fi2)), 
#                          len(fi2)))
    
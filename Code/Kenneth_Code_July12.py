# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:05:15 2019

@author: kenne
"""
#The way to use this code is to create a list of the assets that we are
#interested in, and then create a dataframe for this set of assets. Once we
#have this dataframe, we create a rolling window and apply some function
#on each window. These functions are completely customizable.

#IMPORT MODULES
import pandas as pd
import numpy as np
import math
import sys
import statistics
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
#from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
#from scipy.sparse.csgraph import minimum_spanning_tree


#DEFINE CLASSES (asset class and rolling window class)
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
        
    
class Rolling_Window:
    """
    Take a pandas dataframe and apply a function on each (rolling) window. 
    Depending on the type of Rolling_Window (variable vs fixed), the window 
    size and stride length may or may not change, respectively. Variable means 
    that the number of days per window and the number of days between windows 
    is not constant. Fixed means that each window has a fixed size and the 
    stride length is also fixed.
    """
    def __init__(self, data, function, style = 'variable'):
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
            ws = int(input('Window size (months) = '))
        else:
            print('The rolling window style is not valid.')
        return ws
    
    def stride(self):
        if self.style == 'fixed':
            sl = int(input('Stride length (days) = '))
        elif self.style == 'variable':
            sl = int(input('Stride length (months) = '))
        else:
            print('The rolling window style is not valid.')
        return sl
        
    def endpoints(self):
        i=0
        
        if self.style == 'fixed':
            endpts = []
            while i+self.windowsize < len(self.data):
                endpts.append(i)
                i += self.stride
        elif self.style == 'variable':
            endpts = [0]
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
        dataframe = []
        columns = self.data.columns
        rows = []
        
        if self.style == 'fixed':
            ws = self.windowsize
            for j in self.endpoints:
                window = self.data.iloc[j:j+ws]
                results = self.func(window)
                dataframe.append(results[0])
                rows.append(results[1])
        elif self.style == 'variable':
            sl = self.stride
            ws = self.windowsize
            while i+ws < len(self.endpoints):
                window = self.data.iloc[self.endpoints[i]:self.endpoints[i+ws]]
                results = self.func(window)
                dataframe.append(results[0])
                rows.append(results[1])
                i+=sl
        results = pd.DataFrame(dataframe)
        results.columns = columns
        results.index = rows
        return results.T
          
    
#DEFINE SOME BASIC FUNCTIONS (these may be combined for the rolling window fns)
def dfPCA(df):
        """
        Given a dataframe, perform a PCA.
        Output a list of the eigenvalues.
        """
        pca= PCA()
        results = pca.fit(df)
        eivals = results.explained_variance_ratio_
        return eivals

def df_columns(df):
    """
    Take a pandas DataFrame and convert it to a list of lists, where each list
    in the list is one of the columns of the dataframe.
    """
    data = [list(df[df.columns[i]]) for i in range(len(df.columns))]
    return data

#def sim_mtx(x,sim):
#   """
#    Take a list of objects, and construct a similarity matrix. The function
#    'sim' must take two of these objects and produce a scalar.
#    The similarity function should be symmetric. Be careful regarding the
#    diagonal components.
#    """
#    n = len(x)
#    mtx = []
#    for i in range(n):
#        row = []
#        for j in range(n):
#           if j >= i:
#               row.append(sim(x[i],x[j]))
#           else:
#               row.append(mtx[j][i])
#       mtx.append(row)
#   return mtx

def dot(v,w):
    """
    Take two vectors and compute their dot product.
    """
    if len(v) != len(w):
        sys.exit('The vector lengths do not match.')
    sum = 0
    for i in range(len(v)):
        sum += v[i]*w[i]
    return sum

def sigmas1(D):
    """
    Given a (symmetric) euclidean-distance matrix, D, determine a list of
    appropriate sigmas for the gauss function. Choose sigma in the order of the
    mean distance of a point to its k-th nearest neighbour, where k~log(n)+1.
    """
    n = len(D)
    k = math.log(n)+1
    k = round(k)
    kthdist = []
    for row in D:
        b = sorted(row)
        b = b[:k]
        kthdist.append(statistics.mean(b))
    sigma = statistics.mean(kthdist)
    sigmas = []
    for i in range(n):
        sigmas.append(sigma)
    return sigmas

def sigmas2(D):
    """
    Given a (symmetric) euclidean-distance matrix, D, determine a matrix of
    appropriate sigmas for the gauss function. Choose sigma in the order of the
    mean distance of a point to its k-th nearest neighbour, where k~log(n)+1.
    """
    n = len(D)
    k = math.log(n)+1
    k = round(k)
    sigmas = []
    for row in D:
        b = sorted(row)
        b = b[1:k+1]
        sigma = statistics.mean(b)
        sigmas.append(sigma)
    return sigmas

def sigmas3(D):
    """
    Given a (symmetric) euclidean-distance matrix, D, determine a list of
    appropriate sigmas for the gauss function. Choose sigma in the order of the
    mean distance of a point to its k-th nearest neighbour, where k~log(n)+1.
    """
    n = len(D)
    k = math.log(n)+1
    k = round(k)
    sigmas = []
    for row in D:
        b = sorted(row)
        sigma = b[k]
        sigmas.append(sigma)
    return sigmas
    
def euclid(v,w):
    """
    Take two vectors and compute the euclidean distance between them.
    """
    d = np.subtract(v,w)
    return math.sqrt(dot(d,d))

def euclid_mtx(df):
    """
    Given a dataframe, look at each column (asset) as a vector, and compute
    the (euclidean) distance matrix for these vectors. The resulting matrix
    is symmetric.
    """
    data = df_columns(df)
    n = len(data)
    mtx = []
    for i in range(n):
        row = []
        for j in range(n):
            if j >= i:
                row.append(euclid(data[i],data[j]))
            else:
                row.append(mtx[j][i])
        mtx.append(row)
    return mtx
    
def gauss_mtx(M, sig):
    """
    Given a (symmetric) euclidean-distance matrix, apply the gaussian 
    transformation on each element, using sigmas[i]*sigmas[j], where the list
    of sigmas is given by applying the sig function on M.
    """
    sigmas = sig(M)
    if len(M) != len(sigmas):
        sys.exit('len(M) and len(sigmas) are different')
    n = len(M)
    mtx = []
    for i in range(n):
        row = []
        for j in range(n):
            if j >= i:
                num = M[i][j]**2
                denom = 2*sigmas[i]*sigmas[j]
                row.append(math.exp(-num / denom))
            else:
                row.append(mtx[j][i])
        mtx.append(row)
    return mtx
    
def laplacian(W):
    """
    Take a weighted adjacency matrix, W, and determine the unnormalized 
    Laplacian matrix.
    """
    d = []
    for row in W:
        d.append(sum(row))
    D = np.diag(d)
    L = np.subtract(D, W)
    return L

def Lsym(W):
    """
    Take a weighted adjacency matrix, W, and determine the normalized
    Laplacian(symmetric) matrix.
    """
    d = []
    for row in W:
        d.append(sum(row))
    d = [x**(-0.5) for x in d]
    D = np.diag(d)
    L = np.matmul(D,np.matmul(laplacian(W),D))
    return L
    

def Lrw(W):
    """
    Take a weighted adjacency matrix, W, and determine the normalized
    Laplacian(random walk) matrix.
    """
    d = []
    for row in W:
        d.append(sum(row))
    d = [x**(-1) for x in d]
    D = np.diag(d)
    L = np.matmul(D,laplacian(W))
    return L

    
def maxjump(l):
    """
    Take a numpy.ndarray, and find the index of the maximum jump between 
    adjacent elements.
    """
    dif = np.diff(l)
    k = list(dif).index(max(dif))
    return k

def keivecs(k, eivecs):
    """
    Take a np.ndarray of n n-eigenvectors, and an integer k <= n, and return
    the an nxk list of lists, with columns being the first k eigenvectors.
    """
    n = len(eivecs)
    if k >= n:
        sys.exit('k is too large')
    h = []
    for row in eivecs:
        h.append(list(row[:k+1]))
    return h
    
def labeldata(data, labels):
    """
    Take a list of [[xcoord],[ycoords]], and a pd.series of labels, and split 
    the data into separate lists, depending on the labels.
    """
    m = max(labels)
    xcoords = []
    ycoords = []
    for i in range(m+1):
        xcoords.append([])
        ycoords.append([])
    for j in range(len(labels)):
        cluster = labels[j]
        xcoords[cluster].append(data[0][j])
        ycoords[cluster].append(data[1][j])
    return [xcoords, ycoords]

def ImageCollapse(data):
    """
    Take a list of n-vectors and collapse them into coordinates for a 2D image, 
    while preserving similarities. This uses the SMACOF algorithm.
    """
    embedding = MDS(n_components=2)
    collapse = embedding.fit_transform(data)
    x = [c[0] for c in collapse]
    y = [c[1] for c in collapse]
    return [x,y]

def ClusterPlot(data, labels):
    """
    Take a list of n-vectors, and a pd.series of labels, and plot 
    the data on the same plot, using different colors depending on the labels.
    """
    collapse = ImageCollapse(data)
    labelled = labeldata(collapse, labels)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i in range(len(labelled[0])):
        ax1.scatter(labelled[0][i], labelled[1][i])
    return fig
    


#THESE FUNCTIONS ARE USED ON THE ROLLING WINDOWS   
def image(df):
    """
    Take a DataFrame,calculate eigenvalues and create a bar plot.
    Save the figure.
    """
    title = 'Indicies  ' + df.index[0] + ' to ' + df.index[-1]
    eivals = dfPCA(df)
    x = list(range(len(eivals)+1))
    del x[0]
    fig = plt.subplots(1,1,figsize=(7,7))
    plt.xlabel('Eigenvalues')
    plt.ylabel('Explained Variance')
    plt.axis(ymax = 1.0)
    plt.bar(x,eivals)
    plt.title(title)
    filename = title + '.jpg'
    plt.savefig(filename)

def SpectralClustering_L(df):
    """
    Take a pandas dataframe, and perform a spectral clustering using a guassian
    similarity matrix (sigma varying in each window), and the unnormalized
    Laplacian
    """
    #data = df_columns(df)
    E = euclid_mtx(df)
    W = gauss_mtx(E,sigmas2)        #sigmas can be changed
    #W = sim                         #fully connected graph
    L = laplacian(W)
    vals, vecs = np.linalg.eig(L)
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:,idx]
    k = maxjump(vals)
    #k=4
    #vals = vals[:k+1]
    U = keivecs(k, vecs)
    kmeans = KMeans(n_clusters = k+1).fit(U)
    labels = kmeans.labels_
    #fig = ClusterPlot(data, labels)
    window = list(df.index)[0] + ' to ' + list(df.index)[-1]
    #plt.title(window)
    #plt.show()
    #filename = window + '.jpg'
    #plt.savefig(filename)
    return [labels, window]

def SpectralClustering_Lrw(df):
    """
    Take a pandas dataframe, and perform a spectral clustering using a guassian
    similarity matrix (sigma varying in each window), and the unnormalized
    Laplacian
    """
    #data = df_columns(df)
    E = euclid_mtx(df)
    W = gauss_mtx(E,sigmas2)        #sigmas can be changed
    #W = sim                         #fully connected graph
    L = Lrw(W)
    vals, vecs = np.linalg.eig(L)
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:,idx]
    k = maxjump(vals)
    #k=4
    #vals = vals[:k+1]
    U = keivecs(k, vecs)
    kmeans = KMeans(n_clusters = k+1).fit(U)
    labels = kmeans.labels_
    #fig = ClusterPlot(data, labels)
    window = list(df.index)[0] + ' to ' + list(df.index)[-1]
    #plt.title(window)
    #plt.show()
    #filename = window + '.jpg'
    #plt.savefig(filename)
    return [labels, window]
    

def SpectralClustering_Lsym(df):
    """
    Take a pandas dataframe, and perform a spectral clustering using a guassian
    similarity matrix (sigma varying in each window), and the unnormalized
    Laplacian
    """
    #data = df_columns(df)
    E = euclid_mtx(df)
    W = gauss_mtx(E,sigmas2)        #sigmas can be changed
    #W = sim                         #fully connected graph
    L = Lsym(W)
    vals, vecs = np.linalg.eig(L)
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:,idx]
    k = maxjump(vals)
    #k=4
    #vals = vals[:k+1]
    Umtx = keivecs(k, vecs)
    Tmtx = sklearn.preprocessing.normalize(Umtx)
    kmeans = KMeans(n_clusters = k+1).fit(Tmtx)
    labels = kmeans.labels_
    #fig = ClusterPlot(data, labels)
    window = list(df.index)[0] + ' to ' + list(df.index)[-1]
    #plt.title(window)
    #plt.show()
    #filename = window + '.jpg'
    #plt.savefig(filename)
    return [labels, window]

    
    

    
#neigh = NearestNeighbors(n_neighbors=5, metric='euclidean')


# D A T A  I M P O R T + C L E A N

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



#Import all of the data into a master DataFrame.
#In the original data set, the first date for each index was, for some unknown
#reason, in the wrong format. I just fixed them directly on the .csv file.
df = pd.read_csv(r'C:\Users\kenne\Documents\UofT - MSc Mathematics\MAT4000 - Summer Research\Bloomberg Data V3.0.csv')
df.columns = df.iloc[0]
df = df.iloc[1:,:]




#Put each separate index into its own DataFrame.
#These DataFrames will be cleaned using the function clean_df(m)
#Sections are separated based on asset class.


#CASH
BXIIU3MC = pd.to_numeric(df.iloc[:,[0,1]].set_index('Date').dropna().iloc[:,0])
BXIIU3MC.name = 'Barclays 3 month USD LIBOR'
#list_cash = [BXIIU3MC]
#cash = BXIIU3MC
#cash.columns = ['Barclays 3 month USD LIBOR']
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
#list_equity = [SPX, SVX, SGX, M1USSC, MZUSSV, MZUSSG, S5FINL, S5ENRS, S5MATR, 
#               S5INDU, S5INFT, S5COND, S5HLTH, S5TELS, S5CONS, S5RLST, S5UTIL]

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
LUACTRUU.name = 'U.S. Corporate'
LF98TRUU.name = 'U.S. Corporate High Yield'
LGTRTRUU.name = 'Global Treasuries'
LEGATRUU.name = 'Global Aggregate'
LECPTREU.name = 'Euro Corporate'
LBEATREU.name = 'Euro-Aggregate'
LACHTRUU.name = 'China Aggregate'
LAPCTRJU.name = 'Asian-Pacific Aggregate'
LP06TREU.name = 'Pan-Euro Aggregate'
LP01TREU.name = 'Pan-European High Yield'
#list of dataframes
#list_fixedinc = [LBUSTRUU, LUATTRUU, LUACTRUU, LF98TRUU, LGTRTRUU, LEGATRUU, 
#                 LECPTREU, LBEATREU, LACHTRUU, LAPCTRJU, LP06TREU, LP01TREU]

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
#list_commodity = [CRY, BCOM, CL1, NG1, HG1, GC1, C_1, S_1, W_1, SI1, SB1]

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
#list_hedgefund = [HFRXGL, HFRXEH, HFRXM, HFRXED, HFRXFIC, HFRXDS, HFRXAR, 
#                  HFRXRVA]

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
#list_crypto = [XBTUSD_BGN, XETUSD_BGN, XRPUSD_BGN, XBNUSD_BGN, 
#               XLCUSD_BGN, XEOUSD_BGN, XMRUSD_BGN]

    
#list_all = list_cash + list_equity + list_fixedinc + list_commodity + \
#            list_hedgefund + list_crypto




#DATA SETS TO BE CONSIDERED
#NA_Assets = [S5FINL, S5ENRS, S5MATR, S5INDU, S5INFT, S5COND, S5HLTH, S5TELS, 
#             S5CONS, S5RLST, S5UTIL, LUATTRUU, LUACTRUU, LF98TRUU, CL1, NG1, 
#             HG1, GC1, C_1, S_1, W_1, SI1, SB1]
#GL_Assets = [SVX, SGX, MZUSSV, MZUSSG, LBUSTRUU, LBEATREU, LACHTRUU, LAPCTRJU, 
#             LP06TREU, CL1, NG1, HG1, GC1, C_1, S_1, W_1, SI1, SB1]

#NA_Market = Assets(NA_Assets)
#GL_Market = Assets(GL_Assets)

#NA_rollL = Rolling_Window(NA_Market.logreturns, SpectralClustering_L)
#GL_rollL = Rolling_Window(GL_Market.logreturns.iloc[88:], SpectralClustering_L)
#NA_rollLsym = Rolling_Window(NA_Market.logreturns, SpectralClustering_Lsym)
#GL_rollLsym = Rolling_Window(GL_Market.logreturns.iloc[88:], SpectralClustering_Lsym)
#NA_rollLrw = Rolling_Window(NA_Market.logreturns, SpectralClustering_Lrw)
#GL_rollLrw = Rolling_Window(GL_Market.logreturns.iloc[88:], SpectralClustering_Lrw)
#Before 2011-07-5, the data was monthly

#NA_clustersL = NA_rollL.results
#GL_clustersL = GL_rollL.results
#NA_clustersLsym = NA_rollLsym.results
#GL_clustersLsym = GL_rollLsym.results
#NA_clustersLrw = NA_rollLrw.results
#GL_clustersLrw = GL_rollLrw.results


test = [BXIIU3MC, SVX, SGX, MZUSSV, MZUSSG, LBUSTRUU, LBEATREU, LACHTRUU, 
        LAPCTRJU, LP06TREU, CL1, NG1, HG1, GC1, C_1, S_1, W_1, SI1, SB1, 
        HFRXEH, HFRXM, HFRXED, HFRXFIC, HFRXDS, HFRXAR, HFRXRVA]
#Market = Assets(test)
#a = Market.logreturns
#a = a.iloc[86:] #prior to 2012-04-02, the data for hedgefunds is monthly
#c = Rolling_Window(a, SpectralClustering_L)
#s = c.results

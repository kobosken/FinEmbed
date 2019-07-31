3# -*- coding: utf-8 -*-
"""
    Created on Fri Jul 12 16:05:15 2019
    @author: kenne
    """
#The way to use this code is to create a list of the assets that we are
#interested in, and then create a dataframe for this set of assets. Once we
#have this dataframe, we create a rolling window and apply some function
#on each window. These functions are completely customizable.

# Mina edits: this program would generate a feature dataframe containing:
# 1) cluster df 2) cov df 3) corr df 4) euclidean df 5) guassian df
# 6) eigen vals df 7) eigen vecs df for the rolling windows
# they are concatenated into 1 df at the end.
# notes: the feature df is not labeled with the optimized portfolio: 1, 2, or 3

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
    def __init__(self, data, f, f2, style = 'variable'):
        self.data = data
        self.style = style
        self.func = f
        self.func2 = f2
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
        columns = self.data.columns #asset names
        dataframe =[]
        rows = []
        covflat = []
        corrflat = []
        E = []
        W = []
        vals = []
        vecs = []
 
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
                dataframe.append(results[0]) # result[0] returns labels, which is labeled cluster vector per window's name
                rows.append(results[1]) # result[1] returns windows, which is windows name
                covflat.append(results[2].to_numpy().flatten()) #mina
                corrflat.append(results[3].to_numpy().flatten()) #mina
                E.append(np.asarray(results[4]).flatten())
                W.append(np.asarray(results[4]).flatten())
                evals, evecs = self.func2(results[3])
                vals.append(evals)
                vecs.append(evecs.flatten())
                i+=sl
        
        cluster = pd.DataFrame(dataframe)
        covflat = pd.DataFrame(covflat)
        corrflat = pd.DataFrame(corrflat)
        E = pd.DataFrame(E)
        W = pd.DataFrame(W)
        vals = pd.DataFrame(vals)
        vecs = pd.DataFrame(vecs)
        
        cluster.columns = columns # asset names
        cluster.index = covflat.index = corrflat.index = E.index = W.index = vals.index = vecs.index = rows
    
        return [cluster.T, covflat.T, corrflat.T, E.T, W.T, vals.T, vecs.T]

        # a cluster matrix with columns vector of labeled cluster data per window. window name is column name, row index is asset name.
    #return [labels, window, covariance, correlation, E, W, vals, vecs] #mina

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

def SpectralClustering_L(df): #modified by mina
    """
        Take a pandas dataframe (per window), and perform a spectral clustering using a guassian
        similarity matrix (sigma varying in each window), and the unnormalized
        Laplacian
        """
    #data = df_columns(df)
    covariance = df.cov() # mina
    correlation = df.corr(method='pearson') #mina
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
    return  [labels, window, covariance, correlation, E, W] #mina

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

########################### main function starts here #############################

# US ##############################################################################

#Import all of the US data into a master DataFrame.
#In the original data set, the first date for each index was, for some unknown
#reason, in the wrong format. I just fixed them directly on the .csv file.
df = pd.read_csv('/Users/minaxxan/Documents/Risklab/USGlobal.csv')
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


#FIXED INCOME
LBUSTRUU = clean_df(36)
LUATTRUU = clean_df(38)
LUACTRUU = clean_df(40)
LF98TRUU = clean_df(42)
#name the dataframes
LBUSTRUU.name = 'U.S. Aggregate'
LUATTRUU.name = 'U.S. Treasury'
LUACTRUU.name = 'U.S. Corporate'
LF98TRUU.name = 'U.S. Corporate High Yield'


#COMMODITIES
BCOM = clean_df(44)
CO1 = clean_df(46)
CL1 = clean_df(48)
QS1 = clean_df(50)
XB1 = clean_df(52)
HO1 = clean_df(54)
NG1 = clean_df(56)
LA1 = clean_df(58)
HG1 = clean_df(60)
LN1 = clean_df(62)
LX1 = clean_df(64)
GC1 = clean_df(66)
SI1 = clean_df(68)
KC1 = clean_df(70)
C_1 = clean_df(72)
CT1 = clean_df(74)
LH1 = clean_df(76)
LC1 = clean_df(78)
S_1 = clean_df(80)
SM1 = clean_df(82)
SB1 = clean_df(84)
W_1 = clean_df(86)
#name the dataframes
BCOM.name = 'BBG Commodity'
CO1.name = 'Brent Crude Oil'
CL1.name = 'WTI Crude Oil'
QS1.name = 'Gas Oil'
XB1.name = 'Gasoline'
HO1.name = 'Heating Oil'
NG1.name = 'Natural Gas'
LA1.name = 'Aluminum'
HG1.name = 'Copper'
LN1.name = 'Nickel'
LX1.name = 'Zinc'
GC1.name = 'Gold'
SI1.name = 'Silver'
KC1.name = 'Coffee'
C_1.name = 'Corn'
CT1.name = 'Cotton'
LH1.name = 'Lean Hogs'
LC1.name = 'Live Cattle'
S_1.name = 'Soybean'
SM1.name = 'Soybean Meal'
SB1.name = 'Raw Sugar'
W_1.name = 'Wheat'



# Global ##############################################################################

# we only import additional assets that US master df above doesnt have

#EQUITIES
MXEU = clean_df(88)
MXEA = clean_df(90)
MXAP = clean_df(92)
MXEF = clean_df(94)
#name the dataframes
MXEU.name = 'Europe Equities'
MXEA.name = 'Developped Equities'
MXAP.name = 'Asia Pacific Equities'
MXEF.name = 'MSCI Emerging Equities'


#FIXED INCOME
LGTRTRUU = clean_df(96)
LEGATRUU = clean_df(98)
LECPTREU = clean_df(100)
LBEATREU = clean_df(102)
LACHTRUU = clean_df(104)
LAPCTRJU = clean_df(106)
LP06TREU = clean_df(108)
LP01TREU = clean_df(110)
#name the dataframes
LGTRTRUU.name = 'Global Treasuries'
LEGATRUU.name = 'Global Aggregate'
LECPTREU.name = 'Euro Corporate'
LBEATREU.name = 'Euro-Aggregate'
LACHTRUU.name = 'China Aggregate'
LAPCTRJU.name = 'Asian-Pacific Aggregate'
LP06TREU.name = 'Pan-Euro Aggregate'
LP01TREU.name = 'Pan-European High Yield'




#DATA SETS TO BE CONSIDERED
#US_Assets = [BXIIU3MC, SPX, M1USSC,S5FINL, S5ENRS, S5MATR, S5INDU, S5INFT, S5COND, S5HLTH,S5TELS,S5CONS, S5RLST, S5UTIL, LBUSTRUU, LUATTRUU, LUACTRUU, BCOM, CO1, CL1, QS1, XB1, HO1,NG1,LA1,HG1,LN1,LX1, GC1, SI1, KC1,C_1,CT1,LH1,LC1, S_1, SM1, SB1,W_1]
GL_Assets = [BXIIU3MC, SPX,MXEU, MXEA,MXAP,MXEF, LBUSTRUU, LUATTRUU, LUACTRUU,LF98TRUU, LGTRTRUU,LEGATRUU,LECPTREU, LBEATREU, LACHTRUU, LAPCTRJU, LP06TREU,LP01TREU, BCOM, CO1, CL1, QS1, XB1, HO1,NG1,LA1,HG1,LN1,LX1, GC1, SI1, KC1, C_1, CT1,LH1, LC1, S_1, SM1, SB1, W_1]

#US_Market = Assets(US_Assets)
#GL_Market = Assets(GL_Assets)

#US_rollL = Rolling_Window(US_Market.logreturns, SpectralClustering_L)
#GL_rollL = Rolling_Window(GL_Market.logreturns, SpectralClustering_L)


#US_clustersL = US_rollL.results
#GL_clustersL = GL_rollL.results
             
#US_rollLsym = Rolling_Window(US_Market.logreturns, SpectralClustering_Lsym)
#GL_rollLsym = Rolling_Window(GL_Market.logreturns.iloc[88:], SpectralClustering_Lsym)
#US_rollLrw = Rolling_Window(US_Market.logreturns, SpectralClustering_Lrw)
#GL_rollLrw = Rolling_Window(GL_Market.logreturns.iloc[88:], SpectralClustering_Lrw)
#Before 2011-07-5, the data was monthly
             
#US_clustersLsym = US_rollLsym.results
#GL_clustersLsym = GL_rollLsym.results
#US_clustersLrw = US_rollLrw.results
#GL_clustersLrw = GL_rollLrw.results

#USMarket = Assets(US_Assets)
#USlog = USMarket.logreturns
#USrolling = Rolling_Window(USlog, SpectralClustering_L)
#UScluster = USrolling.results


features_name = ['Cluster','Cov', 'Corr', 'Euclidean', 'Gaussian', 'Eigen Vals', 'Eigen Vecs']
GLMarket = Assets(GL_Assets)
GLlog = GLMarket.logreturns
GLrolling = Rolling_Window(GLlog, SpectralClustering_L, np.linalg.eig)
res = GLrolling.results
features = pd.concat(res, keys = features_name)















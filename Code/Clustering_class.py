import pandas as pd
import math
import numpy as np
import sklearn.neighbors as skln
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.linalg import fractional_matrix_power
from numpy import linalg as LA
from sklearn.cluster import KMeans

#import label data
file_path = 'C:/Users/fjaeckle/Documents/FIM/Research_Toronto/Data/label_data.csv'
label_data = pd.read_csv(file_path,sep =';')
label_data_nn = label_data['Strategy Code']
###input data
#Introduction
#1.Run class Spectral Clustering l.113-375
#2.Create object with desired data frame (here: used dataframe is called test_set2) l.18
#3.Create Dictionary for features considered in the NN, Dictionary is transformed into Dataframe l.55-113
#4.Data-Frames for kmeans-labels needs to be created indidividually since k-means labeling is not consistent;
#function relabeling_cluster relabels clusters to make it consistent, but manual adjustments of labels is still necessary


#Object Creation
test = Clustering_manu(test_set2)

#Create Data Frame for Cluster-Assignment seperately because of re-labeling of clusters
nClusters_L = test.nCluster_L
kmeans_L = test.kmeans_Cluster_L
kmeans_Lrw = test.kmeans_Cluster_Lrw
kmeans_Lsym = test.kmeans_Cluster_Lsym

kmeans_L_re = relabeling_cluster(kmeans_L)


def relabeling_cluster(Cluster):
    Cluster_re = []    
    for x in range(len(Cluster)):
        Clu = Cluster[x]
        if Clu[0]==0:
            Clu_re = Clu
            Cluster_re.append(Clu_re)
        elif Clu[0]==1:
            lut = np.array([1,0,2,3])# 0->1; 1->0; 2->2; 3->3
            Clu_re = lut[Clu]
            Cluster_re.append(Clu_re)
        elif Clu[0]==2:
            lut = np.array([1,2,0,3])# 0->1; 1->2; 2->0; 3->3 
            Clu_re = lut[Clu]
            Cluster_re.append(Clu_re)
        elif Clu[0]==3:
            lut = np.array([3,1,2,0])# 0->3; 1->1; 2->2; 3->0
            Clu_re = lut[Clu]
            Cluster_re.append(Clu_re)
    return Cluster_re

# after re-labeling manual adjustments of cluster assignment for single time periods are still necessary
############################
############################
#Attributes we consider in our Neural Networks-Presentation in Dictionary
#Attributes in Matrix-Form
Attributes_mat = {
     'L': test.L,
     'L_rw':test.L_rw,
     'L_sym': test.L_sym,
     'W_e': test.W_e,
     'W_k': test.W_k,
     'Corr': test.correlation,
     'Cov': test.covariance,
     'eigvec_L': test.eigenvector_L,
     'eigvec_Lrw': test.eigenvector_Lrw,
     'eigvec_Lsym': test.eigenvector_Lsym,
     'fully_graph': test.fullyconnected,
     'kneighbor_graph': test.kneighbor,
     'kmeans_L': test.kmeans_Cluster_L,
     'kmeans_Lrw': test.kmeans_Cluster_Lrw,
     'kmeans_Lsym': test.kmeans_Cluster_Lsym,
     'nCluster_L':test.nCluster_L,
     'nCluster_Lrw': test.nCluster_Lrw,
     'nCluster_Lsym': test.nCluster_Lsym}
#Attributes in Vector-Form
Attributes_vec = {
    'eigval_L': test.eigenvalue_L,
    'eigval_Lrw': test.eigenvalue_Lrw,
    'eigval_Lsym': test.eigenvalue_Lsym,
    'nCluster_L':test.nCluster_L,
    'nCluster_Lrw': test.nCluster_Lrw,
    'nCluster_Lsym': test.nCluster_Lsym}
 
#Convert Dictionary (Attributes in Matrix-Form) in Data Frame
df_mat = pd.DataFrame(Attributes_mat)
DF_mat = pd.DataFrame()      
for x in Attributes_mat.keys():
    li = Attributes_mat[x]
    for i in range(len(li)):
        li[i] = li[i].flatten(order = 'C')
    df = pd.DataFrame(li)
    DF_mat = pd.concat([DF_mat, df],axis = 1) 
DF_mat.columns = range(DF_mat.shape[1])   
DF_mat = DF_mat.astype('float64')

#Convert Dictionary (Attributes in Vector-Form) in Data Frame
df_vec = pd.DataFrame(Attributes_vec)  
DF_vec = pd.DataFrame()      
for x in Attributes_vec.keys():
    li = Attributes_vec[x]
    df = pd.DataFrame(li)
    DF_vec = pd.concat([DF_vec, df],axis = 1)
DF_vec.columns = range(DF_vec.shape[1])
DF_vec = DF_vec.astype('float64')

#Merge DF_mat and DF_vec to DF
DF = pd.concat([DF_mat,DF_vec],axis = 1)
DF.columns = range(DF.shape[1])
DF = DF.astype('float64')    

#Delete unnecessary variables
del i
del li



#Class Spectral Clustering
class Clustering_manu():
#Constructor
    def __init__(self,df):
        
        #Data Preparation
        self.dates = df['Date']
        self.values = df.drop(labels = 'Date',axis = 1)
        self.log_returns = self.calc_log_returns()
        self.log_returns_dates = self.dates.drop(len(self.dates)-1,axis = 0)
        #Rolling Window
        self.rolling_window = self.calc_rolling_window()
    
        #Distance matrices as input for gaussian similiarity matrix
        self.kneighbor,self.kneighbor_rmean = self.calc_kneighbor_graph()
        self.fullyconnected,self.minspanningtree,self.epsilon = self.calc_fullyconnected_graph()
        
        #Gaussian similarity matrices as input for Laplacian
        self.W_k,self.W_e = self.calc_similaritymatrix(self.fullyconnected) #Choose graph as parameter
        
        #Input matrices for PCA
        self.L,self.L_sym,self.L_rw = self.calc_laplacian(self.W_k)#Laplacian
        self.covariance, self.correlation = self.calc_cov_corr()#Covariance,Correlation
        #Eigenvalu/Eigenvector of Laplacians
        self.eigenvalue_L, self.eigenvector_L,self.eigenvalue_Lsym,self.eigenvector_Lsym,self.eigenvalue_Lrw,self.eigenvector_Lrw =self.calc_eigenvalue_vector()
        #Number of Clusters considering Laplcians
        self.nCluster_L,self.nCluster_Lsym,self.nCluster_Lrw = self.calc_numclusters()   
        #K-means of Laplacians
        self.kmeans_Cluster_L,self.kmeans_Cluster_Lrw = self.calc_kmeans()
        self.kmeans_Cluster_Lsym = self.calc_kmeans_Lsym()
        
    #Log-Returns        
    def calc_log_returns (self):
        log_returns = pd.DataFrame()
        log_returns = self.values.applymap(math.log)
        log_returns = pd.DataFrame(data = (np.diff(log_returns,axis =0,n=1)))
        return log_returns 
    
    def calc_rolling_window (self):
        dates = self.log_returns_dates
        rolling_window = [0]
        #convert datetime to str
        for i in range(len(dates)):
            dates[i] = dates[i].strftime('%Y-%m-%d')
        #Determine Index Number of each beginning of month
        for i in range(len(dates)-1):         
            if(dates[i][5:7] != dates[i+1][5:7]):
                rolling_window.append(i+1)
                
        return rolling_window      
    
    #distance matrix
    #euclidean distance applied for all points
    def calc_fullyconnected_graph (self):
        tw = 3
        #rw = 1 by default
        data = self.log_returns
        fully_graph = []
        msp_graph = []
        epsilon = []
        for x in range(len(self.rolling_window)-tw):
            #Fully connected graph
            data_tw = data.loc[self.rolling_window[x]:self.rolling_window[tw + x]-1]
            data_trans = data_tw.transpose()
            distMet = skln.DistanceMetric.get_metric('euclidean')
            dist = distMet.pairwise(data_trans)
            fully_graph.append(dist)
            #Calclulation of sigma for gaussian similarity function(look at Paper Spectral Clustering)
            #Minimum spanning tree
            msp = minimum_spanning_tree(dist)
            msp_dist = msp.toarray()
            msp_graph.append(msp_dist)
            #Epsilon
            msp_dist_max = msp_dist.max()#Max Per Row
            msp_dist_max = msp_dist_max.max()#Max Array
            epsilon.append(msp_dist_max)
        return (fully_graph,msp_graph,epsilon)       
    
    #euclidean distance for k-neighbors
    def calc_kneighbor_graph (self):
        tw = 3
        #rw=1
        data = self.log_returns
        kneighbor_graph = []
        kneighbor_rmean = []
        for x in range(len(self.rolling_window)-tw):
            #K neighborhod graph
            data_tw = data.loc[self.rolling_window[x]:self.rolling_window[tw + x]-1]
            data_trans = data_tw.transpose()
            dist_kn = kneighbors_graph(data_trans, 4, mode='distance', include_self=False)
            dist = dist_kn.toarray()
            kneighbor_graph.append(dist)
                       
            dist[dist==0] = np.nan
            dist_rmean = np.nanmean(dist,axis = 1)
            dist[np.isnan(dist)] = 0
            kneighbor_rmean.append(dist_rmean)
            
        return (kneighbor_graph,kneighbor_rmean)
    
    #covariance matrix
    def calc_cov_corr (self):
        tw = 3
        #rw = 1 by default
        data = self.log_returns
        cov_list = []
        corr_list = []
        for x in range(len(self.rolling_window)-tw):
            #Fully connected graph
            data_tw = data.loc[self.rolling_window[x]:self.rolling_window[tw + x]-1]
            cov = np.cov(data_tw,rowvar = False)
            corr = np.corrcoef(data_tw,rowvar=False)
            cov_list.append(cov)
            corr_list.append(corr)
        return (cov_list,corr_list)
    
    
    #gaussian similarity matrix
    def calc_similaritymatrix(self,graph):
        W_k = []#similarity matrix for sigma=average of distance of k-neighbor
        W_e = []#similarity matrix for sigma=epsilon

        list_sigma_k = self.kneighbor_rmean
        list_sigma_e = self.epsilon
        rows, cols = (len(self.values.columns), len(self.values.columns)) 
        sym_matrix_k = np.array([[0.0 for i in range(cols)] for j in range(rows)])#Create empty array
        sym_matrix_e = np.array([[0.0 for i in range(cols)] for j in range(rows)])#Creat empty array
        #Determine single similarity matrix
        for k in range(len(graph)):
            dist_matrix = graph[k]
        
            sigma_k = list_sigma_k[k] #sigma_k is a 1d array
            sigma_e = list_sigma_e[k] #sigma e is a scalar
            
            for i in range(len(self.values.columns)):
                for j in range(len(self.values.columns)):
                    sym_matrix_k[i][j] = np.exp(-(dist_matrix[i][j]**2)/(2*sigma_k[i]*sigma_k[j]))
                    sym_matrix_e[i][j] = np.exp(-(dist_matrix[i][j]**2)/(2*sigma_e**2))
            W_k.append(np.copy(sym_matrix_k))
            W_e.append(np.copy(sym_matrix_e))
            
        return (W_k,W_e)
    
    #Laplacian
    def calc_laplacian(self,similarity_matrix):
        W = similarity_matrix
        D = []
        D_sym = []
        D_rw = []
        L  = []
        L_sym = []
        L_rw =[]
        for x in range(len(W)):
            w = W[x]
            w_rsum = np.sum(w,axis = 1)
        
            d = np.diagflat(w_rsum)
            d_sym = fractional_matrix_power(d, -0.5)
            d_rw = fractional_matrix_power(d, -1)
        
            l = d-w
            l_sym = np.matmul(d_sym,np.matmul(l,d_sym))
            l_rw = np.matmul(d_rw,l)
        
            D.append(d)
            D_sym.append(d_sym)
            D_rw.append(d_rw)
        
            L.append(l)
            L_sym.append(l_sym)
            L_rw.append(l_rw)
        return (L,L_sym,L_rw)

    #Eigenvalue and Eigenvector
    def calc_eigenvalue_vector(self):
        eigenvalue_L = []
        eigenvec_L = []
        eigenvalue_Lsym = []
        eigenvec_Lsym = []
        eigenvalue_Lrw = []
        eigenvec_Lrw = []
    
        for x in range(len(self.L)):
            #L
            evalue,evector = LA.eigh(self.L[x]) #optimized for symmetric matrices
            eigenvalue_L.append(evalue)
            eigenvec_L.append(evector)
            #L_sym
            evalue,evector = LA.eigh(self.L_sym[x]) #optimized for symmetric matrices
            eigenvalue_Lsym.append(evalue)
            eigenvec_Lsym.append(evector)
            #Lrw
            evalue,evector = LA.eig(self.L_rw[x])#optimized for non-symmetric matrices
            idx = np.argsort(evalue) #sort eigenvalue in increasing order
            evalue_sort = evalue[idx] #apply sorted index
            evector_sort = evector[:,idx] #apply sorted index
            eigenvalue_Lrw.append(evalue_sort)
            eigenvec_Lrw.append(evector_sort)
        return eigenvalue_L, eigenvec_L,eigenvalue_Lsym,eigenvec_Lsym,eigenvalue_Lrw,eigenvec_Lrw 
 
    #Number of Clusters
    def calc_numclusters(self):
        nCluster_L = []
        nCluster_Lsym = []
        nCluster_Lrw = []
    
        for x in range(len(self.eigenvalue_L)):
            #L
            diff_L = np.diff(self.eigenvalue_L[x])
            pos_L_max = np.argmax(diff_L)+1
            nCluster_L.append(pos_L_max)
            #Lsym
            diff_Lsym = np.diff(self.eigenvalue_Lsym[x])
            pos_Lsym_max = np.argmax(diff_Lsym)+1
            nCluster_Lsym.append(pos_Lsym_max)
            #Lrw
            diff_Lrw = np.diff(self.eigenvalue_Lrw[x])
            pos_Lrw_max = np.argmax(diff_Lrw)+1
            nCluster_Lrw.append(pos_Lrw_max)
        return (nCluster_L,nCluster_Lsym,nCluster_Lrw)
        
    #K-means on Laplacian and Laplacian Random Walk
    def calc_kmeans(self):
        kmeans_Cluster_L = []
        kmeans_Cluster_Lrw =[]
        for x in range(len(self.eigenvector_L)):
            eigvec_L = self.eigenvector_L[x]
            eigvec_Lrw = self.eigenvector_Lrw[x]
            eigvec_L_kmeans = eigvec_L[:,0:self.nCluster_L[x]]
            eigvec_Lrw_kmeans = eigvec_Lrw[:,0:self.nCluster_Lrw[x]]
        
            kmeans_L = KMeans(n_clusters = self.nCluster_L[x]).fit(eigvec_L_kmeans)
            kmeans_Lrw = KMeans(n_clusters = self.nCluster_Lrw[x]).fit(eigvec_Lrw_kmeans)
            kmeans_Cluster_L.append(kmeans_L.labels_)
            kmeans_Cluster_Lrw.append(kmeans_Lrw.labels_)
        return kmeans_Cluster_L,kmeans_Cluster_Lrw
    
    #K-means using Ng,Jordan,Weiss
    def calc_kmeans_Lsym(self):
        L_sym_norm = []
        denominator_list = []
        kmeans_Cluster_Lsym = []
        for x in range(len(self.eigenvector_Lsym)):
            eigvec_Lsym = self.eigenvector_Lsym[x] 
            eigvec_Lsym_kmeans = eigvec_Lsym[:,0:self.nCluster_Lsym[x]]    
            #Determine the denominator
            eigvec_Lsym_kmeans_1 = eigvec_Lsym_kmeans**2 #square of the elements
            #print(eigvec_Lsym_kmeans_1)
            eigvec_Lsym_kmeans_2 = np.sum(eigvec_Lsym_kmeans_1,axis = 1) #sum of the elements
            #print(eigvec_Lsym_kmeans_2 )
            denominator = np.array([eigvec_Lsym_kmeans_2**0.5 for j in range(self.nCluster_Lsym[x])]).transpose()
            #denominator = eigvec_Lsym_kmeans_2**0.5  #square of the elements
            #Normalazing Eigenvector
            eigvec_Lsym_norm = eigvec_Lsym_kmeans/denominator
            #Apply k-means of normalized Eigenvector
            kmeans_Lsym = KMeans(n_clusters = self.nCluster_Lsym[x]).fit(eigvec_Lsym_norm)
            #Creating lists
            denominator_list.append(denominator)
            L_sym_norm.append(eigvec_Lsym_norm)
            kmeans_Cluster_Lsym.append(kmeans_Lsym.labels_)        
        return kmeans_Cluster_Lsym

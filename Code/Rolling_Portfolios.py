import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from datetime import datetime
date_format = '%Y-%m-%d' # This depends on the data

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from functools import reduce

####################################################################
####################################################################
# Portfolio Weight functions



def calc_naive(close_prices):
    '''
    Calculates the equal contribution portfolio weights (1/N)
    '''
    num_assets = close_prices.shape[1]
    return np.full(num_assets,1 / num_assets)




def calc_MinVar_weights(close_prices):
    '''
    Calculate the weights for the minimum variance portfolio
    '''
    
    num_assets   = close_prices.shape[1]
    log_returns  = np.log(close_prices[1:] / close_prices[:-1])
    sigma        = np.cov(log_returns.T)  
    
    
    # Portfolio Variance function to minimize
    def f(w):
        return np.matmul(w.T, np.matmul(sigma*252, w))
    
    
    # Minimization params
    def con1(x):
        return x.sum() - 1
    
    cons = ({'type': 'eq', 'fun': con1})                  # Weights sum to 1  
    bnds = tuple([(0,1) for _ in range (0,num_assets)])   # Weights are positive
    init_guess = calc_naive(close_prices)                 # Initial guess is equal weights
    
    
    # Minimize the variance
    opt = minimize(f, init_guess, constraints=cons, bounds=bnds)
    

    
    # Check constraints were obeyed and minimization was succesful
    if opt.success == False:
        print('minimization failed')
    
    ones = np.full(num_assets,1)
    if (opt.x >= 0).all() == False:
        print('negative weights')
        print(opt.x)
        
    if abs(ones.dot(opt.x) - 1) > 0.0001:
        print('constraints failed')
        print(ones.dot(opt.x))
    
    # Return the minimal weights
    return opt.x





# Equal Risk Contribution Portfolio

def calc_sigma(prices):
    #norm_returns = prices / prices[0]
    log_returns = np.log(prices[1:] / prices[:-1])
    sigma = np.cov(log_returns.T)
    return sigma

# Kronecker Delta
def k_delt(i,j):
    if i == j:
        return 1
    else:
        return 0
    
    
    

def calc_ERC_weights(close_prices):
    '''
    Calculate the weights for the risk parity portfolio
    '''
    num_assets   = close_prices.shape[1]
    log_returns  = np.log(close_prices[1:] / close_prices[:-1])
    sigma = np.cov(log_returns.T)
    
    
    # Minimization params
    init_guess = np.full(num_assets,1 / num_assets)
    LC  = LinearConstraint(np.full(num_assets,1),0.9999,1.0001,keep_feasible=True)
    LC2 = LinearConstraint(np.identity(num_assets),0,1,keep_feasible=True)
    
    
    # Function to minimize for risk parity
    def f(w):
        N = num_assets
        sig = np.sqrt(np.matmul(w.T , np.matmul(sigma, w)))
        r = 0
        for i in range(0,N):
            r = r + (w[i] - (sig**2 / (N*np.matmul(sigma, w)[i]))  )**2
        return r
    
    # Jacobian - required for minimization method
    def Df(w):
    
        N   = w.shape[0]
        sig = np.sqrt(np.matmul(w.T , np.matmul(sigma, w)))
        Df  = np.zeros(N)
        
        for k in range(0,N):
            
            Df_k = 0
            for i in range(0,N):
        
                A = w[i] - (sig**2 / N*np.matmul(sigma, w)[i])
                B = 2*np.matmul(sigma, w)[i]*np.matmul(sigma, w)[k]
                C = (sig**2)*sigma[i,k]
                D = N*np.matmul(sigma, w)[i]
        
                Df_k = Df_k + 2*A*(k_delt(k,i) - ((B - C) / D) )
        
            Df[k] = Df_k
        
        return Df
    

    
    opt = minimize(f,init_guess,jac=Df,hess='cs',method='trust-constr',constraints=[LC,LC2])
    
    # Check constraints were obeyed and minimization was succesful
    ones = np.full(num_assets,1)
    if (opt.x >= 0).all() == False:
        print('negative weights')
        print(opt.x)
    if opt.status in [0,3]:
        print('minimize did not converge')
        print(opt.status)
    if abs(ones.dot(opt.x) - 1) > 0.0001:
        print('constraints failed')
        print(ones.dot(opt.x))

    # return minimal weights
    return opt.x

####################################################################
####################################################################



def calc_sharpe_ratio(w,X):
    ''' 
    Given prices X calculate the sharperatio of a portfolio with weights w
    '''
    W = X.dot(w)                                       # Portfolio time series
    logR = np.log(W[1:] / W[:-1])                      # Portfolio log returns
    S = (np.mean(logR) / np.std(logR))*np.sqrt(252)    # Sharpe Ratio
    return S


def to_datetime(date_string):
    '''
    Convert datestring to datetime object, assumes format of datestring
    '''
    if type(date_string) == str:
        return datetime.strptime(date_string,date_format)
    else:
        return date_string

    
    
class AssetClass():
    
    # Generally takes in a pandas dataframe with a date index and multiple float columns
    # Values can be NaN
    
    
    def __init__(self,df,date_column='Date'):
        
        self.dates = np.array(df.index.to_list())
        
        self.values      = df.reset_index(drop=True).values.astype('float')
        self.log_returns = np.log(self.values[1:] / self.values[:-1])
        self.num_assets  = self.values.shape[1]
        self.first_ofs   = self.calc_first_ofs(self.dates)  

    def fit(self):
        
        self.calc_rolling_portfolio_weights()
        
        C = ['Start Date','Strategy Code', 'Strategy Name','1/N Sharpe','ERC Sharpe','MinVar Sharpe']
        #T = np.take(self.dates,self.first_ofs[3:-3].astype(int))
        D = np.concatenate((np.take(self.dates,self.first_ofs[3:-3].astype(int))[:,None],\
                            self.rolling_labels[:,None],\
                            self.rolling_strategies[:,None],\
                            self.rolling_ratios), axis=1)
        
        
        self.train_score_data = pd.DataFrame(data=D,columns=C).astype({'Strategy Code': 'int32'})
        

    # ------------------------------------------
        
    def calc_rolling_portfolio_weights(self):
        
        d = {0:'1/N', 1:'ERC', 2:'MinVar'}           # strategy index dictionary
        top_strategies = np.array([''])              # array for name of optimal strategy
        weights = np.zeros((1,3,self.num_assets))    # array for rolling portfolio weights
        ratios  = np.zeros((1,3))                    # array for rollong portfolio weights
        labels = []                                  # array for index of optimal strategy
        

        for i,j in zip(self.first_ofs[3:-3], self.first_ofs[6:]):
            
            # Take three month data window
            X = self.values[int(i):int(j),:]
            
            # Calc three portfolios for this window
            w0 = calc_naive(X)
            w1 = calc_ERC_weights(X)
            w2 = calc_MinVar_weights(X)
            
            # Combine three portfolio weightings to single array
            w = np.vstack((w1,w2))
            w = np.vstack((w0,w))
            w = np.reshape(w,(1,3,self.num_assets))
            
            # Calc sharpe Ratios for all three strategies
            s = np.array([calc_sharpe_ratio(w0,X),\
                          calc_sharpe_ratio(w1,X),\
                          calc_sharpe_ratio(w2,X)])
            s = np.reshape(s,(1,3))
            
            # Get optimal strategy
            strategy = np.array([d[np.argmax(s)]])
            
            # Add weights, sharpe ratios, and optimal stategy into main arrays
            weights        = np.concatenate((weights,w),axis=0)
            ratios         = np.concatenate((ratios,s),axis=0)
            top_strategies = np.concatenate((top_strategies,strategy))
            labels.append(np.argmax(s))
            
            
        weights        = weights[1:,:,:]
        ratios         = ratios[1:,:]
        top_strategies = top_strategies[1:]
        
        self.rolling_weights    = weights
        self.rolling_ratios     = ratios
        self.rolling_strategies = top_strategies
        self.rolling_labels     = np.array(labels)
        
        
    # ------------------------------------------
    
    def calc_first_ofs(self,date_list):
        '''
        Creates list of index locations for the first trading day of each month
        '''
        new_month_indicies = np.array([])
        i = 0
        current_month = 13
        
        for i in range(0,len(date_list)):
            date = date_list[i]
            if date.month != current_month:
                new_month_indicies = np.append(new_month_indicies,int(i))
                current_month = date.month
        
        return new_month_indicies



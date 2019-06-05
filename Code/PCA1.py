#import csv
import pandas as pd
import math
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#import datetime as dt
#import array as arr

#Graphics
#a = a.cumsum()
fig, ax = plt.subplots(1,1,figsize = (20, 7))
ax.plot(EV_CC.loc[:,[0,1,2,3,4]])
#plt.ylim(0.0,1)
#plt.show()
#fig = plt.subplots(1,1,figsize = (20, 7))
EV_CC.plot(x='Date', y=[0,1,2],figsize = (20,7))
plt.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
#Call functions
#Equity Industry
EV_Equity_Ind = pcafunction (M_Equity_adj.loc[:,['Date','Financials','Energy','Materials','Industrials','Info Tech',
                               'Cons. Discretionary','Health Care','Telecom','Cons. Staples',
                               'Real Estate','Utilities']])
EV_Equity_LvsS1 = pcafunction (M_Equity_adj.loc[:,['Date','LargeCap','SmallCap']])
EV_Equity_LvsS2 = pcafunction (M_Equity_adj.loc[:,['Date','LargeCap Value','LargeCap Growth','SmallCap Value','SmallCap Growth']])

#Fixed Income Industry
#Cryptocurrencies
EV_CC = pcafunction (M_CC_adj)


#functions
def pcafunction (DF_input):
    #time window
    tw = 10 #variable
    #weights = weightfunction (list(range(tw))) #fix
    DF_input_r = returnfunction(DF_input) #fix
    #PCA Model
    components = DF_input_r.shape[1] #variable, DF_input_r.shape[1] ==> number eigenvectors = features(columns of data frame)  
    my_model = PCA(n_components = components)#fix
    my_model.fit(DF_input_r) 
    my_model.explained_variance_ratio_
    
def rollpcafunction (DF_input):
    #time window
    tw = 10 #variable
    #functions 
    weights = weightfunction (list(range(tw))) #fix
    DF_input_r = returnfunction(DF_input) #fix
    #PCA Model
    components = DF_input_r.shape[1] #variable, DF_input_r.shape[1] ==> number eigenvectors = features(columns of data frame)  
    my_model = PCA(n_components = components)#fix
    #date for new data frame of eigenvalues
    date = DF_input['Date'].loc[DF_input.index[0]+tw-1:(DF_input.shape[0]+DF_input.index[0]-2)]#fix
    exp_var = pd.DataFrame() #DF_input['Date'].loc[tw-1:DF_input.shape[0]-2]
    for x in range(DF_input_r.index[0],(DF_input_r.shape[0]+DF_input_r.index[0]-(tw-1))):
        a = DF_input_r.loc[x:tw-1+x].apply(lambda z: z*weights, axis=0)
        my_model.fit(a) 
        exp_var = pd.concat([exp_var,pd.DataFrame(my_model.explained_variance_ratio_)],
                             axis = 1) 
    exp_var = exp_var.transpose()
    exp_var.index = range(0,exp_var.shape[0]) #Aligning index
    date.index = range(0,date.shape[0]) #Aligining index
    return pd.concat([date,exp_var],axis = 1)

def returnfunction (DF_input1):  #function transforming price in return
    DF_input1 = DF_input1.loc[:,DF_input1.columns != 'Date'].applymap(math.log)#applying log-function
    r = pd.DataFrame(data = (np.diff(DF_input1,axis =0,n=1)))
    #r = np.divide(div,DF_input1.loc[DF_input1.index[0]:(DF_input1.shape[0]+DF_input1.index[0]-2),    #DF_input1.index[0]
                                    #DF_input1.columns != 'Date'])   
    return r



def weightfunction (count): #function calculating weight-vector
    w = []
    s = 0
    for x in list(range(10)): #count 
        w.append(math.exp(x))
        s = math.exp(x)+s
    w = np.asarray(w)
    w = w/s
    return w

#Import Data of Assets
M = pd.read_excel('C:/Users/fjaeckle/Documents/FIM/Master-Thesis/Data/Bloomberg Data.xlsx',sheet_name = 1)

###########################Creating Equity Data Frame###########################
M_Equity = pd.DataFrame(data = (M['Date.1'].dt.strftime('%Y-%m-%d'),M['SPX Index'],
                                M['Date.2'].dt.strftime('%Y-%m-%d'),M['SVX Index'],
                                M['Date.3'].dt.strftime('%Y-%m-%d'),M['SGX Index'],
                                M['Date.4'].dt.strftime('%Y-%m-%d'),M['M1USSC Index'],
                                M['Date.5'].dt.strftime('%Y-%m-%d'),M['MZUSSV Index'],
                                M['Date.6'].dt.strftime('%Y-%m-%d'),M['MZUSSG Index'],
                                M['Date.7'].dt.strftime('%Y-%m-%d'),M['S5FINL Index'],
                                M['Date.8'].dt.strftime('%Y-%m-%d'),M['S5ENRS Index'],
                                M['Date.9'].dt.strftime('%Y-%m-%d'),M['S5MATR Index'],
                                M['Date.10'].dt.strftime('%Y-%m-%d'),M['S5INDU Index'],
                                M['Date.11'].dt.strftime('%Y-%m-%d'),M['S5INFT Index'],
                                M['Date.12'].dt.strftime('%Y-%m-%d'),M['S5COND Index'],
                                M['Date.13'].dt.strftime('%Y-%m-%d'),M['S5HLTH Index'],
                                M['Date.14'].dt.strftime('%Y-%m-%d'),M['S5TELS Index'],
                                M['Date.15'].dt.strftime('%Y-%m-%d'),M['S5CONS Index'],
                                M['Date.16'].dt.strftime('%Y-%m-%d'),M['S5RLST Index'],
                                M['Date.17'].dt.strftime('%Y-%m-%d'),M['S5UTIL Index']))
                       
M_Equity = M_Equity.transpose() # Transpose M_Equity Data Frame

#Adjust data types of Data Frame
for x in M_Equity.columns:
    if 'Date' in x:
        M_Equity[x] = pd.to_datetime(M_Equity[x],format='%Y-%m-%d')
    else:
        M_Equity[x] = pd.to_numeric(M_Equity[x])   
        
#Creating Data Frame for Single Equity Index
M_Equity_SPX = M_Equity.loc[:,['Date.1','SPX Index']]
M_Equity_SPX.columns = ['Date','LargeCap']
M_Equity_SVX = M_Equity.loc[:,['Date.2','SVX Index']]
M_Equity_SVX.columns = ['Date','LargeCap Value']
M_Equity_SGX = M_Equity.loc[:,['Date.3','SGX Index']]
M_Equity_SGX.columns = ['Date','LargeCap Growth']
M_Equity_M1USSC = M_Equity.loc[:,['Date.4','M1USSC Index']]
M_Equity_M1USSC.columns = ['Date','SmallCap']
M_Equity_MZUSSV = M_Equity.loc[:,['Date.5','MZUSSV Index']]
M_Equity_MZUSSV.columns = ['Date','SmallCap Value'] 
M_Equity_MZUSSG = M_Equity.loc[:,['Date.6','MZUSSG Index']]
M_Equity_MZUSSG.columns = ['Date','SmallCap Growth'] 
M_Equity_S5FINL = M_Equity.loc[:,['Date.7','S5FINL Index']]
M_Equity_S5FINL.columns = ['Date','Financials'] 
M_Equity_S5ENRS = M_Equity.loc[:,['Date.8','S5ENRS Index']]
M_Equity_S5ENRS.columns = ['Date','Energy'] 
M_Equity_S5MATR = M_Equity.loc[:,['Date.9','S5MATR Index']]
M_Equity_S5MATR.columns = ['Date','Materials'] 
M_Equity_S5INDU = M_Equity.loc[:,['Date.10','S5INDU Index']]
M_Equity_S5INDU.columns = ['Date','Industrials'] 
M_Equity_S5INFT = M_Equity.loc[:,['Date.11','S5INFT Index']]
M_Equity_S5INFT.columns = ['Date','Info Tech'] 
M_Equity_S5COND = M_Equity.loc[:,['Date.12','S5COND Index']]
M_Equity_S5COND.columns = ['Date','Cons. Discretionary'] 
M_Equity_S5HLTH = M_Equity.loc[:,['Date.13','S5HLTH Index']]
M_Equity_S5HLTH.columns = ['Date','Health Care'] 
M_Equity_S5TELS = M_Equity.loc[:,['Date.14','S5TELS Index']]
M_Equity_S5TELS.columns = ['Date','Telecom'] 
M_Equity_S5CONS = M_Equity.loc[:,['Date.15','S5CONS Index']]
M_Equity_S5CONS.columns = ['Date','Cons. Staples'] 
M_Equity_S5RLST = M_Equity.loc[:,['Date.16','S5RLST Index']]
M_Equity_S5RLST.columns = ['Date','Real Estate'] 
M_Equity_S5UTIL = M_Equity.loc[:,['Date.17','S5UTIL Index']]
M_Equity_S5UTIL.columns = ['Date','Utilities'] 

#Creating adj. Equity Data Frame
M_Equity_adj = M_Equity_SPX
for x in (M_Equity_SVX,M_Equity_SGX,M_Equity_M1USSC,M_Equity_MZUSSV,M_Equity_MZUSSG,
          M_Equity_S5FINL,M_Equity_S5ENRS,M_Equity_S5MATR,M_Equity_S5INDU,M_Equity_S5INFT,
          M_Equity_S5COND,M_Equity_S5HLTH,M_Equity_S5TELS,M_Equity_S5CONS,M_Equity_S5RLST,
          M_Equity_S5UTIL):
    M_Equity_adj = pd.merge(M_Equity_adj,x,on = 'Date')
                        
#Delete Single Equity Data Frames                         
del M_Equity_SPX
del M_Equity_SVX
del M_Equity_SGX
del M_Equity_M1USSC
del M_Equity_MZUSSV
del M_Equity_MZUSSG
del M_Equity_S5FINL
del M_Equity_S5ENRS
del M_Equity_S5MATR
del M_Equity_S5INDU
del M_Equity_S5INFT
del M_Equity_S5COND
del M_Equity_S5HLTH
del M_Equity_S5TELS
del M_Equity_S5CONS
del M_Equity_S5RLST
del M_Equity_S5UTIL

########################Creating Data Frame for Fixed Income#########################
M_FI = pd.DataFrame(data =     (M['Date.18'].dt.strftime('%Y-%m-%d'),M['LBUSTRUU Index'],
                                M['Date.19'].dt.strftime('%Y-%m-%d'),M['LUATTRUU Index'],
                                M['Date.20'].dt.strftime('%Y-%m-%d'),M['LUACTRUU Index'],
                                M['Date.21'].dt.strftime('%Y-%m-%d'),M['LF98TRUU Index'],
                                M['Date.22'].dt.strftime('%Y-%m-%d'),M['LGTRTRUU Index'],
                                M['Date.23'].dt.strftime('%Y-%m-%d'),M['LEGATRUU Index'],
                                M['Date.24'].dt.strftime('%Y-%m-%d'),M['LECPTREU Index'],
                                M['Date.25'].dt.strftime('%Y-%m-%d'),M['LBEATREU Index'],
                                M['Date.26'].dt.strftime('%Y-%m-%d'),M['LACHTRUU Index'],
                                M['Date.27'].dt.strftime('%Y-%m-%d'),M['LAPCTRJU Index'],
                                M['Date.28'].dt.strftime('%Y-%m-%d'),M['LP06TREU Index'],
                                M['Date.29'].dt.strftime('%Y-%m-%d'),M['LP01TREU Index']))

M_FI = M_FI.transpose() # Transpose M_FI Data Frame

#Adjust data types of Data Frame
for x in M_FI.columns:
    if 'Date' in x:
        M_FI[x] = pd.to_datetime(M_FI[x],format='%Y-%m-%d')
    else:
        M_FI[x] = pd.to_numeric(M_FI[x])   
        
#Creating Data Frame for Single FI Index
M_FI_LBUSTRUU = M_FI.loc[:,['Date.18','LBUSTRUU Index']]
M_FI_LBUSTRUU.columns = ['Date','US Aggregate']
M_FI_LBUSTRUU = M_FI_LBUSTRUU.dropna(axis = 0,how = 'any')

M_FI_LUATTRUU = M_FI.loc[:,['Date.19','LUATTRUU Index']]
M_FI_LUATTRUU.columns = ['Date','US Treasury']
M_FI_LUATTRUU = M_FI_LUATTRUU.dropna(axis = 0,how = 'any')

M_FI_LUACTRUU = M_FI.loc[:,['Date.20','LUACTRUU Index']]
M_FI_LUACTRUU.columns = ['Date','US Corportate']
M_FI_LUACTRUU = M_FI_LUACTRUU.dropna(axis = 0,how = 'any')

M_FI_LF98TRUU = M_FI.loc[:,['Date.21','LF98TRUU Index']]
M_FI_LF98TRUU.columns = ['Date','US High Yield']
M_FI_LUATTRUU = M_FI_LUATTRUU.dropna(axis = 0,how = 'any')

M_FI_LGTRTRUU = M_FI.loc[:,['Date.22','LGTRTRUU Index']]
M_FI_LGTRTRUU.columns = ['Date','Global Treasury'] 
M_FI_LGTRTRUU = M_FI_LGTRTRUU.dropna(axis = 0,how = 'any')

M_FI_LEGATRUU = M_FI.loc[:,['Date.23','LEGATRUU Index']]
M_FI_LEGATRUU.columns = ['Date','Global Aggregate']
M_FI_LEGATRUU = M_FI_LEGATRUU.dropna(axis = 0,how = 'any')
 
M_FI_LECPTREU = M_FI.loc[:,['Date.24','LECPTREU Index']]
M_FI_LECPTREU.columns = ['Date','Euro Corporate'] 
M_FI_LECPTREU = M_FI_LECPTREU.dropna(axis = 0,how = 'any')

M_FI_LBEATREU = M_FI.loc[:,['Date.25','LBEATREU Index']]
M_FI_LBEATREU.columns = ['Date','Euro Aggregate']
M_FI_LBEATREU = M_FI_LBEATREU.dropna(axis = 0,how = 'any')
 
M_FI_LACHTRUU = M_FI.loc[:,['Date.26','LACHTRUU Index']]
M_FI_LACHTRUU.columns = ['Date','China Aggregate'] 
M_FI_LACHTRUU = M_FI_LACHTRUU.dropna(axis = 0,how = 'any')

M_FI_LAPCTRJU = M_FI.loc[:,['Date.27','LAPCTRJU Index']]
M_FI_LAPCTRJU.columns = ['Date','Asian-Pacific Aggregate'] 
M_FI_LAPCTRJU = M_FI_LAPCTRJU.dropna(axis = 0,how = 'any')

M_FI_LP06TREU = M_FI.loc[:,['Date.28','LP06TREU Index']]
M_FI_LP06TREU.columns = ['Date','Pan-European Aggregate']
M_FI_LP06TREU = M_FI_LP06TREU.dropna(axis = 0,how = 'any')
 
M_FI_LP01TREU = M_FI.loc[:,['Date.29','LP01TREU Index']]
M_FI_LP01TREU.columns = ['Date','Pan-European High Yield'] 
M_FI_LP01TREU = M_FI_LP01TREU.dropna(axis = 0,how = 'any')

#Creating adj. Fixed Income Data Frame
M_FI_adj = M_FI_LBUSTRUU
for x in (M_FI_LUATTRUU,M_FI_LUACTRUU,M_FI_LF98TRUU,M_FI_LGTRTRUU,M_FI_LEGATRUU,
          M_FI_LECPTREU,M_FI_LBEATREU,M_FI_LACHTRUU,M_FI_LAPCTRJU,M_FI_LP06TREU,M_FI_LP01TREU):
    M_FI_adj = pd.merge(M_FI_adj,x,on = 'Date')
                            
#Delete Single FI Data Frames                         
del M_FI_LBUSTRUU
del M_FI_LUATTRUU
del M_FI_LUACTRUU
del M_FI_LF98TRUU
del M_FI_LGTRTRUU

del M_FI_LEGATRUU
del M_FI_LECPTREU
del M_FI_LBEATREU
del M_FI_LACHTRUU
del M_FI_LAPCTRJU
del M_FI_LP06TREU
del M_FI_LP01TREU


########################Creating Data Frame for Commodity#########################
M_Comdty = pd.DataFrame(data = (M['Date.30'].dt.strftime('%Y-%m-%d'),M['CRY Index'],
                                M['Date.31'].dt.strftime('%Y-%m-%d'),M['BCOM Index'],
                                M['Date.32'].dt.strftime('%Y-%m-%d'),M['CLN9 Comdty'],
                                M['Date.33'].dt.strftime('%Y-%m-%d'),M['NGN19 Comdty'],
                                M['Date.34'].dt.strftime('%Y-%m-%d'),M['HGN9 Comdty'],
                                M['Date.35'].dt.strftime('%Y-%m-%d'),M['GCQ9 Comdty'],
                                M['Date.36'].dt.strftime('%Y-%m-%d'),M['C N9 Comdty'],
                                M['Date.37'].dt.strftime('%Y-%m-%d'),M['COU9 Comdty'],
                                M['Date.38'].dt.strftime('%Y-%m-%d'),M['S N9 Comdty'],
                                M['Date.39'].dt.strftime('%Y-%m-%d'),M['LAN19 Comdty'],
                                M['Date.40'].dt.strftime('%Y-%m-%d'),M['SIN9 Comdty'],
                                M['Date.41'].dt.strftime('%Y-%m-%d'),M['CON9 Comdty'],
                                M['Date.42'].dt.strftime('%Y-%m-%d'),M['LXN9 Comdty'],
                                M['Date.43'].dt.strftime('%Y-%m-%d'),M['XBN9 Comdty'],
                                M['Date.44'].dt.strftime('%Y-%m-%d'),M['SMN9 Comdty'],
                                M['Date.45'].dt.strftime('%Y-%m-%d'),M['QSN9 Comdty'],
                                M['Date.46'].dt.strftime('%Y-%m-%d'),M['SBN9 Comdty'],
                                M['Date.47'].dt.strftime('%Y-%m-%d'),M['BON9 Comdty'],
                                M['Date.48'].dt.strftime('%Y-%m-%d'),M['LNN9 Comdty'],
                                M['Date.49'].dt.strftime('%Y-%m-%d'),M['W N9 Comdty'],
                                M['Date.50'].dt.strftime('%Y-%m-%d'),M['HON9 Comdty'],
                                M['Date.51'].dt.strftime('%Y-%m-%d'),M['KCN9 Comdty'],
                                M['Date.52'].dt.strftime('%Y-%m-%d'),M['LCQ9 Comdty'],
                                M['Date.53'].dt.strftime('%Y-%m-%d'),M['LHN9 Comdty'],
                                M['Date.54'].dt.strftime('%Y-%m-%d'),M['CTN9 Comdty'],
                                M['Date.55'].dt.strftime('%Y-%m-%d'),M['LHM9 Comdty'],
                                M['Date.56'].dt.strftime('%Y-%m-%d'),M['KWN9 Comdty']))

M_Comdty = M_Comdty.transpose() # Transpose Comdty Data Frame

#Adjust data types of Data Frame
for x in M_Comdty.columns:
    if 'Date' in x:
        M_Comdty[x] = pd.to_datetime(M_Comdty[x],format='%Y-%m-%d')
    else:
        M_Comdty[x] = pd.to_numeric(M_Comdty[x])   
        
#Creating Data Frame for Single Comdty Index
M_Comdty_CRY = M_Comdty.loc[:,['Date.30','CRY Index']]
M_Comdty_CRY.columns = ['Date','CRY Index']
M_Comdty_CRY = M_Comdty_CRY.dropna(axis = 0,how = 'any')

M_Comdty_BCOM = M_Comdty.loc[:,['Date.31','BCOM Index']]
M_Comdty_BCOM.columns = ['Date','BCOM Index']
M_Comdty_BCOM = M_Comdty_BCOM.dropna(axis = 0,how = 'any')

M_Comdty_CLN9 = M_Comdty.loc[:,['Date.32','CLN9 Comdty']]
M_Comdty_CLN9.columns = ['Date','CLN9']
M_Comdty_CLN9 = M_Comdty_CLN9.dropna(axis = 0,how = 'any')

M_Comdty_NGN19 = M_Comdty.loc[:,['Date.33','NGN19 Comdty']]
M_Comdty_NGN19.columns = ['Date','NGN19']
M_Comdty_NGN19 = M_Comdty_NGN19.dropna(axis = 0,how = 'any')

M_Comdty_HGN9 = M_Comdty.loc[:,['Date.34','HGN9 Comdty']]
M_Comdty_HGN9.columns = ['Date','HGN9 Comdty'] 
M_Comdty_HGN9 = M_Comdty_HGN9.dropna(axis = 0,how = 'any')

M_Comdty_GCQ9 = M_Comdty.loc[:,['Date.35','GCQ9 Comdty']]
M_Comdty_GCQ9.columns = ['Date','GCQ9']
M_Comdty_GCQ9 = M_Comdty_GCQ9.dropna(axis = 0,how = 'any')
 
M_Comdty_CN9 = M_Comdty.loc[:,['Date.36','C N9 Comdty']]
M_Comdty_CN9.columns = ['Date','CN9'] 
M_Comdty_CN9 = M_Comdty_CN9.dropna(axis = 0,how = 'any')

M_Comdty_COU9 = M_Comdty.loc[:,['Date.37','COU9 Comdty']]
M_Comdty_COU9.columns = ['Date','COU9']
M_Comdty_COU9 = M_Comdty_COU9.dropna(axis = 0,how = 'any')
 
M_Comdty_SN9 = M_Comdty.loc[:,['Date.38','S N9 Comdty']]
M_Comdty_SN9.columns = ['Date','SN9'] 
M_Comdty_SN9 = M_Comdty_SN9.dropna(axis = 0,how = 'any')

M_Comdty_LAN19 = M_Comdty.loc[:,['Date.39','LAN19 Comdty']]
M_Comdty_LAN19.columns = ['Date','LAN19'] 
M_Comdty_LAN19 = M_Comdty_LAN19.dropna(axis = 0,how = 'any')

M_Comdty_SIN9 = M_Comdty.loc[:,['Date.40','SIN9 Comdty']]
M_Comdty_SIN9.columns = ['Date','SIN9'] 
M_Comdty_SIN9 = M_Comdty_SIN9.dropna(axis = 0,how = 'any')

M_Comdty_CON9 = M_Comdty.loc[:,['Date.41','CON9 Comdty']]
M_Comdty_CON9.columns = ['Date','CON9']
M_Comdty_CON9 = M_Comdty_CON9.dropna(axis = 0,how = 'any')
 
M_Comdty_LXN9 = M_Comdty.loc[:,['Date.42','LXN9 Comdty']]
M_Comdty_LXN9.columns = ['Date','LXN9'] 
M_Comdty_LXN9 = M_Comdty_LXN9.dropna(axis = 0,how = 'any')

M_Comdty_XBN9 = M_Comdty.loc[:,['Date.43','XBN9 Comdty']]
M_Comdty_XBN9.columns = ['Date','XBN9'] 
M_Comdty_XBN9 = M_Comdty_XBN9.dropna(axis = 0,how = 'any')

M_Comdty_SMN9 = M_Comdty.loc[:,['Date.44','SMN9 Comdty']]
M_Comdty_SMN9.columns = ['Date','SMN9'] 
M_Comdty_SMN9 = M_Comdty_SMN9.dropna(axis = 0,how = 'any')

M_Comdty_QSN9 = M_Comdty.loc[:,['Date.45','QSN9 Comdty']]
M_Comdty_QSN9.columns = ['Date','QSN9'] 
M_Comdty_QSN9 = M_Comdty_QSN9.dropna(axis = 0,how = 'any')

M_Comdty_SBN9 = M_Comdty.loc[:,['Date.46','SBN9 Comdty']]
M_Comdty_SBN9.columns = ['Date','SBN9'] 
M_Comdty_SBN9 = M_Comdty_SBN9.dropna(axis = 0,how = 'any')

M_Comdty_BON9 = M_Comdty.loc[:,['Date.47','BON9 Comdty']]
M_Comdty_BON9.columns = ['Date','BON9'] 
M_Comdty_BON9 = M_Comdty_BON9.dropna(axis = 0,how = 'any')

M_Comdty_LNN9 = M_Comdty.loc[:,['Date.48','LNN9 Comdty']]
M_Comdty_LNN9.columns = ['Date','LNN9'] 
M_Comdty_LNN9 = M_Comdty_LNN9.dropna(axis = 0,how = 'any')

M_Comdty_WN9 = M_Comdty.loc[:,['Date.49','W N9 Comdty']]
M_Comdty_WN9.columns = ['Date','WN9'] 
M_Comdty_WN9 = M_Comdty_WN9.dropna(axis = 0,how = 'any')

M_Comdty_HON9 = M_Comdty.loc[:,['Date.50','HON9 Comdty']]
M_Comdty_HON9.columns = ['Date','HON9'] 
M_Comdty_HON9 = M_Comdty_HON9.dropna(axis = 0,how = 'any')

M_Comdty_KCN9 = M_Comdty.loc[:,['Date.51','KCN9 Comdty']]
M_Comdty_KCN9.columns = ['Date','KCN9'] 
M_Comdty_KCN9 = M_Comdty_KCN9.dropna(axis = 0,how = 'any')

M_Comdty_LCQ9 = M_Comdty.loc[:,['Date.52','LCQ9 Comdty']]
M_Comdty_LCQ9.columns = ['Date','LCQ9'] 
M_Comdty_LCQ9 = M_Comdty_LCQ9.dropna(axis = 0,how = 'any')

M_Comdty_LHN9 = M_Comdty.loc[:,['Date.53','LHN9 Comdty']]
M_Comdty_LHN9.columns = ['Date','LHN9'] 
M_Comdty_LHN9 = M_Comdty_LHN9.dropna(axis = 0,how = 'any')

M_Comdty_CTN9 = M_Comdty.loc[:,['Date.54','CTN9 Comdty']]
M_Comdty_CTN9.columns = ['Date','CTN9'] 
M_Comdty_CTN9 = M_Comdty_CTN9.dropna(axis = 0,how = 'any')

M_Comdty_LHM9 = M_Comdty.loc[:,['Date.55','LHM9 Comdty']]
M_Comdty_LHM9.columns = ['Date','LHM9'] 
M_Comdty_LHM9 = M_Comdty_LHM9.dropna(axis = 0,how = 'any')

M_Comdty_KWN9 = M_Comdty.loc[:,['Date.56','KWN9 Comdty']]
M_Comdty_KWN9.columns = ['Date','KWN9'] 
M_Comdty_KWN9 = M_Comdty_KWN9.dropna(axis = 0,how = 'any')

#Creating adj. Comdty Data Frame
M_Comdty_adj = M_Comdty_CRY
for x in (M_Comdty_BCOM,M_Comdty_CLN9,M_Comdty_NGN19,M_Comdty_HGN9,
          M_Comdty_GCQ9,M_Comdty_CN9,M_Comdty_COU9,M_Comdty_SN9,M_Comdty_LAN19,M_Comdty_SIN9,
          M_Comdty_CON9,M_Comdty_LXN9,M_Comdty_XBN9,M_Comdty_SMN9,M_Comdty_QSN9,M_Comdty_SBN9,
          M_Comdty_BON9,M_Comdty_LNN9,M_Comdty_WN9,M_Comdty_HON9,M_Comdty_KCN9,M_Comdty_LCQ9,
          M_Comdty_LHN9,M_Comdty_CTN9,M_Comdty_LHM9,M_Comdty_KWN9):
    M_Comdty_adj = pd.merge(M_Comdty_adj,x,on = 'Date')
                            
#Delete Single Comdty Data Frames   
del M_Comdty_CRY                      
del M_Comdty_BCOM
del M_Comdty_CLN9
del M_Comdty_NGN19 
del M_Comdty_HGN9
del M_Comdty_GCQ9
del M_Comdty_CN9
del M_Comdty_COU9 
del M_Comdty_SN9 
del M_Comdty_LAN19 
del M_Comdty_SIN9
del M_Comdty_CON9
del M_Comdty_LXN9 
del M_Comdty_XBN9 
del M_Comdty_SMN9 
del M_Comdty_QSN9 
del M_Comdty_SBN9
del M_Comdty_BON9 
del M_Comdty_LNN9 
del M_Comdty_WN9
del M_Comdty_HON9
del M_Comdty_KCN9 
del M_Comdty_LCQ9
del M_Comdty_LHN9
del M_Comdty_CTN9 
del M_Comdty_LHM9 

del M_Comdty_KWN9

########################Creating Data Frame for Hedge Funds#########################
M_HF = pd.DataFrame(data =     (M['Date.57'].dt.strftime('%Y-%m-%d'),M['HFRXGL Index'],
                                M['Date.58'].dt.strftime('%Y-%m-%d'),M['HFRXEH Index'],
                                M['Date.59'].dt.strftime('%Y-%m-%d'),M['HFRXM Index'],
                                M['Date.60'].dt.strftime('%Y-%m-%d'),M['HFRXED Index'],
                                M['Date.61'].dt.strftime('%Y-%m-%d'),M['HFRXFIC Index'],
                                M['Date.62'].dt.strftime('%Y-%m-%d'),M['HFRXDS Index'],
                                M['Date.63'].dt.strftime('%Y-%m-%d'),M['HFRXAR Index'],
                                M['Date.64'].dt.strftime('%Y-%m-%d'),M['HFRXRVA Index']))

M_HF = M_HF.transpose() # Transpose HF Data Frame

#Adjust data types of Data Frame
for x in M_HF.columns:
    if 'Date' in x:
        M_HF[x] = pd.to_datetime(M_HF[x],format='%Y-%m-%d')
    else:
        M_HF[x] = pd.to_numeric(M_HF[x])   
        
#Creating Data Frame for Single HF Index
M_HF_HFRXGL = M_HF.loc[:,['Date.57','HFRXGL Index']]
M_HF_HFRXGL.columns = ['Date','HF_Global']
M_HF_HFRXGL = M_HF_HFRXGL.dropna(axis = 0,how = 'any')

M_HF_HFRXEH = M_HF.loc[:,['Date.58','HFRXEH Index']]
M_HF_HFRXEH.columns = ['Date','HF_Equity']
M_HF_HFRXEH = M_HF_HFRXEH.dropna(axis = 0,how = 'any')

M_HF_HFRXM = M_HF.loc[:,['Date.59','HFRXM Index']]
M_HF_HFRXM.columns = ['Date','HF_Macro']
M_HF_HFRXM = M_HF_HFRXM.dropna(axis = 0,how = 'any')

M_HF_HFRXED = M_HF.loc[:,['Date.60','HFRXED Index']]
M_HF_HFRXED.columns = ['Date','HF_Event']
M_HF_HFRXED = M_HF_HFRXED.dropna(axis = 0,how = 'any')

M_HF_HFRXHFC = M_HF.loc[:,['Date.61','HFRXHFIC Index']]
M_HF_HFRXHFC.columns = ['Date','HF_FI'] 
M_HF_HFRXHFC = M_HF_HFRXHFC.dropna(axis = 0,how = 'any')

M_HF_HFRXDS = M_HF.loc[:,['Date.62','HFRXDS Index']]
M_HF_HFRXDS.columns = ['Date','HF_ED_DI']
M_HF_HFRXDS = M_HF_HFRXDS.dropna(axis = 0,how = 'any')
 
M_HF_HFRXAR = M_HF.loc[:,['Date.63','HFRXAR Index']]
M_HF_HFRXAR.columns = ['Date','HF_Absolute'] 
M_HF_HFRXAR = M_HF_HFRXAR.dropna(axis = 0,how = 'any')

M_HF_HFRXRVA = M_HF.loc[:,['Date.64','HFRXRVA Index']]
M_HF_HFRXRVA.columns = ['Date','HF_Relative']
M_HF_HFRXRVA = M_HF_HFRXRVA.dropna(axis = 0,how = 'any')
 
#Creating adj. HF Data Frame
M_HF_adj = M_HF_HFRXGL
for x in (M_HF_Global,M_HF_Equity,M_HF_Macro,M_HF_Event,M_HF_FI,
          M_HF_ED_DI,M_HF_Absolute,HF_Relative):
    M_HF_adj = pd.merge(M_HF_adj,x,on = 'Date')
    
#Delete Single Hedge Fund Indices
del M_HF_HFRXGL
del M_HF_HFRXEH
del M_HF_HFRXM                           
del M_HF_HFRXED
del M_HF_HFRXHFC
del M_HF_HFRXDS
del M_HF_HFRXAR
del M_HF_HFRXRVA
########################Creating Data Frame for Cryptocurrencies#########################
M_CC = pd.DataFrame(data =     (M['Date.65'].dt.strftime('%Y-%m-%d'),M['BGCI Index'],
                                M['Date.66'].dt.strftime('%Y-%m-%d'),M['XBTUSD BGN Curncy'],
                                M['Date.67'].dt.strftime('%Y-%m-%d'),M['XETUSD BGN Curncy'],
                                M['Date.68'].dt.strftime('%Y-%m-%d'),M['XRPUSD BGN Curncy'],
                                M['Date.69'].dt.strftime('%Y-%m-%d'),M['XBNUSD BGN Curncy'],
                                M['Date.70'].dt.strftime('%Y-%m-%d'),M['XLCUSD BGN Curncy'],
                                M['Date.71'].dt.strftime('%Y-%m-%d'),M['XEOUSD BGN Curncy'],
                                M['Date.72'].dt.strftime('%Y-%m-%d'),M['XMRUSD BGN Curncy']))

M_CC = M_CC.transpose() # Transpose CC Data Frame

#Adjust data types of Data Frame
for x in M_CC.columns:
    if 'Date' in x:
        M_CC[x] = pd.to_datetime(M_CC[x],format='%Y-%m-%d')
    else:
        M_CC[x] = pd.to_numeric(M_CC[x])   
        
#Creating Data Frame for Single CC Index
M_CC_BGCI = M_CC.loc[:,['Date.65','BGCI Index']]
M_CC_BGCI.columns = ['Date','BGCI']
M_CC_BGCI = M_CC_BGCI.dropna(axis = 0,how = 'any')

M_CC_Bit = M_CC.loc[:,['Date.66','XBTUSD BGN Curncy']]
M_CC_Bit.columns = ['Date','Bitcoin']
M_CC_Bit = M_CC_Bit.dropna(axis = 0,how = 'any')

M_CC_Eth = M_CC.loc[:,['Date.67','XETUSD BGN Curncy']]
M_CC_Eth.columns = ['Date','Eth']
M_CC_Eth = M_CC_Eth.dropna(axis = 0,how = 'any')

M_CC_Ripple = M_CC.loc[:,['Date.68','XRPUSD BGN Curncy']]
M_CC_Ripple.columns = ['Date','Ripple']
M_CC_Ripple = M_CC_Ripple.dropna(axis = 0,how = 'any')

M_CC_BitC = M_CC.loc[:,['Date.69','XBNUSD BGN Curncy']]
M_CC_BitC.columns = ['Date','BitC'] 
M_CC_BitC = M_CC_BitC.dropna(axis = 0,how = 'any')

M_CC_Lite = M_CC.loc[:,['Date.70','XLCUSD BGN Curncy']]
M_CC_Lite.columns = ['Date','Lite']
M_CC_Lite = M_CC_Lite.dropna(axis = 0,how = 'any')
 
M_CC_EOS = M_CC.loc[:,['Date.71','XEOUSD BGN Curncy']]
M_CC_EOS.columns = ['Date','EOS'] 
M_CC_EOS = M_CC_EOS.dropna(axis = 0,how = 'any')

M_CC_Mon = M_CC.loc[:,['Date.72','XMRUSD BGN Curncy']]
M_CC_Mon.columns = ['Date','Mon']
M_CC_Mon = M_CC_Mon.dropna(axis = 0,how = 'any')
 
#Creating adj. CC Data Frame
M_CC_adj = M_CC_Bit
for x in (M_CC_Eth,M_CC_Ripple,M_CC_BitC,M_CC_Lite,M_CC_EOS,M_CC_Mon):
    M_CC_adj = pd.merge(M_CC_adj,x,on = 'Date')
                            
#Delete single Data Frames
del M_CC_Bit
del M_CC_Eth
del M_CC_Ripple
del M_CC_BitC
del M_CC_Lite
del M_CC_EOS
del M_CC_Mon
















































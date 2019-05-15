#import csv
import pandas as pd
#import datetime as dt
#import numpy as np
#import matplotlib.pyplot as plt

#Import Data of Assets
M = pd.read_excel('C:/Users/fjaeckle/Documents/FIM/Master-Thesis/Data/Bloomberg Data.xlsx',sheet_name = 1)

#Creating Equity Data Frame
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
    
result = pd.concat([M_Equity], join = 'inner', axis=0, join_axes=[M_Equity['Date1','Date2']])
















 #index=('Date','Equity','Eventdriven','Convertible',
                                #'Macro','Merger','Relative'))
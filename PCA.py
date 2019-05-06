#with open(working_dir + 'HFRX_Equity-Hedge-Index-Performance.csv') as csv_file:
#    HF_Equity = csv.reader(csv_file, delimiter=',')
##    quotechar, escapechar // further parameters for reader-function
#    for row in HF_Equity:
#    print {join(row)}  
#HF_Dictenory = {'date':[HF_Equity[0:4000,1]],'HF_Equity':[HF_Equity[0:4000,2]]}
#
#array = np.array([[HF_Equity['Date']], [HF_Equity['Daily ROR']], [HF_Eventdriven['Daily ROR']]])
#HF_matrix = pd.DataFrame(data = HF_Dictenory)
#HF_matrix = pd.merge(HF_Equity,
#                     HF_Eventdriven[['Daily ROR']],HF_Convertible[['Daily ROR']],
#                     on ='Date')
#####################################################################################################
import csv
import pandas as pd
import numpy as np

#Define variable for working space
working_dir = 'C:\\Users\\fjaeckle\\Documents\\FIM\\Master-Thesis\\Data\\'

#Import Data of Hedge Fund Strategies
HF_Equity = pd.read_csv(working_dir + 'HFRX_Equity-Hedge-Index-Performance.csv')
HF_Eventdriven = pd.read_csv(working_dir + 'HFRX_historical_Event-Driven-Index-Performance.csv')
HF_Convertible = pd.read_csv(working_dir + 'HFRX_historical_FI-Convertible-Arbitrage.csv')
HF_Global = pd.read_csv(working_dir + 'HFRX_historical_Global-Hedge-Fund-Index.csv')
HF_Macro = pd.read_csv(working_dir + 'HFRX_historical_Macro-CTA-Index-Performance.csv')
HF_Merger = pd.read_csv(working_dir + 'HFRX_historical_Merger-Arbitrage-Index.csv')
HF_Relative = pd.read_csv(working_dir + 'HFRX_historical_Relative-Value-Arbitrage-Index.csv')

#Transformation of return data into decimal
HF_Equity['Daily ROR']      = pd.to_numeric(HF_Equity['Daily ROR'].str.replace('%',''),errors ='coerce')
HF_Equity['Daily ROR']      = (HF_Equity['Daily ROR'])/100
HF_Eventdriven['Daily ROR'] = pd.to_numeric(HF_Eventdriven['Daily ROR'].str.replace('%',''),errors ='coerce')
HF_Eventdriven['Daily ROR'] = (HF_Eventdriven['Daily ROR'])/100
HF_Convertible['Daily ROR'] = pd.to_numeric(HF_Convertible['Daily ROR'].str.replace('%',''),errors ='coerce')
HF_Convertible['Daily ROR'] = (HF_Convertible['Daily ROR'])/100
HF_Global['Daily ROR']      = pd.to_numeric(HF_Global['Daily ROR'].str.replace('%',''),errors ='coerce')
HF_Global['Daily ROR']      = (HF_Global['Daily ROR'])/100
HF_Macro['Daily ROR']       = pd.to_numeric(HF_Macro['Daily ROR'].str.replace('%',''),errors ='coerce')
HF_Macro['Daily ROR']       = (HF_Macro['Daily ROR'])/100
HF_Merger['Daily ROR']      = pd.to_numeric(HF_Merger['Daily ROR'].str.replace('%',''),errors ='coerce')
HF_Merger['Daily ROR']      = (HF_Merger['Daily ROR'])/100
HF_Relative['Daily ROR']    = pd.to_numeric(HF_Relative['Daily ROR'].str.replace('%',''),errors ='coerce')
HF_Relative['Daily ROR']    = (HF_Relative['Daily ROR'])/100

#Set up one data frame for all strategies
HF_matrix = pd.DataFrame(data = (HF_Equity['Date'],HF_Equity['Daily ROR'],HF_Eventdriven['Daily ROR'],
                                HF_Convertible['Daily ROR'],HF_Macro['Daily ROR'],HF_Merger['Daily ROR'],
                                HF_Relative['Daily ROR']),index=('Date','Equity','Eventdriven','Convertible',
                                'Macro','Merger','Relative'))
HF_matrix = HF_matrix.transpose()

#Drop last 3 rows (includ no data)
HF_matrix = HF_matrix.drop([4000,4001,4002])
#Checking for any missing values in our data set
HF_matrix.isnull().values.any()















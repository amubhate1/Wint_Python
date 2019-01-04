# -*- coding: utf-8 -*-
"""
Created on Wed Jan 02 17:43:39 2019

@author: bhatame
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as skp
from statsmodels.tsa import seasonal as tsa
import re as regex
file_path="C:\\Users\\bhatame\\Downloads\\all\\"

train_file_name="train.csv"

train_data=pd.read_csv(file_path+train_file_name,index_col='Id')
lst=map(lambda x:(x,1.0*sum(pd.isnull(train_data[x]))/len(train_data[x])),train_data.columns.values)
null_df=pd.DataFrame(lst,columns=["Names","percentNull"])

cols_toberemoved=list(null_df[null_df['percentNull']>0.4]['Names'])

train_data.drop(cols_toberemoved,axis=1)

imputer=skp.Imputer(strategy='mean',axis=0)
imputer_fit=imputer.fit(train_data.values)
res=imputer.transform(train_data.values)
train_data.iloc[:,:]=res

res=skp.scale(train_data.values,axis=0,with_mean=True,with_std=True)
train_data.iloc[:,:]=res
column_names=train_data.columns.values


column_names_features=[i for i in column_names if(i.find('Feature_')!=-1)]
column_names_prevDailyRet=[i for i in column_names if(i.find('Ret_Minus')!=-1)]
column_names_futureDailyRet=[i for i in column_names if(i.find('Ret_Plus')!=-1)]
column_names_Weights=[i for i in column_names if(i.find('Weight_')!=-1)]
column_names_intraDayRet=[i for i in column_names if(i not in column_names_features and i not in column_names_prevDailyRet and i not in column_names_futureDailyRet and i not in column_names_Weights )]


tSeries=pd.Series(train_data.loc[10,column_names_intraDayRet])
seconds=map(lambda x:int(x[x.find('_')+1:]),tSeries.index.values)
tSeries.index=pd.to_datetime(seconds,unit='s')
#tSeries.index=pd.date_range(start='2019-01-01 00:00:00', periods=179, freq='D')
tsa.seasonal_decompose(tSeries,model='additive')
pd.datetime(2019,1,1,9,30,0)
pd.to_datetime([1, 2, 3], unit='D', origin=pd.datetime(2019,1,1,9,30,0))














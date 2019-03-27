# -*- coding: utf-8 -*-
"""
Created on Wed Jan 02 17:43:39 2019

@author: bhatame
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as skp
from statsmodels import tsa as ts
from statsmodels.graphics import tsaplots as tsp
import re as regex
from  functools import partial as partial
file_path="C:\\Users\\ameyab\\Documents\\Python Scripts\\winton stock\\all\\"

train_file_name="train.csv"

train_data=pd.read_csv(file_path+train_file_name,index_col='Id')
lst=map(lambda x:(x,1.0*sum(pd.isnull(train_data[x]))/len(train_data[x])),train_data.columns.values)
null_df=pd.DataFrame(list(lst),columns=["Names","percentNull"])


cols_toberemoved=list(null_df[null_df['percentNull']>0.4]['Names'])

train_data.drop(cols_toberemoved,axis=1)

imputer=skp.Imputer(strategy='mean',axis=0)
imputer_fit=imputer.fit(train_data.values)
res=imputer.transform(train_data.values)
train_data.iloc[:,:]=res

res=skp.scale(train_data.values,axis=0,with_mean=True,with_std=True)
train_data.iloc[:,:]=res
column_names=train_data.columns.values

train_data.to_csv(file_path+"train2.csv")
"""   end      """
column_names_features=[i for i in column_names if(i.find('Feature_')!=-1)]
column_names_prevDailyRet=[i for i in column_names if(i.find('Ret_Minus')!=-1)]
column_names_futureDailyRet=[i for i in column_names if(i.find('Ret_Plus')!=-1)]
column_names_Weights=[i for i in column_names if(i.find('Weight_')!=-1)]
column_names_intraDayRet=[i for i in column_names if(i not in column_names_features and i not in column_names_prevDailyRet and i not in column_names_futureDailyRet and i not in column_names_Weights )]


def intraday_plot(row_index):
    tSeries=pd.Series(train_data.loc[row_index,column_names_intraDayRet])
    index_series=pd.Series(tSeries.index).apply(lambda x:pd.to_datetime(str(int(x.split('_')[1])*60),unit='s',origin='unix'))
    tSeries.index=index_series
    tSeries.plot()
    tsp.plot_acf(tSeries)
    tsp.plot_pacf(tSeries)
    
def cuminntraday_plot(row_index):
    tSeries=pd.Series(train_data.loc[row_index,column_names_intraDayRet])
    intraday=tSeries.loc[column_names_intraDayRet]
    cumret=np.cumprod(intraday.apply(lambda x:1+x/100))[len(tSeries)-1]-1
    cumret=cumret*100
    daily_series=pd.Series(np.append(train_data.loc[row_index,column_names_prevDailyRet],[cumret]))    
    daily_series.plot()

    
   
intraday_plot(2)
tSeries.index=pd.to_datetime(list(seconds),unit='s')
index_series=pd.Series(tSeries.index).apply(lambda x:pd.to_datetime(str(int(x.split('_')[1])*60),unit='s',origin='unix'))
np.cumproduct([1,2,3,4])

#tSeries.index=pd.date_range(start='2019-01-01 00:00:00', periods=179, freq='D')
pd.datetime.strptime(tSeries.index.values,"Ret_%M")
pd.datetime.strptime("1",'%m')
seasonal_decompose(tSeries,model='additive')
pd.datetime(2019,1,1,9,30,0)
pd.to_datetime([1, 2, 3], unit='D', origin=pd.datetime(2019,1,1,9,30,0))
pd.datetime.minute(pd.datetime(2019,1,1,9,30,0))
pd.datetime.strptime("Ret_50","Ret_%M").hour















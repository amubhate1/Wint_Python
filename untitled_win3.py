# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from numpy import testing as numpy_testing
from keras.models import Sequential as keras_models_seq
from keras.layers import Activation as keras_layers_activation
from keras.layers import Dense as keras_layers_dense
from keras.optimizers import Adam as keras_optim_adam
from sklearn.model_selection import train_test_split as sklearn_modelselection_traintestsplit
import sklearn.preprocessing as sklearn_preprocessing_skp
file_path="C:\\Users\\ameyab\\Documents\\Python Scripts\\winton stock\\all\\"
train_file_name="train.csv"
train_data=pd.read_csv(file_path+train_file_name,index_col='Id')
lst=map(lambda x:(x,1.0*sum(pd.isnull(train_data[x]))/len(train_data[x])),train_data.columns.values)
null_df=pd.DataFrame(list(lst),columns=["Names","percentNull"])


cols_toberemoved=list(null_df[null_df['percentNull']>0.4]['Names'])

train_data.drop(cols_toberemoved,axis=1)

imputer=sklearn_preprocessing_skp.Imputer(strategy='mean',axis=0)
imputer_fit=imputer.fit(train_data.values)
res=imputer.transform(train_data.values)
train_data.iloc[:,:]=res

res=sklearn_preprocessing_skp.scale(train_data.values,axis=0,with_mean=True,with_std=True)
train_data.iloc[:,:]=res
train_data.to_csv(file_path+"train2.csv")
column_names=train_data.columns.values
column_names_features=[i for i in column_names if(i.find('Feature_')!=-1)]
column_names_prevDailyRet=[i for i in column_names if(i.find('Ret_Minus')!=-1)]
column_names_futureDailyRet=[i for i in column_names if(i.find('Ret_Plus')!=-1)]
column_names_Weights=[i for i in column_names if(i.find('Weight_')!=-1)]
column_names_intraDayRet=[i for i in column_names if(i not in column_names_features and i not in column_names_prevDailyRet and i not in column_names_futureDailyRet and i not in column_names_Weights )]
train_data_X=train_data.iloc[:][column_names_features+column_names_prevDailyRet+column_names_intraDayRet[0:119]]
train_data_Y=train_data.iloc[:][column_names_intraDayRet[120:]+column_names_futureDailyRet]

train_data_X_train,train_data_X_test,train_data_Y_train,train_data_Y_test=sklearn_modelselection_traintestsplit(train_data_X,train_data_Y,test_size=0.35, shuffle=True)


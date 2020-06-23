#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/3 下午4:59
@File   :un_seq.py
================================='''
from numpy.random import random
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import os
from model.feature_eng.feature_transform import feature_tool
from model.feature_eng.base_model import base_model
from utils.data_prepare import data_prepare

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
#----------------------------------------------------
data_folder = '../../data/'
origin_data_folder = data_folder + 'origin_data/'
submit_data_folder = data_folder + 'submit_data/'
eda_data_folder = data_folder + 'eda_data/'
fea_data_folder = data_folder + 'fea_data/'
#-----------------------------------------------------------------
model_tool = base_model(submit_data_folder)
fea_tool = feature_tool(fea_data_folder)
prepare_tool=data_prepare()
#-----------------------------------------------------------------
np.random.seed(2020)
tf.random.set_seed(2020)

train_df=pd.read_csv(origin_data_folder+'unseq_train.csv').rename(columns={'target':'label'},nrows=3)
test_df=pd.read_csv(origin_data_folder+'unseq_test.csv').rename(columns={'target':'label'},nrows=1)

sparse_fea=[str(i) for i in range(14,40)]
dense_fea=[str(i) for i in range(1,14)]
target_fea=['label']

df,(train_idx,test_idx)=prepare_tool.concat_test_train(train_df,test_df)
spareDf=df[sparse_fea]
denseDf=df[dense_fea]
targetDf=df[target_fea]

spareDf,spareInfo=prepare_tool.spare_fea_deal(spareDf)
denseDf,denseInfo=prepare_tool.dense_fea_deal(denseDf)

# train_df,test_df,y_train,y_test=prepare_tool.extract_train_test(train_idx=train_idx,test_idx=test_idx,spareDf=spareDf,denseDf=denseDf,targetDf=targetDf)
train_df,test_df,y_train,y_test=prepare_tool.extract_train_test(train_idx=train_idx,test_idx=test_idx,spareDf=spareDf,targetDf=targetDf)

#----------------------------train model--------------------------------------
from model.ctr_model.model.models import *

model=AutoInt(spareInfo=spareInfo)
print(model.summary())
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
model.fit(train_df,y_train,batch_size=64,validation_data=(test_df,y_test),epochs=100,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,verbose=5)],shuffle=False)
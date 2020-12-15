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
from kon.model.ctr_model.model.models import *

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

train_df=pd.read_csv(origin_data_folder+'unseq_train.csv',nrows=100).rename(columns={'target':'label'})
test_df=pd.read_csv(origin_data_folder+'unseq_test.csv',nrows=100).rename(columns={'target':'label'})

sparse_fea=[str(i) for i in range(14,40)]
dense_fea=[str(i) for i in range(1,14)]
target_fea=['label']

val_index=np.random.choice(train_df.index.tolist(),size=int(train_df.shape[0]*0.3))
train_index=[i for i in train_df.index.tolist()if i not in val_index]

df,(train_idx,test_idx)=prepare_tool.concat_test_train(train_df,test_df)
sparseDf=df[sparse_fea]
denseDf=df[dense_fea]
targetDf=df[target_fea]

sparseDf,sparseInfo=prepare_tool.sparse_fea_deal(sparseDf)
denseDf,denseInfo=prepare_tool.dense_fea_deal(denseDf)

train_df,test_df,y_train,y_test=prepare_tool.extract_train_test(train_idx=train_idx,test_idx=test_idx,sparseDf=sparseDf,denseDf=denseDf,targetDf=targetDf,use_softmax=True)
# train_df,test_df,y_train,y_test=prepare_tool.extract_train_test(train_idx=train_idx,test_idx=test_idx,sparseDf=sparseDf,targetDf=targetDf)
train_x,train_y,val_set=prepare_tool.split_val_set(train_df,y_train,train_index,val_index)
#----------------------------train model--------------------------------------

model=FM(prepare_tool.FeatureInput(sparseInfo=sparseInfo,denseInfo=denseInfo,useAddLinear=False,useLinear=True,useFlattenLinear=False))
print(model.summary())
model.compile(loss=tf.losses.binary_crossentropy,optimizer='adam',metrics=[tf.keras.metrics.AUC()])
model.fit(train_x,train_y,validation_data=val_set,batch_size=64,epochs=100,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,verbose=5)],shuffle=False)
model.predict(test_df)


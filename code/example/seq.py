#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/22 下午4:40
@File   :seq.py
================================='''
import pandas as pd
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
data_pre=data_prepare()
#-----------------------------------------------------------------
trainDf=pd.read_csv(origin_data_folder+'seq_train.csv')
testDf=pd.read_csv(origin_data_folder+'seq_test.csv')

spare_fea=['user_id','item_id','item_cate']
seq_fea=['buy_list']
target_fea=['target']

df,(train_idx,test_idx)=data_pre.concat_test_train(trainDf,testDf)
seqDf=df[seq_fea]
spareDf=df[spare_fea]
targetDf=df[target_fea]

seqDf,seqIdx,seqInfo=data_pre.seq_deal(
    seqDf,max_len=[90]*2,embedding_dim=[8]*2,mask_zero=True,is_trainable=True,
    pre_weight=None,sample_num=None)
spareDf,spareInfo=data_pre.spare_fea_deal(spareDf)

train_df,test_df,y_train,y_test=data_pre.extract_train_test(
    targetDf=targetDf,test_idx=test_idx,train_idx=train_idx,spareDf=spareDf,seqDf=seqDf)


from model.ctr_model.model.models import *

candidateFea=['item_id','item_cate']
behaviorFea=['buy_list','cate_list']

# model=DIEN(spareInfo=spareInfo,seqInfo=seqInfo,candidateFea=candidateFea,behaviorFea=behaviorFea)
model=SeqFM(spareInfo=spareInfo,seqInfo=seqInfo)
print(model.summary())
model.compile(loss="mean_squared_error",optimizer='adam',metrics=['accuracy'])
model.fit(train_df,y_train,validation_data=(test_df,y_test),epochs=100,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,verbose=5)])
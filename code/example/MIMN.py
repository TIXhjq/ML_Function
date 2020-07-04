#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/6/29 上午11:53
@File   :MIMN.py
@email  :hjq1922451756@gmail.com or 1922451756@qq.com  
================================='''
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, r2_score
from hyperopt import fmin, tpe, hp, partial
from numpy.random import random, shuffle
import matplotlib.pyplot as plt
from pandas import DataFrame
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import lightgbm as lgb
import networkx as nx
import pandas as pd
import numpy as np
import warnings
import cv2
import os
import gc
import re
import datetime
import sys
from model.embedding.setence_model import *
from model.feature_eng.feature_transform import feature_tool
from model.feature_eng.base_model import base_model
from model.ctr_model.model.models import *

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
data_pre=data_prepare(batch_size=10)
#-----------------------------------------------------------------
trainDf=pd.read_csv(origin_data_folder+'seq_train.csv',nrows=100)
testDf=pd.read_csv(origin_data_folder+'seq_test.csv',nrows=10)

sparse_fea=['user_id','item_id','item_cate']
seq_fea=['buy_list','cate_list']
target_fea=['target']

df,(train_idx,test_idx)=data_pre.concat_test_train(trainDf,testDf)
seqDf=df[seq_fea]
sparseDf=df[sparse_fea]
targetDf=df[target_fea]

seqDf,seqIdx,seqInfo=data_pre.seq_deal(
    seqDf,max_len=[90]*2,embedding_dim=[8]*2,mask_zero=True,is_trainable=True,
    pre_weight=None,sample_num=5)
sparseDf,sparseInfo=data_pre.sparse_fea_deal(sparseDf)

train_df,test_df,y_train,y_test=data_pre.extract_train_test(
    targetDf=targetDf,test_idx=test_idx,train_idx=train_idx,sparseDf=sparseDf,seqDf=seqDf)


from model.ctr_model.model.models import *

candidateFea=['item_id','item_cate']
behaviorFea=['buy_list','cate_list']

model=MIMN(sparseInfo=sparseInfo,seqInfo=seqInfo,candidateFea=candidateFea,behaviorFea=behaviorFea)
print(model.summary())
model.compile(loss="mean_squared_error",optimizer='adam',metrics=['accuracy'])
model.fit(train_df,y_train,validation_data=(test_df,y_test),epochs=100,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,verbose=5)])

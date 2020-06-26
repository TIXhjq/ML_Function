#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/6/24 下午1:13
@File   :timeInterval.py
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
data_pre=data_prepare()
#-----------------------------------------------------------------
trainDf=pd.read_csv(origin_data_folder+'time_inter_train.csv')
testDf=pd.read_csv(origin_data_folder+'time_inter_test.csv')

sparse_fea=['did','region','vid','cid']
seq_fea=['click_item','click_interval']
target_fea=['label']

df,(train_idx,test_idx)=data_pre.concat_test_train(trainDf,testDf)
seqDf=df[seq_fea]
sparseDf=df[sparse_fea]
targetDf=df[target_fea]

seqDf,seqIdx,seqInfo=data_pre.seq_deal(
    seqDf=seqDf,embedding_dim=[8,0],max_len=[90]*2,is_str_list=False,
    is_str=True,sample_num=5)
sparseDf,sparseInfo=data_pre.sparse_fea_deal(sparseDf)

train_df,test_df,y_train,y_test=data_pre.extract_train_test(
    targetDf=targetDf,test_idx=test_idx,train_idx=train_idx,sparseDf=sparseDf,seqDf=seqDf)

userFea=['region']
timestampFea=['click_interval']
behaviorFea=['click_item']
targetFea=['vid']

model=DTSF(sparseInfo=sparseInfo,seqInfo=seqInfo,userFea=userFea,
           timestampFea=timestampFea,behaviorFea=behaviorFea,targetFea=targetFea)
print(model.summary())
model.compile(loss="mean_squared_error",optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
model.fit(train_df,y_train,validation_data=(test_df,y_test),epochs=100,
          callbacks=[EarlyStopping(patience=10,verbose=5)])
#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/6/27 上午11:20
@File   :new_test.py
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
#-----------------------------------------------------------------
def prepare_():
    use_cols=['target','timestamp','newsid','deviceid','ts']
    df=fea_tool.reduce_mem_usage(pd.read_csv(origin_data_folder+'tuling_data/train.csv',usecols=use_cols))
    df.sort_values(['ts'],inplace=True)

    click_data=df[df.timestamp.isnull()==False]
    click_data['gsp']=click_data['timestamp']-click_data['ts']
    click_data=click_data[click_data['gsp']>0]
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=4,progress_bar=True)
    click_data=click_data[click_data['deviceid'].isnull()==False]
    del click_data['target']

    click_data=click_data.astype('str')

    need_=click_data.groupby(['deviceid'])['gsp'].apply(lambda x:','.join(x.tolist()[:-1]))
    need_=pd.concat([need_,click_data.groupby(['deviceid'])['newsid'].apply(lambda x:','.join(x.tolist()[:-1]))],axis=1)
    # need_=pd.concat([need_,click_data.groupby(['deviceid'])['timestamp'].apply(lambda x:','.join(x.tolist()[:-1]))],axis=1)
    need_.reset_index(inplace=True)

    df=df[df['deviceid'].isin(need_.deviceid.unique().tolist())]
    df.sort_values(['ts'],inplace=True)

    need=df.groupby(['deviceid'])['ts'].apply(lambda x:x.tolist()[-1])
    need=pd.concat([need,df.groupby(['deviceid'])['newsid'].apply(lambda x:x.tolist()[-1])],axis=1)
    need=pd.concat([need,df.groupby(['deviceid'])['target'].apply(lambda x:x.tolist()[-1])],axis=1)
    need.reset_index(inplace=True)

    need_.columns=['userid', 'gsp_seq', 'item_seq']
    need.columns=['userid','pos_ts', 'target_item', 'target']

    need_=need_.merge(need,how='left',on=['userid'])
    need_.to_csv(origin_data_folder+'tuling_data/format_data.csv',index=None)
    print(need_.head())

def run():
    # prepare_()
    df = pd.read_csv(origin_data_folder + 'tuling_data/format_data.csv')
    df.dropna(inplace=True, axis=0)
    df.reset_index(inplace=True, drop=True)

    list_ = list(range(df.shape[0]))
    np.random.shuffle(list_)
    split_ = int(len(list_) * 0.8)

    train_idx = list_[:split_]
    test_idx = list_[split_:]

    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    train_df.to_csv(origin_data_folder + 'gsp_train.csv', index=None)
    test_df.to_csv(origin_data_folder + 'gsp_test.csv', index=None)

if __name__=='__main__':
    t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                     [[3, 3, 3], [4, 4, 4]],
                     [[5, 5, 5], [6, 6, 6]]],dtype='float32')
    print(t)
    _t = tf.constant([[1,2],
                     [3,2],
                     [5,3]])


    slot=[tf.squeeze(i,axis=0) for i in tf.split(_t,[1]*_t.shape[0],axis=0)]
    idx_list=tf.concat([tf.expand_dims(
        tf.math.top_k(i,k=1)[1],axis=0) for idx_,i in enumerate(slot)],axis=0)


    idx_=tf.expand_dims(tf.reduce_sum(tf.one_hot(idx_list,depth=2),axis=1),axis=-1)
    print(idx_)
    print(tf.tile(idx_,[1,1,3]))
    print(t*idx_)

    print(idx_list)

#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/6/9 上午11:06
@File   :session.py
================================='''
from numpy.random import random
from kon.model.ctr_model.model.models import *
from kon.utils.data_prepare import data_prepare

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
np.random.seed(2020)
tf.random.set_seed(2020)

trainDf=pd.read_csv(origin_data_folder+'session_train.csv')
testDf=pd.read_csv(origin_data_folder+'session_test.csv')

session_maxLen=10
session_maxNum=20
sparse_fea=['region','prev','vid','cid','class_id']
dense_fea=['title_length']
seq_fea=['click_item_session']
target_fea=['label']

df,(train_idx,test_idx)=data_pre.concat_test_train(trainDf,testDf)
seqDf=df[seq_fea]
sparseDf=df[sparse_fea]
denseDf=df[dense_fea]
targetDf=df[target_fea]

seqDf,seqInfo=data_pre.sparse_wrap(seqDf,seqIdx_path=origin_data_folder+'session_seq_idx.pkl',max_len=[session_maxLen*session_maxNum]*1,embedding_dim=[8]*1)
sparseDf,sparseInfo=data_pre.sparse_fea_deal(sparseDf)
denseDf,denseInfo=data_pre.dense_fea_deal(denseDf)

train_df,test_df,y_train,y_test=data_pre.extract_train_test(
    targetDf=targetDf,test_idx=test_idx,train_idx=train_idx,sparseDf=sparseDf,seqDf=seqDf,denseDf=denseDf)

candidateFea=['vid']
behaviorFea=['click_item_session']

model=DSIN(sparseInfo=sparseInfo,seqInfo=seqInfo,candidateFea=candidateFea,behaviorFea=behaviorFea)
print(model.summary())
model.compile(loss="mean_squared_error",optimizer='adam',metrics=['accuracy'])
model.fit(train_df,y_train,validation_data=(test_df,y_test),epochs=100,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,verbose=5)])
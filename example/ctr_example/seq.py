#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/22 下午4:40
@File   :seq.py
================================='''
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
data_pre=data_prepare(batch_size=32)
#-----------------------------------------------------------------
trainDf=pd.read_csv(origin_data_folder+'seq_train.csv')
testDf=pd.read_csv(origin_data_folder+'seq_test.csv')

sparse_fea=['user_id','item_id','item_cate']
seq_fea=['buy_list','cate_list']
# seq_fea=['buy_list']
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

candidateFea=['item_id','item_cate']
behaviorFea=['buy_list','cate_list']

model=DIEN(data_pre.FeatureInput(sparseInfo=sparseInfo,seqInfo=seqInfo),candidateFea=candidateFea,behaviorFea=behaviorFea)
print(model.summary())
model.compile(loss="mean_squared_error",optimizer='adam',metrics=['accuracy'])
model.fit(train_df,y_train,validation_data=(test_df,y_test),epochs=100,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,verbose=5)])
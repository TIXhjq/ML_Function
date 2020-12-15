# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/11/22 上午10:18
@File   :sim_seq.py
@email  :hjq1922451756@gmail.com or 1922451756@qq.com  
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

df,(train_idx,test_idx)=data_pre.concat_test_train(trainDf,testDf)

reduceSeq,reduceCate=data_pre.hard_search(seqData=fea_tool.batch_convert_list(df["cate_list"]),
                                          seqCate=fea_tool.batch_convert_list(df["cate_list"]),
                                          targetCate=df["item_cate"].tolist())
df["reduce_seq"]=reduceSeq
df["reduce_seq"]=df["reduce_seq"].astype("str")
df["reduce_cate"]=reduceCate
df["reduce_cate"]=df["reduce_cate"].astype("str")

sparse_fea=['user_id','item_id','item_cate']
reduce_fea=['reduce_seq','reduce_cate']
seq_fea=["buy_list","cate_list"]+reduce_fea
target_fea=['target']

seqDf=df[seq_fea]
sparseDf=df[sparse_fea]
targetDf=df[target_fea]

seqDf,seqIdx,seqInfo=data_pre.seq_deal(
    seqDf,max_len=[90]*4,embedding_dim=[8]*4,mask_zero=True,is_trainable=True,
    pre_weight=None,sample_num=5,use_wrap=True)

sparseDf,sparseInfo=data_pre.sparse_fea_deal(sparseDf)

train,val=data_pre.extract_train_test(
    targetDf=targetDf,test_idx=test_idx,train_idx=train_idx,sparseDf=sparseDf,seqDf=seqDf)

model=SIM(data_pre.FeatureInput(sparseInfo=sparseInfo,seqInfo=seqInfo),reduceFea=reduce_fea,candidateFea=["item_id","item_cate"],behaviorFea=seq_fea)
print(model.summary())
model.compile(loss=tf.losses.binary_crossentropy,optimizer='adam',metrics=[tf.keras.metrics.AUC()])
model.fit(train,validation_data=val,epochs=100,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,verbose=5)])
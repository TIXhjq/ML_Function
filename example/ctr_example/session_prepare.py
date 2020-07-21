#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/17 下午4:27
@File   :session_prepare.py
================================='''
from pandas import DataFrame
import gc
from scipy import stats
from kon.model.ctr_model.model.models import *
from kon.utils.data_prepare import data_prepare

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
#----------------------------------------------------
data_folder = '../../data/'
origin_data_folder = data_folder + 'origin_data/mgtv_data/'
submit_data_folder = data_folder + 'submit_data/'
eda_data_folder = data_folder + 'eda_data/'
fea_data_folder = data_folder + 'fea_data/'
#-----------------------------------------------------------------
model_tool = base_model(submit_data_folder)
fea_tool = feature_tool(fea_data_folder)
data_format=data_prepare()
#-----------------------------------------------------------------
def pareper():
    context=pd.read_parquet(origin_data_folder+'context1.parquet')
    item=pd.read_parquet(origin_data_folder+'item.parquet')
    user=pd.read_csv(origin_data_folder+'user.parquet')

    user=user.merge(context,how='left',on=['did'])
    user=user.merge(item,how='left',on=['vid'])

    logs_fea=['click_item','click_time']
    user_fea=['did','region','prev']
    ad_fea=['vid','cid','class_id','title_length']
    target_fea=['label']

    use_fea=logs_fea+user_fea+ad_fea+target_fea

    user=user[use_fea]
    user.drop_duplicates(['did'],inplace=True)
    user.to_csv(origin_data_folder+'data.csv',index=None)

    df=pd.read_csv(origin_data_folder+'part_29/data.csv')
    df=pd.concat([df,pd.read_csv(origin_data_folder+'part_30/data.csv')],axis=0)
    df.to_csv(origin_data_folder+'data.csv',index=None)


def generator_session_idx(df, group_cols: list = ['did', 'click_time'], item_cols: str = 'click_item'):
    '''
    :param df:
        format:
            user_id time item
                1     1    1
    :param group_cols:
        format: list ==> [user,time]
        [groupby sign index:user_id,groupby time index:session split time]
    :param item_cols:
        item cols
    :return:
    '''

    def session_list(x):
        return len(x.tolist())

    df = df.groupby(group_cols)[item_cols].agg(session_list).reset_index().rename(
        columns={item_cols: '{}_session_idx'.format(item_cols)})

    def seq_idx(x):
        s_ = 0
        need_list = ['0']
        for i in x.tolist():
            s_ += i
            need_list.append(str(s_))
        return ','.join(need_list)

    df = df.groupby([group_cols[0]])['{}_session_idx'.format(item_cols)].agg(seq_idx).reset_index()

    return df

save_folder=data_folder + 'origin_data/'
def perpare():
    ori_df=pd.read_csv(origin_data_folder+'data.csv')
    ori_df['seq_len']=[len(str(i).split(',')) for i in ori_df['click_item'].tolist()]
    seqDf,seq_idx,seqInfo=data_format.seq_deal(seqDf=ori_df[['click_item']],embedding_dim=[8],is_str=True,is_str_list=False,use_wrap=False)
    ori_df['click_item']=[','.join([str(j) for j in i]) for i in seqDf['click_item']]
    fea_tool.pickle_op(path=save_folder+'session_seq_idx.pkl',is_save=True,file=seq_idx)

    return ori_df

import time
def get_time(timeStamp):
    timeArray = time.localtime(int(timeStamp))
    return time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

def gen_session_seq(session_maxLen,session_maxNum):
    ori_df=perpare()
    df=ori_df
    df.dropna(inplace=True)
    df['click_time']=[','.join([get_time(j) for j in i.split(',')]) for i in df['click_time'].tolist()]
    # 1h as split session
    time_list=[i.split(',')for i in df['click_time'].tolist()]
    item_list=[i.split(',')for i in df['click_item'].tolist()]
    did_list=[[i]*len(l) for i,l in zip(df['did'].tolist(),item_list)]

    df=DataFrame()
    t_list = []
    i_list = []
    d_list = []
    for t_,i_,d_ in zip(time_list,item_list,did_list):
        t_list+=t_
        i_list+=i_
        d_list+=d_
    df['click_time']=t_list
    df['click_item']=i_list
    df['did']=d_list

    df['click_time']=pd.to_datetime(df['click_time'])
    df['click_time']=df['click_time'].dt.day*100+df['click_time'].dt.hour
    df['click_item']=df['click_item'].astype('str')

    df=data_format.generator_session(df,group_cols=['did','click_time'],item_cols='click_item',session_maxLen=session_maxLen)
    df=data_format.generator_seq(df,group_cols=['did','click_time'],item_cols='click_item',session_maxNum=session_maxNum,session_maxLen=session_maxLen)

    del ori_df['click_time']
    ori_df=ori_df.merge(df,how='left',on=['did'])
    ori_df.to_csv('../../data/origin_data/data.csv',index=None)

def get_session_seq(df,item_col,max_session_length=10):
    session_seq=[
        [item_.split(',')[int(s_):int(e_)]
         for s_,e_ in zip(idx_.split(',')[:-1],idx_.split(',')[1:])]
        for item_,idx_ in zip(df[item_col].tolist(),df['{}_session_idx'.format(item_col)].tolist())]
    return [[tf.keras.preprocessing.sequence.pad_sequences(seq,maxlen=max_session_length) for seq in i]for i in session_seq]


def check_length():
    df=pd.read_csv('../../data/origin_data/data.csv')
    df['seq_len']=[len(i.split(' ')) for i in df['click_item'].tolist()]
    df['session_len_mod']=[stats.mode([len(j.split(',')) for j in i.split(' ')]) for i in df['click_item'].tolist()]
    df['session_len_mean']=[np.mean([len(j.split(',')) for j in i.split(' ')]) for i in df['click_item'].tolist()]
    df['session_len_mediandf   ']=[np.median([len(j.split(',')) for j in i.split(' ')]) for i in df['click_item'].tolist()]

    print(df.session_len_mod.value_counts())
    print(df.session_len_mean.value_counts())
    print(df.session_len_median.value_counts())

session_maxLen=10
session_maxNum=20
gen_session_seq(session_maxLen,session_maxNum)

df=pd.read_csv('../../data/origin_data/data.csv')
del df['seq_len'],df['did'],df['click_item']
gc.collect()

train_df=df.loc[:df.shape[0]*0.8]
test_df=df.loc[df.shape[0]*0.8:]

train_df.to_csv(save_folder+'session_train.csv',index=None)
test_df.to_csv(save_folder+'session_test.csv',index=None)


#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/2 下午5:14
@File   :data_prepare.py
================================='''
from sklearn.preprocessing import LabelEncoder
from numpy.random import random, shuffle
from pandas import DataFrame
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import warnings
import os
from collections import namedtuple
from kon.model.feature_eng.feature_transform import feature_tool
from kon.model.feature_eng.base_model import base_model

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

class data_prepare(object):
    def __init__(self,batch_size=None,use_shuffle=True):
        print('data prepare is backend')
        self.sparseFea=namedtuple('sparseFea',['fea_name','word_size','input_dim','cross_unit','linear_unit','pre_weight','mask_zero','is_trainable','input_length','sample_num','batch_size'])
        self.denseFea=namedtuple('denseFea',['fea_name','batch_size'])
        self.batch_size=batch_size
        self.use_shuffle=use_shuffle


    def concat_test_train(self, train_df: DataFrame, test_df: DataFrame):
        train_idx = train_df.index.tolist()
        test_idx = list(np.array(test_df.index) + train_idx[-1] + 1)
        df = pd.concat([train_df, test_df], ignore_index=True)

        return df, (train_idx, test_idx)

    def sparse_fea_deal(self,sparseDf:DataFrame,embed_dim=8,linear_dim=1,pre_weight=None):
        if not pre_weight:
            pre_weight=[None]*sparseDf.shape[1]

        sparseDf = sparseDf.fillna('-1')
        for fea in sparseDf:
            sparseDf[fea]=LabelEncoder().fit_transform(sparseDf[fea].astype('str'))

        sparseInfo=[self.sparseFea(
            fea_name=fea, input_dim=sparseDf[fea].shape[0],
            cross_unit=embed_dim, linear_unit=linear_dim,word_size=sparseDf[fea].nunique(),
            pre_weight=weight_,input_length=1,is_trainable=True,mask_zero=False,sample_num=None,
            batch_size=self.batch_size
        ) for fea,weight_ in zip(sparseDf,pre_weight)]

        return sparseDf,sparseInfo

    def single_seq_deal(self,seq_list, is_str_list=True,is_str=False,max_len=None,sample_num=None):
        '''
        :param is_str_list:
                format:"[[1,2,3],[1,2,3]]"==>True
            else:
                format:[[1,2,3],[1,2,3]]==>False
        :param is_str:
                format: ['1,2','3']
        '''
        sample_seq={}
        if is_str_list:
            seq_list = fea_tool.batch_convert_list(seq_list)
        if is_str:
            seq_list = [str(i).split(',') for i in seq_list]

        w2str = [[str(j) for j in i] for i in seq_list]
        seq = [" ".join(i) for i in w2str]

        token = tf.keras.preprocessing.text.Tokenizer(lower=False, char_level=False, split=' ')
        token.fit_on_texts(seq)
        format_seq = token.texts_to_sequences(seq)
        format_seq = tf.keras.preprocessing.sequence.pad_sequences(format_seq, maxlen=max_len,value=0)
        seq_idx = token.word_index

        # if sample_num:
        #     sample_seq=[[[label]+list(np.random.choice([i for i in seq if i!=label and i!=0],size=sample_num)) if label!=0 else []
        #             for label in seq]for seq in format_seq]

        return (format_seq, seq_idx,sample_seq)

    def seq_deal(self,seqDf,embedding_dim:list,max_len:list=None,is_str_list=True,is_str=False,mask_zero=True,is_trainable=True,pre_weight:list=None,sample_num=None,use_wrap=True):
        '''
        notice:
                <1> seqDf:
                    format===>single_seq_deal
                <2> pre_weight:
                    format===>[[fea1_weight],[fea2_weight]...]
                <3> sample_num:
                    notice:
                        negative must make sure len(seq)>=negative_num+1
                    e.g
                        negative_num:5==>get(5*negative sample)<DIEN>

        :param max_len: seq max length
        :param embedding_dim: seq embed dim
        :param is_str_list&is_str==> single_seq_deal introduce
        :param mask_zero: use mask==True
        :param is_trainable: use embed trainable==True
        :param pre_weight: embedding pre-train(e.g w2c as backend)
        :param use_wrap: use sparseFea wrap==True
        :return:seqDf,seqIdx,seqInfo
        '''
        if not pre_weight:
            pre_weight=[None]*seqDf.shape[1]
        if not max_len:
            max_len=[None]*seqDf.shape[1]
        sample_seq=None
        seq_tuple={
            seq_fea:self.single_seq_deal(seqDf[seq_fea],is_str_list=is_str_list,is_str=is_str,max_len=len_,sample_num=sample_num)
                   for seq_fea,len_ in zip(seqDf,max_len)}
        seqDf={key:seq_tuple[key][0] for key in seq_tuple}
        seqIdx = {key: seq_tuple[key][1] for key in seq_tuple}
        if sample_num:
            sample_seq={key:[i[1:] for i in seq_tuple[key][2]] for key in seq_tuple}
        del seq_tuple

        seqInfo=None
        if use_wrap:
            seqDf,seqInfo=self.sparse_wrap(seqDf,seqIdx=seqIdx,embedding_dim=embedding_dim,max_len=max_len,mask_zero=mask_zero,is_trainable=is_trainable,pre_weight=pre_weight,sample_num=sample_num)

        return seqDf,seqIdx,seqInfo

    def sparse_wrap(self,seqDf,embedding_dim:list,seqIdx=None,seqIdx_path=None,max_len:list=None,mask_zero=True,is_trainable=True,pre_weight:list=None,sample_num=None):
        if not pre_weight:
            pre_weight=[None]*seqDf.shape[1]
        if not max_len:
            max_len=[None]*seqDf.shape[1]
        if seqIdx_path:
            seqIdx = fea_tool.pickle_op(seqIdx_path, is_save=False)

        seqInfo = [self.sparseFea(
            fea_name=seq_fea, word_size=len(seqIdx[seq_key].keys()) + 1, input_dim=seqDf[seq_fea].shape[0],
            cross_unit=embed_, linear_unit=1, pre_weight=weight_, mask_zero=mask_zero,
            is_trainable=is_trainable, input_length=max_, sample_num=sample_num,batch_size=self.batch_size
        ) for seq_fea, seq_key, weight_, max_, embed_ in zip(seqDf, seqIdx, pre_weight, max_len, embedding_dim)]

        if not isinstance(seqDf,dict):
            seqDf={fea:np.array([[int(j) for j in i.split(',')]for i in seqDf[fea].values]) for fea in seqDf}

        return seqDf,seqInfo

    def generator_session(self,df, group_cols: list, item_cols: str,
                          session_maxLen, use_check=False):
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
        :param use_check:
            print=>session size distribute,can to find session_maxLen
        :return:
            DataFrame==> columns=user_id,session_list
        '''

        if use_check:
            def need_(x):
                return len(x.tolist())

            print(df.groupby(group_cols)[item_cols].agg(need_).reset_index()[item_cols].value_counts().head(20))
        else:
            def session_seq(x):
                return ','.join(x.tolist())

            df = df.groupby(group_cols)[item_cols].agg(session_seq).reset_index().rename(
                columns={item_cols: '{}_session'.format(item_cols)})
            df['{}_session'.format(item_cols)] = [','.join([str(j) for j in i]) for i in
                                                  tf.keras.preprocessing.sequence.pad_sequences(
                                                      [i.split(',') for i in df['{}_session'.format(item_cols)]],
                                                      maxlen=session_maxLen)]
            del df[group_cols[1]]

            return df

    def generator_seq(self,df, group_cols, item_cols, session_maxLen, session_maxNum, use_check=False):
        '''
        :param df:df['user_id','{}_session']
        :param group_cols: same to generator_session
        :param item_cols: same to generator_session
        :param session_maxLen: same to generator_session
        :param session_maxNum: same to generator_session
        :param use_check: same to generator_session
        :return: df==> <user_id,seq[concat session]>
        '''
        if use_check:
            def need_(x):
                return len(x.tolist())

            print(df.groupby([group_cols[0]])['{}_session'.format(item_cols)].agg(
                need_).reset_index().click_item_session.value_counts().head(20))
        else:
            def seq(x):
                use_list = x.tolist()
                if len(use_list) > session_maxNum:
                    use_list = use_list[:session_maxNum]
                else:
                    use_list += [','.join([str(i) for i in [0] * session_maxLen]) for i in
                                 range(session_maxNum - len(use_list))]

                need_list = ""
                for i in use_list:
                    need_list += i + ','
                return need_list[:-1]

            df = df.groupby([group_cols[0]])['{}_session'.format(item_cols)].agg(seq).reset_index()

            return df

    def sparse_prepare(self, sparse_info: list):
        return [tf.keras.Input(batch_shape=(info_.batch_size,info_.input_length,),
                               name=info_.fea_name) for info_ in sparse_info]

    def dense_fea_deal(self,denseDf:DataFrame,is_fillna=True):
        if is_fillna:
            denseDf = DataFrame({fea: denseDf[fea].fillna(denseDf[fea].mode()[0]) for fea in denseDf})
        [denseDf[fea].fillna(denseDf[fea].mode()[0])for fea in denseDf]
        denseDf[denseDf.columns.tolist()]=MinMaxScaler(feature_range=(0,1)).fit_transform(denseDf)
        denseInfo=[self.denseFea(fea,self.batch_size) for fea in denseDf]

        return denseDf,denseInfo

    def dense_prepare(self,dense_info:list):
        return [tf.keras.Input(batch_shape=(info_.batch_size,1,), name=info_.fea_name)for info_ in dense_info]

    def df_format(self,df:DataFrame):
        df_={}
        for fea in df:
            df_.update({fea:df[fea].values})
        return df_

    def df_format_input(self,df:list):
        df=pd.concat(df,axis=1)
        df=self.df_format(df)
        return df

    def df_prepare(self,sparseInfo:list=None,denseInfo:list=None,seqInfo:list=None):
        df_name=[]
        inputs=[[],[],[]]
        if denseInfo:
            dense_inputs=self.dense_prepare(denseInfo)
            df_name+=[info_.fea_name for info_ in denseInfo]
            inputs[0]=dense_inputs
        if sparseInfo:
            sparse_inputs=self.sparse_prepare(sparseInfo)
            df_name+=[info_.fea_name for info_ in sparseInfo]
            inputs[1]=sparse_inputs
        if seqInfo:
            seq_inputs=self.sparse_prepare(seqInfo)
            df_name+=[info_.fea_name for info_ in seqInfo]
            inputs[2]=seq_inputs

        return inputs


    def extract_train_test(self,train_idx, test_idx,targetDf,sparseDf=None, denseDf=None,seqDf=None,use_softmax=True):
        try:
            train_dense = denseDf.loc[train_idx]
            test_dense = denseDf.loc[test_idx]
        except AttributeError:
            train_dense,test_dense=None,None

        try:
            train_sparse = sparseDf.loc[train_idx]
            test_sparse = sparseDf.loc[test_idx]
        except AttributeError:
            train_sparse, test_sparse = None, None

        try:
            train_seq={key:seqDf[key][train_idx] for key in seqDf}
            test_seq={key:seqDf[key][test_idx] for key in seqDf}
        except TypeError:
            train_seq,test_seq= {}, {}

        y_train=targetDf.loc[train_idx]
        y_test=targetDf.loc[test_idx]
        if use_softmax:
            y_train=tf.keras.utils.to_categorical(y_train.values.tolist())
        else:
            y_train=y_train.values

        train_df=self.df_format_input([train_dense,train_sparse])
        test_df=self.df_format_input([test_dense,test_sparse])
        train_df.update(train_seq)
        test_df.update(test_seq)

        if self.batch_size!=None:
            train_df=self.static_batch(train_df)
            test_df=self.static_batch(test_df)
            y_train=self.static_batch(y_train)
            y_test=self.static_batch(y_test)

        return train_df,test_df,y_train,y_test

    def input_loc(self,df,use_idx:list):
        '''
        :param df: format df
        :param use_idx: use idx,e.g k-flods
        :return: df[use idx]
        '''
        if isinstance(df, dict):
            return {key: np.array(df[key])[use_idx] for key in df}
        else:
            return df[use_idx]

    def static_batch(self,df):
        if isinstance(df,dict):
            df_num=np.array(df[list(df.keys())[0]]).shape[0]
        else:
            df_num=len(df)

        batch_num = (df_num // self.batch_size) * self.batch_size
        need_idx = np.random.choice(list(range(df_num)), size=batch_num)
        if self.use_shuffle:
            shuffle(need_idx)
        df = self.input_loc(df, use_idx=need_idx)

        return df

    def split_val_set(self,train_df,y_train,train_index,val_index):
        train_x = self.input_loc(df=train_df, use_idx=train_index)
        train_y = self.input_loc(df=y_train, use_idx=train_index)
        val_x = self.input_loc(df=train_df, use_idx=val_index)
        val_y = self.input_loc(df=y_train, use_idx=val_index)

        return train_x,train_y,(val_x,val_y)
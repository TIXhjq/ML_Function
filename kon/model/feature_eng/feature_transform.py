# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-11-15 上午11:59
================================='''
import logging
from pandarallel import pandarallel
import gensim
import matplotlib.pyplot as plt
from scipy.stats import stats
from gensim.models import Word2Vec
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from kon.model.embedding.util.util_tool import save_edgelist
import re
from kon.model.embedding.model_test import model_test
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
# from PIL import Image
import pandas as pd
import numpy as np
import pickle
import time
import warnings
import datetime
# import cv2
import os
import gc
import seaborn as sns
from multiprocessing import Pool

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())

#----------------------------------------------------
from kon.model.feature_eng.base_model import base_model
data_folder='../../data/'
origin_data_folder=data_folder+'origin_data/'
submit_data_folder=data_folder+'submit/'
eda_data_folder=data_folder+'eda_data/'
result_fea_folder=submit_data_folder+'use_feature/'
#----------------------------------------------------
pandarallel.initialize()

class feature_tool(object):
    def __init__(self,save_folder:str):
        print('feature tool is backend')
        self.model = base_model(save_folder=submit_data_folder)
        self.save_folder=save_folder
        self.embedding_folder=save_folder+'embedding_data/'

    def gen_time_interval(self, df, time_col,padding_init=True):
        padding_ = []
        if padding_init:
            padding_=[0]

        return [','.join(np.array(padding_ + np.diff(np.array(i.split(',')).astype('int')).tolist()).astype('str'))
                for i in df[time_col].tolist()]

    def pickle_op(self,path,is_save,file=None):
        if is_save:
            with open(path, "wb") as fp:
                pickle.dump(file, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "rb") as fp:
                file = pickle.load(fp)

        return file

    def auto_generator_dir(self,save_folder: str):
        if os.path.exists(save_folder):
            os.system('rm -rf {}'.format(save_folder))
        os.system('mkdir {}'.format(save_folder))

    def check_train_test_columns(self,train_data:DataFrame,test_data:DataFrame,target_col:str):
        '''
        find train kon different columns
        '''
        train_cols = train_data.columns.tolist()
        train_cols.remove(target_col)
        test_cols = test_data.columns.tolist()

        for col in test_cols:
            train_cols.remove(col)
        if train_cols == []:
            print('same columns')
        else:
            print(train_cols)

    def timestamp_to_date(self,df:DataFrame,time_col:str,is_save_origin=True):
        time_stamp = []
        for time_ in tqdm(df[time_col].tolist(), desc='timestamp'):
            try:
                time_ = datetime.datetime.fromtimestamp(time_ / 1000)
            except ValueError:
                time_stamp.append(time_)
            else:
                time_stamp.append(time_)
        if is_save_origin:
            df.rename(columns={time_col:'origin_'+time_col},inplace=True)
        df[time_col]=time_stamp

        return df

    def ctr_eda(self,train_data: DataFrame, test_data: DataFrame, use_col: str, target_col: str):
        '''
        ------->AD data eda<-------
        :param train_data: train_df
        :param test_data:  test_df
        :param use_col:    user_col
        :param target_col: target_col
        :return:
        '''

        # 每天每小时的转换率
        by_time = DataFrame(train_data.groupby(['day', 'hour'])[target_col].aggregate('mean')).reset_index()
        print(by_time.head())

        by_time = by_time.pivot('day', 'hour', target_col)
        plt.figure(figsize=(12, 6))
        sns.heatmap(by_time)
        plt.title('hour of day-cvr')
        plt.show()

        # 相同hour相同的时间片
        by_time = DataFrame(train_data.groupby(['day', 'hour'])[target_col].aggregate('mean')).reset_index()
        hour_list = train_data.hour.unique().tolist()
        print(hour_list)
        by_hour = by_time.groupby(['hour'])

        i = 1
        plt.figure(figsize=(8, 4))
        for hour_ in tqdm(np.sort(hour_list), desc='on_hour -day cvr'):
            by_hour_data = DataFrame(by_hour.get_group(hour_)).sort_values(['day'])

            x = by_hour_data['day'].tolist()
            y = by_hour_data[target_col].tolist()
            plt.subplot(6, 4, i)
            plt.bar(x, y)
            plt.ylabel('hour:' + str(hour_))
            i += 1
        plt.xlabel('hour of same day')
        plt.show()

        # 随天数的CVR
        by_day = DataFrame(train_data.groupby(['day'])[target_col].mean()).reset_index()
        x = by_day.day.tolist()
        y = by_day[target_col].tolist()

        plt.figure(figsize=(12, 6))
        plt.bar(x, y)
        plt.xlabel('day')
        plt.ylabel('cvr')
        plt.title('day-cvr')
        plt.show()

        # 随天数数据量统计
        by_time_count = DataFrame(train_data.groupby(['day'])[target_col].count()).reset_index().sort_values(
            ['day'])

        x = by_time_count.day.tolist()
        y = by_time_count[target_col].tolist()
        plt.figure(figsize=(12, 6))
        plt.bar(x, y)
        plt.xlabel('day_time')
        plt.ylabel('data_num')
        plt.title('day_DataCount')
        plt.show()

        # 用户总查询次数
        by_guid_count = DataFrame(train_data.groupby([use_col])[target_col].count()).reset_index().sort_values(
            [target_col], ascending=False)

        x = range(len(by_guid_count[use_col].tolist()))
        y = by_guid_count[target_col].tolist()
        plt.figure(figsize=(12, 6))
        plt.plot(x, y)
        plt.xlabel('user')
        plt.ylabel('search_num')
        plt.title('user-search')
        plt.show()

        # 用户总下载次数
        by_guid_download = DataFrame(train_data[train_data[target_col] == 1].groupby([use_col])[
                                         target_col].count()).reset_index().sort_values([target_col],
                                                                                        ascending=False)
        x = range(len(by_guid_download[use_col].tolist()))
        y = by_guid_download[target_col].tolist()
        plt.figure(figsize=(12, 6))
        plt.plot(x, y)
        plt.xlabel('user')
        plt.ylabel('download_num')
        plt.title('user-download')
        plt.show()

        # 用户的CVR----长尾,用户搜索次数与用户CRV与用户实际购买之间的关系
        by_guid_cvr = DataFrame(train_data.groupby([use_col])[target_col].mean()).reset_index().sort_values(
            [target_col], ascending=False)
        by_guid_cvr = by_guid_cvr.merge(by_guid_count, 'left', on=[use_col])
        by_guid_cvr = by_guid_cvr.merge(by_guid_download, 'left', on=[use_col])
        print(by_guid_cvr.head())
        by_guid_cvr = by_guid_cvr.fillna(0)

        x = range(len(by_guid_cvr[use_col].tolist()))
        y_cvr = np.array(by_guid_cvr[target_col+'_x'].tolist()) * 100000
        y_count = by_guid_cvr[target_col+'_y'].tolist()
        y_download = by_guid_cvr[target_col] * 10

        plt.figure(figsize=(12, 6))
        plt.plot(x, y_count, color='black')
        plt.plot(x, y_cvr, color='blue')
        plt.plot(x, y_download, color='yellow')
        plt.legend(['search_num', 'cvr', 'download_num'])
        plt.xlabel('user')
        plt.show()

        # train和test的用户重叠度
        test_data_guid = test_data[use_col].unique().tolist()
        test_data = DataFrame(columns=[use_col], data=test_data_guid)
        test_data = test_data.merge(by_guid_cvr, how='left', on=[use_col])

        all = test_data[use_col].count()
        old = (test_data[target_col + '_x'].count()) / all
        new = (all - old) / all

        plt.figure(figsize=(12, 6))
        plt.pie([old, new], colors=['blue', 'yellow'], labels=['old_user', 'new_user'], labeldistance=1.1,
                autopct='%2.0f%%', shadow=False,
                startangle=90, pctdistance=0.6)
        plt.show()

    def null_feature(self,df,fea_cols): #row and col
        '''
        cal null feature [is_null(axis=0),null_count(axis=0),num_per(axis=0/1)]
        '''
        #col is null
        is_null_fea=[]
        for col in fea_cols:
            is_null_col=col+'_is_null'
            df[is_null_col]=df[col].isnull().astype(int)
            is_null_fea.append(is_null_col)

        #row null count
        count_data=DataFrame(df[fea_cols].count(axis=1)).reset_index()
        df['num_null']=([len(fea_cols)]*count_data.shape[0])-(count_data[0])
        # df['per_num_null']=df['num_null']/(df['num_null'].sum())

        # null_fea=is_null_fea+['num_null','per_num_null']
        null_fea=is_null_fea+['num_null']
        return df,null_fea

    def format_groupby_list(self,df:DataFrame,groupby_fea:list,is_nature_code=True):
        '''
        :param df: df
        :param groupby_fea: by_fea_list
        :return: df[cross_fea],cross_fea
        '''
        if len(groupby_fea)>1:
            max_cross_col='_'.join(groupby_fea)
            df[max_cross_col]=''
            for col in tqdm(groupby_fea):
                df[max_cross_col]+=df[col].astype('str')
                df[max_cross_col]+='_'
            if is_nature_code:
                df[max_cross_col]=LabelEncoder().fit_transform(df[max_cross_col])
        else:
            max_cross_col=groupby_fea[0]

        return df,max_cross_col


    def cal_cross_fea(self,df:DataFrame,first_fea:list,second_fea:list=None,third_fea:list=None,cross_rank=2,is_nature_code=True,is_str_cross=True):
        '''
            <first_fea>inner product<second_fea>
            param: 2nd-cross,3rd-cross
        '''
        cross_fea=[]
        if cross_rank==2:
            for f_1 in tqdm(first_fea,desc='rank_'+str(cross_rank)+' cross fea'):
                for f_2 in second_fea:
                    if f_1!=f_2:
                        cross_col=str(f_1)+'_cross_'+str(f_2)
                        if is_str_cross:
                            df[cross_col]=(df[f_1].astype('str')+"_"+df[f_2].astype('str')).astype('category')
                        else:
                            df[cross_col] = [np.array([fea_1,fea_2]) for fea_1,fea_2 in (zip(df[f_1].tolist(),df[f_2].tolist()))]
                        if is_nature_code:
                            df[cross_col] = LabelEncoder().fit_transform(df[cross_col])
                            df[cross_col] = df[cross_col].astype('category')
                        cross_fea.append(cross_col)
        else:
            for f_1 in tqdm(first_fea,desc='rank_'+str(cross_rank)+' cross fea'):
                for f_2 in second_fea:
                    if f_1!=f_2:
                        for f_3 in third_fea:
                            if f_1!=f_3:
                                if f_2!=f_3:
                                    cross_col = str(f_1) + '_cross_' + str(f_2)+'_cross_'+str(f_3)
                                    df[cross_col] = (df[f_1].astype('str') + "_" + df[f_2].astype('str')+"_"+df[f_3].astype('str')).astype('category')
                                    if is_nature_code:
                                        df[cross_col]=LabelEncoder().fit_transform(df[cross_col])
                                    df[cross_col] = df[cross_col].astype('category')
                                    cross_fea.append(cross_col)
        return df, cross_fea

    def count_col(self,df:DataFrame,cate_fea:list,by_cols:list=None):
        '''
        cate_fea count feature
        '''
        count_fea_list=[]
        # for fea in tqdm(cate_fea,desc='cate fea count'):
        for fea in cate_fea:
            if by_cols==None:
                count_fea=fea+'_count'
                count_data=DataFrame(df[fea].value_counts()).reset_index().rename(columns={fea:count_fea,'index':fea})
                merge_cols=[fea].copy()
            else:
                count_fea=fea+'_'+'by_'+'_'.join(by_cols)+'_count'
                count_data=DataFrame(df.groupby(by_cols)[fea].value_counts()).rename(columns={fea:count_fea}).reset_index()
                merge_cols=by_cols+[fea]

            count_fea_list.append(count_fea)
            df = df.merge(count_data,how='left',on=merge_cols)

        return df,count_fea_list

    def fun(self,df, stat_agg,by_fea,stat_, stat_name):
        return df.groupby(by_fea)[stat_].agg(
            {stat_name + '_' + stat_ + '_' + agg_: agg_ for agg_ in stat_agg})

    def bach_stat_fea(self,df:DataFrame,by_fea,aim_fea,stat_agg,stat_name,is_merge=True):
        stat_df=[]

        p=Pool()

        for stat_ in tqdm(aim_fea,desc='stat_fea'):
            # print(self.fun(df,stat_agg,by_fea,stat_,stat_name))
            stat_df.append(p.apply_async(self.fun,(df,stat_agg,by_fea,stat_,stat_name)))
        p.close()
        p.join()
        if is_merge:
            return df.merge(pd.concat([i.get(timeout=1) for i in stat_df],axis=1))
        else:
            return pd.concat([i.get(timeout=1) for i in stat_df],axis=1)


    def stat_fea(self,df:DataFrame,cate_fea_list:list,num_fea_list:list,data_sign:str='',agg_param=['mean','sum','std'],is_format_cate_input=False,is_save_df=True):
        '''
        :param cate_fea_list:   input_format=[[],[],[]]
        :param data_sign:       give fea data sign,default=''
        '''
        if is_format_cate_input:
            cate_fea_list = [[col] for col in cate_fea_list]
        cate_len=len(cate_fea_list)
        stat_fea_list=[]

        for cate_fea in tqdm(cate_fea_list,desc='by cate stat'):
            cate_len-=1
            by_agg_data=DataFrame(df.groupby(cate_fea)[num_fea_list].agg(agg_param)).reset_index()
            for num_fea in tqdm(num_fea_list,desc='_'.join(cate_fea)+'_stat_num_fea'+' rest:'+str(cate_len)):
                agg_cols=[data_sign+'_by_'+'_'.join(cate_fea)+'_on_'+num_fea+'_'+agg_operator for agg_operator in agg_param]
                agg_data_=by_agg_data[num_fea]
                agg_=DataFrame(data=agg_data_.values,columns=agg_cols)
                agg_[cate_fea]=by_agg_data[cate_fea]
                if is_save_df:
                    df=df.merge(agg_,'left',on=cate_fea)
                else:
                    df=agg_
                stat_fea_list+=agg_cols
        return df,stat_fea_list

    def filter_feature(self,df:DataFrame,fea_cols,label,cate_fea,file_name):
        model_important,zero_fea=self.model.fit_transform(train_data=df,is_pred=False,cate_cols=cate_fea,label_col=label,use_cols=fea_cols)
        model_important.to_csv(self.save_folder+file_name+'_feature_important.csv',index=None)
        np.save(self.save_folder+file_name+'_zero_import.csv',zero_fea)

    def test_distribute(self,train:DataFrame,test:DataFrame,cate_fea:list=None):
        '''
        kon [train-kon distribute]
        final auc <0.65 ==> practically same
        '''

        train['is_train']=1
        test['is_train']=0
        df=pd.concat([train,test],ignore_index=True)
        cols=df.columns.tolist()
        cols.remove('is_train')

        self.model.fit_transform(train_data=df, use_cols=cols, cate_cols=cate_fea,label_col='is_train',is_pred=False)

    def reduce_mem_usage(self,df:DataFrame,df_save_name=None,is_save=False,verbose=True):
        '''
        reduce df need memory
        '''
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in tqdm(df.columns,desc='reduce mem'):
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
                end_mem, 100 * (start_mem - end_mem) / start_mem))
        if is_save:
            df.to_csv(self.save_folder+df_save_name+'.csv',index=None)

        return df

    def stat_discover_fea(self,df: DataFrame, need_stat_fea: str, stat_fea_target: str, by_fea: str):
        '''
        AD-Q[the item is clicked  by user]
        '''
        by_guid = df.groupby([by_fea])
        guid_of_day_history_click = []

        for id in tqdm(df[by_fea].unique().tolist(), desc='jugle is discover'):
            guid_data = DataFrame(by_guid.get_group(id)).sort_values('ts')
            history_click = guid_data.loc[guid_data[stat_fea_target] == 1][[by_fea, need_stat_fea]]
            history_click, fea = self.count_col(df=history_click, cate_fea=[need_stat_fea])
            history_click['is_discover'] = 1
            guid_of_day_history_click.append(history_click)

        history_click = pd.concat(guid_of_day_history_click)

        del guid_of_day_history_click
        gc.collect()

        return history_click

    def by_stat_discover(self,df: DataFrame, by_fea: str, by_fea_type: list, aim_fea: list, need_stat_fea: str,
                         stat_by_fea: str, stat_fea_target: str):

        all_stat_discover_data = [df[df[by_fea].isin(by_fea_type[0])]]
        for fea_, aim_fea_ in tqdm(zip(by_fea_type, aim_fea)):
            print('***********************')
            fea_data = df[df[by_fea].isin(fea_)]
            aim_data = df[df[by_fea].isin(aim_fea_)]
            print(fea_data.head())
            print(fea_)
            history_click = self.stat_discover_fea(fea_data, need_stat_fea=need_stat_fea, stat_fea_target=stat_fea_target,
                                              by_fea=stat_by_fea)

            del fea_data
            gc.collect()

            aim_data[need_stat_fea + '_count'] = 0
            aim_data['is_discover'] = 0
            aim_data.merge(history_click, 'left', on=[stat_by_fea, need_stat_fea])
            all_stat_discover_data.append(aim_data)
            print(aim_data.head())
        all_stat_discover_data = pd.concat(all_stat_discover_data)

        return all_stat_discover_data

    def extract_train_test_data(self,df: DataFrame, target_col: str):
        '''
        :param df: df(train+kon)
        :param target_col: (target_col)
        :return: train_data,test_data
        '''
        df['not_train'] = df[target_col].isnull()
        train_data = df[df['not_train'] == False]
        test_data = df[df['not_train'] == True]

        del df['not_train']
        gc.collect()

        return train_data, test_data

    def combine_feature(self,combine_fea_list,is_in_bit=False):
        '''
        :param combine_fea_list: format=[fist_col:list,second_col:list,...]
        :return:[[first_col[0]+second_col],first_col[1]+second_col,...]
        '''
        combine_fea_len=len(combine_fea_list)
        combine_fea = []
        if not is_in_bit:
            for f_cate in combine_fea_list[0]:
                fea_=[]
                for i in range(1,combine_fea_len):
                    fea_=[f_cate]+combine_fea_list[i]
                combine_fea.append(fea_)

        return combine_fea

    def list_to_seq(self,item_list: list):
        '''
        :param item_list: user click item seq[sort by time...]
        :return: DI-edges_list
        '''
        item_seq_ = []
        start_node = item_list[0]
        for item in item_list[1:]:
            item_seq_.append(str(start_node) + ' ' + str(item) + '\n')
            start_node = item
        return item_seq_

    def generator_user_seq(self,df: DataFrame, user_col: str,
                           item_edgelist_path: str = 'item_edgelist.txt'):
        '''
        :param df: df
        :param user_col:  user_col
        :param item_edgelist_path: user seq transform [item is clicked] graph edge_list
        :return:
        '''
        item_edgelist_path=self.save_folder+item_edgelist_path
        user_item_edgelist = []

        by_user = df.groupby(['deviceid'])
        for user_ in df[user_col].unique().tolist():
            user_data = DataFrame(by_user.get_group(user_)).sort_values(['timestamp'])
            if user_data.shape[0] != 1:
                user_item_edgelist.append(self.list_to_seq(user_data['newsid'].tolist()))

        save_edgelist(edgelist_list=user_item_edgelist, save_path=item_edgelist_path)

        return user_item_edgelist

    def extract_embedding_df(self,item_embedding):
        '''
        :param item_embedding: word2vec output[format(dict{key:value})]
        :return: len(cols)=dimension-->df
        '''
        item = list(item_embedding.keys())
        embedding = list(item_embedding.values())

        embedding_cols = ['item_embedding_{}'.format(i) for i in range(len(embedding[0]))]
        item_embedding_df = DataFrame(columns=embedding_cols, data=embedding)
        item_embedding_df['item'] = item

        return item_embedding_df

    def embed_proc(self,df,format_user_col,format_time_col, time_, idx_,embedding_save_folder,item_edgelist_path,item_save_name):
        t=time.time()
        print('{} proc is running,as start time:{}'.format(idx_,t))
        by_time_data = df[df[format_time_col] == time_].sort_values([format_time_col])
        user_item_seq = self.generator_user_seq(df=by_time_data, user_col=format_user_col)
        item_embedding = model_test(build_name='deepwalk', edgelist_path=self.save_folder + item_edgelist_path)
        print('generator item embedding time:{}'.format(time.time() - t))
        item_df = self.extract_embedding_df(item_embedding)
        item_df['by_{}_dimension_on_{}'.format(format_time_col, format_user_col)] = time_
        del user_item_seq
        item_df.to_csv(embedding_save_folder + '/{}_item_embedding_df_idx_{}.csv'.format(item_save_name, str(idx_)),
                       index=None)
        return item_df

    def generator_item_embedding(self,df: DataFrame, user_col: list, time_col: list,timeline:str='timestamp',
                                 item_col:str='newsid',
                                item_edgelist_path='item_edgelist.txt', is_save=False,
                                item_save_name: str = None,cpu:int=4):
        '''
        :param df: click_data
        :param by_user_col: user_col=[by [user] to build item click seq]
        :param by_time_col: session_scale_fea[may be-->(hour,day,month,...)]
        :param item_edgelist_path:
        :param is_save:
        :param item_save_name: item_df save_name[save_name]
        :return: item_df
        '''

        df=df[user_col+time_col+[timeline,item_col]]
        print(df.head())
        df,format_time_col=self.format_groupby_list(df=df,groupby_fea=time_col,is_nature_code=True)
        df,format_user_col=self.format_groupby_list(df=df,groupby_fea=user_col,is_nature_code=True)
        embedding_save_folder=self.embedding_folder+'by_{}_dim_on_{}'.format(format_time_col,format_user_col)

        self.auto_generator_dir(embedding_save_folder)
        item_emb_df = []

        p = Pool(cpu)
        for idx_,time_ in tqdm(enumerate(df[format_time_col].unique().tolist()), desc='generator seq on time'):
            item_emb_df.append(p.apply_async(self.embed_proc,args=(df,format_user_col,format_time_col, time_, idx_,embedding_save_folder,item_edgelist_path,item_save_name,)))

        p.close()
        p.join()
        print('all proc is over')

        item_emb_df = pd.concat(item_emb_df)
        item_emb_df.to_csv(embedding_save_folder + '/{}_item_embedding_df_all.csv'.format(item_save_name), index=None)

        return df,item_emb_df

    def read_csv(self,file_dir,nrows=None):
        if nrows!=None:
            df=self.reduce_mem_usage(pd.read_csv(file_dir,nrows=nrows))
        else:
            df = self.reduce_mem_usage(pd.read_csv(file_dir))
        return df

    def strList_2_list(self,a):
        a = a[1:-1]
        a = re.sub(pattern=', ', repl=' ', string=a)
        a = a.replace('[', '')
        a = a.split(']')
        a = [i.split(' ') for i in a]

        list_ = []
        for i in a:
            temp_ = []
            for j in i:
                try:
                    temp_.append(int(j))
                except ValueError:
                    continue
            if temp_:
                list_.append(temp_)
        return list_

    def batch_convert_list(self,a):
        list_ = []
        for i in tqdm(a, desc='convert list'):
            list_+=self.strList_2_list(i)

        return list_


    def trian_save_word2vec(self,docs, embed_size=300, save_name='w2v.txt', split_char=' '):
        '''
        输入
        docs:输入的文本列表
        embed_size:embed长度
        save_name:保存的word2vec位置

        输出
        w2v:返回的模型
        '''
        input_docs = []
        for i in docs:
            input_docs.append(i.split(split_char))
        logging.basicConfig(
            format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
        w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=8, seed=1017, workers=24, min_count=1, iter=10)
        w2v.wv.save_word2vec_format(save_name)
        print("w2v model done")
        return w2v

    # 得到embedding矩阵
    def get_embedding_matrix(self,word_index, embed_size=300, Emed_path="w2v_300.txt"):
        embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
            Emed_path, binary=False)
        nb_words = len(word_index) + 1
        embedding_matrix = np.zeros((nb_words, embed_size))
        count = 0
        for word, i in tqdm(word_index.items()):
            if i >= nb_words:
                continue
            try:
                embedding_vector = embeddings_index[word]
            except:
                embedding_vector = np.zeros(embed_size)
                count += 1
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        print("null cnt", count)
        return embedding_matrix


    def stat_mode(df, cate_fea, num_fea):
        aim_ = pd.concat([df.groupby(cate_fea)[num_].agg(
            {num_ + '_mode': lambda x: stats.mode(x)[0][0], num_ + '_mode_count': lambda x: stats.mode(x)[1][0]}) for num_
                          in num_fea], axis=1)
        aim_.reset_index(inplace=True)

        return aim_

    def format_df(self,df, target_id, ad, cate_col):
        train_df = df[df[target_id].isnull() == False]
        test_df = df[df[target_id].isnull() == True]
        train_target = train_df[target_id].values
        test_target = test_df[target_id].values
        train_user = train_df[cate_col].tolist()
        test_user = test_df[cate_col].tolist()

        del df
        gc.collect()
        train_upbound = train_df.shape[0]
        test_upbound = test_df.shape[0]

        inputs = train_df['{}_logs'.format(ad)].tolist() + test_df['{}_logs'.format(ad)].tolist()

        tf_idf = TfidfVectorizer(ngram_range=(1, 1))
        df = tf_idf.fit_transform(inputs).tocsr()
        train_df = df[:train_upbound]
        test_df = df[train_upbound:]

        return (train_df, test_df), (train_target, test_target), (train_upbound, test_upbound), train_user + test_user


    def get_sklearn_classfiy_stacking(self,clf, train_feature, test_feature, score, model_name, class_number, n_folds, train_num,
                                      test_num, target_id):
        print('\n****开始跑', model_name, '****')
        stack_train = np.zeros((train_num, class_number))
        stack_test = np.zeros((test_num, class_number))
        score_mean = []
        skf = StratifiedKFold(n_splits=n_folds, random_state=1017)
        tqdm.desc = model_name
        for i, (tr, va) in enumerate(skf.split(train_feature, score)):
            clf.fit(train_feature[tr], score[tr])
            score_va = clf._predict_proba_lr(train_feature[va])
            score_te = clf._predict_proba_lr(test_feature)
            score_single = accuracy_score(score[va], np.argmax(clf._predict_proba_lr(train_feature[va]), axis=1))
            score_mean.append(np.around(score_single, 5))
            print('{}+1/{}:{}'.format(i, n_folds, score_single))
            stack_train[va] += score_va
            stack_test += score_te
        stack_test /= n_folds
        stack = np.vstack([stack_train, stack_test])
        df_stack = pd.DataFrame()
        for i in range(stack.shape[1]):
            df_stack['tfidf_ori_1_1_' + model_name + '_classfiy_{}_{}'.format(i, target_id)] = stack[:, i]
        print(model_name, '处理完毕')
        return df_stack, score_mean


    def gen_tfidf_fea(self,df, ad_list, label_list, label_classify, cate_fea):
        model_list = [
            ['PassiveAggressiveClassifier', PassiveAggressiveClassifier(random_state=1017, C=2, n_jobs=12)],
            ['LogisticRegression', LogisticRegression(random_state=1017, C=3, n_jobs=12)],
            ['SGDClassifier', SGDClassifier(random_state=1017, loss='log', n_jobs=12)],
            ['RidgeClassfiy', RidgeClassifier(random_state=1017)],
            ['LinearSVC', LinearSVC(random_state=1017, verbose=1)]
        ]

        tf_list = []
        for ad in tqdm(ad_list, desc='ad_feature'):
            print('loading weight')
            for name, classify in zip(label_list, label_classify):
                if not os.path.exists('work/tf_idf/tfIdf_feature_{}_label_{}.csv'.format(ad, name)):
                    df = self.gen_behavior_seq(df=df, cate_fea=cate_fea, item_col=ad)
                    df = df.merge(pd.read_csv('work/train/train_user.csv', usecols=['phone_no_m', 'label']).rename(
                        columns={'label': 'target'}), how='left', on=['phone_no_m'])
                    df, target, upbound, user_df = self.format_df(df, target_id=name, ad=ad, cate_col=cate_fea[0])
                    feature = pd.DataFrame()
                    for i in model_list:
                        stack_result, score_mean = self.get_sklearn_classfiy_stacking(i[1], df[0], df[1], target[0], i[0],
                                                                                 classify, 5, upbound[0], upbound[1],
                                                                                 target_id=name)
                        feature = pd.concat([feature, stack_result], axis=1, sort=False)
                        print('五折结果', score_mean)
                        print('平均结果', np.mean(score_mean))
                    print(feature.head())
                    feature['phone_no_m'] = user_df
                    feature.to_csv('work/tf_idf/tfIdf_feature_{}_label_{}.csv'.format(ad, name), index=False)
                    tf_list.append(feature)
                else:
                    tf_list.append(pd.read_csv('work/tf_idf/tfIdf_feature_{}_label_{}.csv'.format(ad, name)))

        return pd.concat(tf_list, axis=1)


    def eda_null(self,df):
        for i in df:
            print('{}null radio:{}'.format(i, df[i].isnull().sum() / df.shape[0]))


    def gen_w2v(self,input_docs, save_path, embed_size=16):
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
        w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=10, seed=1017, workers=24, min_count=1, iter=10)
        w2v.wv.save_word2vec_format(save_path)
        print("w2v model done")


    def get_vec(self,input_docs, cate_fea, item_col, Emed_path, embed_size=8):
        cate_name = cate_fea
        if isinstance(cate_fea, list):
            cate_name = '_'.join(cate_fea)
        embeddings_dict = gensim.models.KeyedVectors.load_word2vec_format(Emed_path, binary=False)
        # data=DataFrame(data=[np.sum([embeddings_dict[i] for i in docs],axis=0)for docs in input_docs],columns=['{}_{}_{}_sum'.format(cate_name,item_col,i) for i in range(embed_size)])
        data = DataFrame(data=[np.mean([embeddings_dict[i] for i in docs], axis=0) for docs in input_docs],
                         columns=['{}_{}_{}_mean'.format(cate_name, item_col, i) for i in range(embed_size)])
        # data=pd.concat([data,DataFrame(data=[np.std([embeddings_dict[i] for i in docs],axis=0)for docs in input_docs],columns=['{}_{}_{}_std'.format(cate_name,item_col,i) for i in range(embed_size)])],axis=1)
        # data=pd.concat([data,DataFrame(data=[np.min([embeddings_dict[i] for i in docs],axis=0)for docs in input_docs],columns=['{}_{}_{}_min'.format(cate_name,item_col,i) for i in range(embed_size)])],axis=1)
        # data=pd.concat([data,DataFrame(data=[[embeddings_dict[i] for i in docs][-1]for docs in input_docs],columns=['{}_{}_{}_last'.format(cate_name,item_col,i) for i in range(embed_size)])],axis=1)

        return data


    def gen_history_seq(self,df, cate_fea, item_col, time_col, embed_size=8):
        cate_name = cate_fea
        if isinstance(cate_name, list):
            cate_name = '_'.join(cate_name)
        df[item_col] = df[item_col].astype('str')
        t_ = df[cate_fea + [item_col, time_col]].sort_values(time_col).groupby(cate_fea)[item_col].apply(
            lambda x: ','.join(x.tolist())).reset_index()
        save_path = 'work/w2v/{}_{}_{}_w2v.txt'.format(cate_name, item_col, embed_size)
        input_docs = [i.split(',') for i in t_[item_col].tolist()]
        if not os.path.exists(save_path):
            self.gen_w2v(input_docs=input_docs, save_path=save_path, embed_size=embed_size)
        t_ = pd.concat([t_[cate_fea],
                        self.get_vec(input_docs=input_docs, cate_fea=cate_fea, Emed_path=save_path, item_col=item_col,
                                embed_size=embed_size)], axis=1)

        return t_


    def stat_fea(self,cate_, num_fea, agg_op, train_df=DataFrame(), test_df=DataFrame(), df=None):
        ori_df = df

        if not train_df.empty:
            df = train_df

        t_ = DataFrame()
        for fea in num_fea:
            cate_name = cate_
            if isinstance(cate_, list):
                cate_name = '_'.join(cate_)
            op_dict = {'{}_{}_{}'.format(cate_name, fea, agg_): agg_ for agg_ in agg_op}
            if t_.empty:
                t_ = df.groupby(cate_)[fea].agg(op_dict)
            else:
                t_ = pd.concat([t_, df.groupby(cate_)[fea].agg(op_dict)], axis=1)

        return t_.reset_index()


    def batch_stat(self,cate_fea, num_fea, agg_op, train_df=DataFrame(), test_df=DataFrame(), df=None, need_df=None):
        aim_ = [self.stat_fea(df=df, cate_=cate_, num_fea=num_fea, agg_op=agg_op) for cate_ in tqdm(cate_fea)]
        return aim_


    def gen_behavior_seq(self,df, cate_fea, item_col,time_col):
        cate_name = cate_fea
        if isinstance(cate_name, list):
            cate_name = '_'.join(cate_name)
        df[item_col] = df[item_col].astype('str')
        t_ = df[cate_fea + [item_col, time_col]].sort_values(time_col).groupby(cate_fea)[item_col].apply(
            lambda x: ','.join(x.tolist())).reset_index()
        t_.rename(columns={item_col: '{}_logs'.format(item_col)}, inplace=True)

        return t_


    def format_unstack(self,aim_):
        aim_.columns = [str(i[0]) + '_' + str(i[1]) for i in aim_.columns.tolist()]
        aim_.reset_index(inplace=True)

        return aim_
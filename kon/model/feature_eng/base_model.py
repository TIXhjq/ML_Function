 # _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-10-30 下午9:36
================================='''
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score, r2_score
from hyperopt import fmin, tpe, hp, partial
from numpy.random import random, shuffle
import matplotlib.pyplot as plt
from pandas import DataFrame
# import tensorflow as tf
from tqdm import tqdm
# from PIL import Image
import lightgbm as lgb
import networkx as nx
import pandas as pd
import numpy as np
import warnings
# import cv2
import os
import gc
import re
import datetime
import sys

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
#----------------------------------------------------
class base_model():
    def __init__(self,save_folder,random_state=2048):
        print('base model is backend')
        self.random_state=random_state
        self.save_folder=save_folder

    def model_fit(self,X_train,y_train,cate_fea,X_vail,y_vail,is_pred=True,test_data=None,loss=['cross_entropy','binary'],is_classiffy=True,threshold=0.103):
        if is_classiffy:
            loss=loss[0]
        else:
            loss=loss[1]

        lgb_model = lgb.LGBMRegressor(
            num_leaves=40, reg_alpha=1, reg_lambda=0.1, objective=loss,
            max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=self.random_state,
            n_estimators=8000, subsample=0.8, colsample_bytree=0.8,is_unbalance=True,
            device='gpu'
            # n_jobs=-1
        )

        lgb_model.fit(X_train,y_train,eval_set=[(X_vail,y_vail)],eval_metric='auc',
                      categorical_feature=cate_fea,
                      early_stopping_rounds=300,verbose=10)

        result_weight=lgb_model.best_score_['valid_0']['auc']
        # result_weight=lgb_model.best_score_['training']['binary_logloss']

        model_import = DataFrame()
        model_import['feature'] = X_train.columns.tolist()
        model_import['feature_importance'] = lgb_model.feature_importances_
        model_import['model_weight'] = result_weight
        model_import.sort_values(by=['feature_importance'], ascending=False, inplace=True)
        zero_fea_list = model_import[model_import['feature_importance'] != 0]['feature'].tolist()

        print(model_import.head())
        print('-------------------------------')

        if is_classiffy:
            vail_y_pred = lgb_model.predict(X_vail, num_iteration=lgb_model.best_iteration_)
            vail_result = DataFrame(data=vail_y_pred, columns=['vail_pred'])
            vail_result['y_vail'] = y_vail
            vail_result.sort_values(['vail_pred'], ascending=False, inplace=True)
            vail_result.reset_index(inplace=True)

            del vail_result['index']
            gc.collect()

            vail_result.loc[vail_result.index <= int(vail_result.shape[0] * threshold), 'vail_pred'] = 1
            vail_result.loc[vail_result.vail_pred != 1, 'vail_pred'] = 0
            print(vail_result.head())
            try:
                print(f1_score(y_pred=vail_result['vail_pred'].tolist(),y_true=vail_result['y_vail'].tolist()))
            except ValueError:
                print('ERROR')
            del vail_result

        if is_pred==True:
            result_data = np.array(lgb_model.predict(test_data, num_iteration=lgb_model.best_iteration_ + 10))
            result_=DataFrame(columns=['result'],data=result_data)
            result_['weight']=result_weight
            return result_,zero_fea_list,model_import
        return zero_fea_list,model_import



    def avg_model_pred(self,result_data,n_split,test_data,is_plot=True,is_avg=True):
        print(result_data.head())

        # cal weight_avg_result
        result_cols = []
        weight_cols = []
        for i in range(0, n_split):
            result_cols.append('result_' + str(i))
            weight_cols.append('weight_' + str(i))

        result_data['result'] = 0

        for w_col, r_col in zip(weight_cols, result_cols):
            if not is_avg:
                result_data[w_col] /= result_data['weight']
            else:
                result_data[w_col]=1/n_split
            print(result_data[w_col].head())
            result_data[r_col] *= result_data[w_col]

        for col in result_cols:
            result_data['result'] += result_data[col]

        score = result_data['weight'].unique().tolist()[0] / n_split

        submit_data = DataFrame()
        submit_data['ID'] = test_data.ID.tolist()
        submit_result = []

        for r in result_data.result:
            if r <= 0:
                submit_result.append(0.1)
            else:
                submit_result.append(r)
        submit_data['Label'] = submit_result

        del result_data
        gc.collect()

        print('model_score:{}'.format(score))

        if is_plot:
            data = DataFrame(submit_data.Label.value_counts()).reset_index()
            plt.bar(data['index'], data['Label'])

        return submit_data,score


    def n_fold_fit(self,train_data,cols,cate_col,test_data=None,label_col='Label',is_pred=True):
        #train by k_fold
        result_data=DataFrame()
        if is_pred:
            result_data['weight']=[0]*test_data.shape[0]
        fea_filter =[]
        n_split=10
        rank=0

        k=StratifiedKFold(n_splits=n_split,random_state=self.random_state,shuffle=True)

        all_feature_important=DataFrame()
        all_feature_important['feature']=cols
        for train_idx,test_idx in tqdm(k.split(train_data[cols],train_data[label_col]),desc='k_split_fitting'):
            X_train=train_data[cols].loc[train_idx]
            X_vail=train_data[cols].loc[test_idx]

            y_train=train_data[[label_col]].loc[train_idx]
            y_vail=train_data[[label_col]].loc[test_idx]

            if is_pred:
                result_,zero_fea,feature_important=self.model_fit(X_train=X_train,y_train=y_train,X_vail=X_vail,y_vail=y_vail,test_data=test_data[cols],cate_fea=cate_col,is_pred=is_pred)
                result_data['result_'+str(rank)]=result_['result']
                result_data['weight_'+str(rank)]=result_['weight']
                result_data['weight']+=result_['weight']
                del result_
                gc.collect()

            if not is_pred:
                zero_fea,feature_important=self.model_fit(X_train=X_train,y_train=y_train,X_vail=X_vail,y_vail=y_vail,cate_fea=cate_col,is_pred=is_pred)

            feature_important.columns=['feature']+[str(col)+'_'+str(rank) for col in feature_important.columns.tolist()[1:]]
            all_feature_important=all_feature_important.merge(feature_important,'left',on=['feature'])
            fea_filter.append(zero_fea)
            rank+=1

        np.save(self.save_folder+'zero_feature',fea_filter)

        return result_data,n_split,all_feature_important,fea_filter

    def save_feature_submit(self,submit_data,score,cols,cate_fea):
        cate_fea_label = []
        for col in cols:
            if col in cate_fea:
                cate_fea_label.append(1)
            else:
                cate_fea_label.append(0)

        model_features = DataFrame()
        model_features['cols'] = cols
        model_features['is_cate'] = cate_fea_label

        feature_path = self.save_folder+'use_feature/' + str(datetime.datetime.now().date()) + '/'
        result_path = self.save_folder+ 'result/' + str(datetime.datetime.now().date()) + '/'

        for path in [feature_path, result_path]:
            if not os.path.exists(path):
                os.mkdir(path)

        model_features.to_csv(
            feature_path + 'model_feature_' + str(datetime.datetime.now()) + '_' + str(score) + '.csv', index=None)

        submit_data.drop(columns=['weight'],inplace=True)
        submit_data.to_csv(result_path + 'submit_' + str(datetime.datetime.now()) + '_' + str(score) + '.csv',
                           index=None)

    def fit_transform(self,train_data:DataFrame,use_cols,cate_cols,label_col:str,test_data=None,is_pred=True):
        if is_pred:
            result_data,n_split,feature_important,zero_fea=self.n_fold_fit(train_data=train_data,test_data=test_data,label_col=label_col,cols=use_cols,cate_col=cate_cols,is_pred=is_pred)
            submit_data,score=self.avg_model_pred(result_data=result_data,n_split=n_split,test_data=test_data)
            self.save_feature_submit(score=score,submit_data=submit_data,cate_fea=cate_cols,cols=use_cols)
        else:
            result_data,n_split,feature_important,zero_fea=self.n_fold_fit(train_data=train_data,label_col=label_col,is_pred=is_pred,cols=use_cols,cate_col=cate_cols)
        return feature_important,zero_fea

    def single_fit_transform(self,X_train,y_train,X_vail,y_vail,cate_cols,test_data,pred_id,is_classiffy=True,threshold=0.103):
        result_, zero_fea_list, model_import=self.model_fit(X_train,y_train,cate_cols,X_vail,y_vail,test_data=test_data,is_classiffy=is_classiffy,threshold=threshold)
        result_['id']=pred_id
        score=result_['weight'].unique().tolist()[0]
        cols=X_train.columns.tolist()
        self.save_feature_submit(score=score, submit_data=result_, cate_fea=cate_cols, cols=cols)

if __name__=='__main__':
    data_folder ='../../data/'
    submit_data_folder = data_folder + 'submit_data/'

    from sklearn.datasets import load_iris
    iris = load_iris()
    train_data=iris.data
    target_data=iris.target
    train_fea=iris.feature_names
    train_data=DataFrame(data=train_data,columns=train_fea)
    target_data=DataFrame(data=target_data,columns=['target'])
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(train_data,target_data,test_size=0.3,random_state=2048)
    base_model(submit_data_folder).single_fit_transform(X_train,y_train,X_test,y_test,cate_cols=[],test_data=train_data,pred_id=train_data.index.tolist(),is_classiffy=False,threshold=0.103)
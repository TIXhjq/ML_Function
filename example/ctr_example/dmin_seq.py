#!/usr/bin/env python

# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/11/23 下午7:15
@File   :dmin_seq.py
@email  :hjq1922451756@gmail.com or 1922451756@qq.com  
================================='''
from kon.model.ctr_model.model.models import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
# ----------------------------------------------------
data_folder = '../../data/'
origin_data_folder = data_folder + 'origin_data/'
submit_data_folder = data_folder + 'submit_data/'
eda_data_folder = data_folder + 'eda_data/'
fea_data_folder = data_folder + 'fea_data/'
# -----------------------------------------------------------------
model_tool = base_model(submit_data_folder)
fea_tool = feature_tool(fea_data_folder)
data_pre = data_prepare(batch_size=32)
# -----------------------------------------------------------------
columns = ["date", "user_id", "price", "ad_id", "cate_id", "target", "day"]

trainDf = pd.read_csv(origin_data_folder + 'ali_data/train.csv', usecols=columns, nrows=100)
testDf = pd.read_csv(origin_data_folder + 'ali_data/test.csv', usecols=columns, nrows=100)

df = pd.concat([trainDf, testDf], axis=0)
df["date"] = pd.to_datetime(df.date)
df.sort_values(["date"], inplace=True)
print(df.head())

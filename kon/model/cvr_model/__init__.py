#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/9/26 下午6:23
@File   :__init__.py.py
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
from kon.model.embedding.setence_model import *
from kon.model.feature_eng.feature_transform import feature_tool
from kon.model.feature_eng.base_model import base_model
from kon.model.ctr_model.model.models import *
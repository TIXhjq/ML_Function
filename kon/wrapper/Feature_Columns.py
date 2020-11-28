#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/9/30 下午7:56
@File   :Feature_Columns.py
@email  :hjq1922451756@gmail.com or 1922451756@qq.com  
================================='''
from collections import namedtuple

class NumsFeaWrapper(object):
    def __init__(self):
        self.denseFea=namedtuple('denseFea',['fea_name','batch_size'])

class SparseFeaWrapper(object):
    def __init__(self):
        self.sparseFeature=namedtuple('sparseFea',
                   ['fea_name', 'word_size', 'input_dim', 'cross_unit', 'linear_unit', 'pre_weight', 'mask_zero',
                    'is_trainable', 'input_length', 'sample_num', 'batch_size', 'emb_reg'])

class NumsFea(NumsFeaWrapper):
    def __init__(self):
        super(NumsFea, self).__init__()
        self.fea=self.denseFea

class CateFea(SparseFeaWrapper):
    def __init__(self):
        super(CateFea, self).__init__()
        self.fea=self.sparseFeature

class BehaviorFea(SparseFeaWrapper):
    def __init__(self):
        super(BehaviorFea, self).__init__()
        self.fea = self.sparseFeature






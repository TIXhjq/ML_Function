#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/3 上午11:41
@File   :core_layer.py
================================='''
import tensorflow as tf
from tensorflow_core.python.keras.initializers import glorot_uniform
import pandas as pd
import warnings
import os
from kon.model.feature_eng.feature_transform import feature_tool
from kon.model.feature_eng.base_model import base_model

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
#----------------------------------------------------
data_folder='../../data/'
origin_data_folder=data_folder+'origin_data/'
submit_data_folder=data_folder+'submit_data/'
eda_data_folder=data_folder+'eda_data/'
fea_data_folder=data_folder+'fea_data/'
#-----------------------------------------------------------------
model_tool=base_model(submit_data_folder)
fea_tool=feature_tool(fea_data_folder)
#-----------------------------------------------------------------
class StackLayer(tf.keras.layers.Layer):
    '''
        support:
            concat(flatten)
    '''
    def __init__(self,use_flat=True,axis=None):
        super(StackLayer, self).__init__()
        if axis:
            self.concat = tf.keras.layers.Concatenate(axis=axis)
        else:
            self.concat = tf.keras.layers.Concatenate()
        self.use_flat=use_flat

    def build(self, input_shape):
        super(StackLayer, self).build(input_shape)
        self.flat = [tf.keras.layers.Flatten(name='stack_flatten_{}'.format(str(i))) for i in range(len(input_shape))]

    def call(self, inputs, **kwargs):
        if self.use_flat:
            inputs=[flat_(input_) for input_,flat_ in zip(inputs,self.flat)]
        if len(inputs)==1:
            return inputs[0]
        else:
            return self.concat(inputs)


class ScoreLayer(tf.keras.layers.Layer):
    def __init__(self,use_add=False,use_inner=False,use_global=False,seed=2020):
        from kon.model.ctr_model.layer.interactive_layer.interactive_layer import InnerLayer
        super(ScoreLayer, self).__init__()
        self.use_add=use_add
        self.add=tf.keras.layers.Add()
        self.activate=tf.keras.layers.Activation('sigmoid')
        self.use_inner=use_inner
        self.inner=InnerLayer(use_inner=True)
        self.use_global=use_global
        self.seed=seed

    def build(self, input_shape):
        super(ScoreLayer, self).build(input_shape)
        if self.use_global:
            self.global_bias=self.add_weight(shape=(1,),initializer=glorot_uniform(self.seed))

    def call(self, inputs, **kwargs):
        if self.use_add:
            inputs=self.add(inputs)
            if self.use_global:
                inputs=self.add([inputs,self.global_bias])
        if self.use_inner:
            inputs=self.inner(inputs)

        output=self.activate(inputs)
        return output

class MergeScoreLayer(tf.keras.layers.Layer):
    def __init__(self,use_merge:bool=True,output_dim=2):
        super(MergeScoreLayer, self).__init__()
        self.concat=StackLayer()
        self.dense=tf.keras.layers.Dense(units=output_dim,activation='softmax')
        self.use_merge=use_merge

    def build(self, input_shape):
        super(MergeScoreLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.use_merge:
            inputs=self.concat(inputs)
        x=self.dense(inputs)
        return x

class HiddenLayer(tf.keras.layers.Layer):
    '''
        notice:
            can to replace dense,to use other method to cal
            e.g:can to mult-head-attention achieve autoint
        Dnn core:
            hidden achieve
        In feature, to drop it
    '''
    def __init__(self,hidden_units:int,use_bn:bool=True,seed=2020,other_dense=None):
        super(HiddenLayer, self).__init__()
        self.dense=tf.keras.layers.Dense(
            units=hidden_units,kernel_initializer=glorot_uniform(seed=seed),
            bias_initializer=glorot_uniform(seed=seed)
        )
        if other_dense:
            self.dense=other_dense
        self.bn=tf.keras.layers.BatchNormalization()
        self.use_bn=use_bn

    def build(self, input_shape):
        super(HiddenLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x=self.dense(inputs)
        if self.use_bn:
            x=self.bn(x)
        return x,inputs

class ResActivateLayer(tf.keras.layers.Layer):
    '''
        notice:
            res layer activate,support ln,bn...
    '''
    def __init__(self,use_bn,use_ln,hidden_activate):
        super(ResActivateLayer, self).__init__()
        self.use_ln = use_ln
        self.use_bn = use_bn
        self.ln = tf.keras.layers.LayerNormalization()
        self.bn = tf.keras.layers.BatchNormalization()
        self.active = hidden_activate


    def build(self, input_shape):
        super(ResActivateLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.use_bn:
            inputs = self.bn(inputs)
        if self.use_ln:
            inputs = self.ln(inputs)

        x = self.active(inputs)

        return x


class DnnLayer(tf.keras.layers.Layer):
    def __init__(self,hidden_units:list=None,hidden_activate=tf.keras.layers.ReLU(),use_bn:bool=False,res_unit=1,
                 output_dim=-1,seed=2020,other_dense=None,use_ln:bool=False,use_flatten=False,**kwargs):
        '''
        notice:
            dense of dnn can to replace other layer,
            e.g:mult head atten(autoInt),
            to_replace:other_dense,succ to replace.

        :param hidden_units:please make sure to need units list
            when use other dense,need to input it too.
            e.g need 3 hidden,but use other dense==>[[],[],[]]
                num is not import,shape is very import

        :param res_unit:res add skip num

        :param activate:hidden activate
            Dnn core:
                supports auto bn
        '''
        super(DnnLayer, self).__init__(**kwargs)
        self.hidden_list=other_dense
        if not other_dense:
            self.hidden_list=[HiddenLayer(hidden_units=dim,use_bn=False,other_dense=other_dense)for dim in hidden_units]
        self.activate=hidden_activate
        self.activate=[ResActivateLayer(use_bn=use_bn,use_ln=use_ln,hidden_activate=hidden_activate) for idx_ in range(len(self.hidden_list))]
        self.add=tf.keras.layers.Add()
        self.seed=2020
        self.output_dim=output_dim
        self.res_unit=res_unit
        if output_dim!=-1:
            self.logit_layer=tf.keras.layers.Dense(
                units=output_dim,kernel_initializer=glorot_uniform(seed=seed),
                bias_initializer=glorot_uniform(seed=seed)
            )
        if use_flatten:
            self.flat=tf.keras.layers.Flatten()
        self.use_flatten=use_flatten

    def build(self, input_shape):
        super(DnnLayer, self).build(input_shape)

    def call(self, inputs,**kwargs):
        x=inputs
        res=[[],[]]
        for idx_,hidden_layer in enumerate(self.hidden_list):
            [x,ori]=hidden_layer(x)
            if idx_==0:
                res=[ori,x]
            if (idx_+1)%self.res_unit!=0 or self.res_unit==1:
                res[-1]=x
            if (idx_+1)%self.res_unit==0:
                try:
                    x=self.add(res)
                except ValueError:
                    x=res[-1]

            x=self.activate[idx_](x)
            if (idx_+1)%self.res_unit==0:
                res[0]=x

        if self.use_flatten:
            x = self.flat(x)

        if self.output_dim!=-1:
            x=self.logit_layer(x)

        return x

class IntraViewPoolingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(IntraViewPoolingLayer, self).__init__()

    def build(self, input_shape):
        super(IntraViewPoolingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        output=tf.expand_dims(tf.reduce_mean(inputs,axis=1),axis=1)

        return output

class AlignLayer(tf.keras.layers.Layer):
    '''
        format dim,if [a,b.,.] dim not eq,
        format to [a,b...] higher dim
    '''
    def __init__(self):
        super(AlignLayer, self).__init__()

    def build(self, input_shape):
        super(AlignLayer, self).build(input_shape)
        dim_list=[i[-1] for i in input_shape]
        max_dim=max(dim_list)
        self.format_dense=[tf.keras.layers.Dense(
            units=max_dim) if i<max_dim else None for i in dim_list]

    def call(self, inputs, **kwargs):
        return [format_(input_) if format_!=None else input_
            for input_,format_ in zip(inputs,self.format_dense)]
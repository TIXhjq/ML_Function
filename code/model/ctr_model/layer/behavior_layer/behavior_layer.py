#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/3 上午11:40
@File   :behavior_layer.py
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
from model.embedding.setence_model import *
from model.feature_eng.feature_transform import feature_tool
from model.feature_eng.base_model import base_model
from model.ctr_model.layer.interactive_layer.interactive_layer import *
from model.ctr_model.layer.core_layer.core_layer import *

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


class SeqBaseLayer(tf.keras.layers.Layer):
    '''
        DIN's base-seq transform
            avg([seq embed list])
    '''
    def __init__(self,supports_masking=True,mask_zero=False):
        super(SeqBaseLayer, self).__init__()
        self.supports_masking=supports_masking
        self.mask_zero=mask_zero

    def build(self, input_shape):
        super(SeqBaseLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return [tf.expand_dims(tf.reduce_sum(input_,axis=1),axis=1) for input_ in inputs]

    def compute_mask(self, pre_mask, mask=None):
        if not self.mask_zero:
            return None
        return pre_mask

class AttentionUnitLayer(tf.keras.layers.Layer):
    '''
        DIN Attention core:
            attention=g(candidate,behavior)
            g=dnn()([candidate,candidate-behavior,behavior])
        return weight
    '''
    def __init__(self,hidden_units):
        super(AttentionUnitLayer, self).__init__()
        self.dnn=DnnLayer(hidden_units=hidden_units,output_dim=1,res_unit=2)
        self.add=tf.keras.layers.Add()
        self.softmax=tf.keras.layers.Activation('softmax')
        self.stack = StackLayer(use_flat=False)

    def build(self, input_shape):
        super(AttentionUnitLayer, self).build(input_shape)

    def call(self, inputs, mask=None , **kwargs):
        [stack_candidate, stack_behavior] = inputs
        stack_candidate = tf.keras.backend.repeat_elements(stack_candidate, stack_behavior.shape[1], 1)
        activation_inputs = self.stack([stack_candidate, stack_candidate - stack_behavior, stack_behavior])
        attention_weight = self.dnn(activation_inputs)
        attention_weight = tf.cast(attention_weight, tf.float32)
        mask_weight = tf.ones_like(attention_weight) * (-2 ** 32 + 1)
        attention_weight = tf.where(tf.expand_dims(mask, -1), attention_weight, mask_weight)
        attention_weight = self.softmax(attention_weight)


        return attention_weight

class ActivationUnitLayer(tf.keras.layers.Layer):
    '''
        origin paper not use softmax
            if use:==>use_softmax=True

        notice:
            inputs:must[stack(key),stack(query)]
    '''
    def __init__(self,hidden_units,supports_mask=True,mask_zero=False,need_stack=True,return_seq=False):
        super(ActivationUnitLayer, self).__init__()
        self.attention_weight=AttentionUnitLayer(hidden_units)
        self.supports_masking=supports_mask
        self.mask_zero=mask_zero
        self.stack=StackLayer(use_flat=False)
        self.need_stack = need_stack
        self.return_seq=return_seq

    def build(self, input_shape):
        super(ActivationUnitLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if self.need_stack:
            inputs=[self.stack(i) for i in inputs]
        behavior_=inputs[1]
        attention_weight=self.attention_weight(inputs,mask=mask)

        if self.return_seq:
            return behavior_*(tf.keras.backend.repeat_elements(
                attention_weight,behavior_.shape[-1],axis=-1))

        return tf.squeeze(tf.matmul(
            tf.transpose(attention_weight, [0, 2, 1]), behavior_),1)

    def compute_mask(self, pre_mask, mask=None):
        if not self.mask_zero:
            return None
        return pre_mask

class AuxiliaryLayer(tf.keras.layers.Layer):
    '''
        notice:
            paper use prob=(sigmoid)[HIDDEN_UNIT*i for i in sample_seq]
            however this default dnn to classify,
            if need to use origin,please hidden_unit=[]
    '''
    def __init__(self,hidden_units:list,sample_num:int):
        super(AuxiliaryLayer, self).__init__()

        self.classify = ScoreLayer()
        self.stack=StackLayer(use_flat=False)
        self.softmax=tf.keras.layers.Activation('sigmoid')
        if hidden_units!=[]:
            self.classify=[DnnLayer(hidden_units=hidden_units,output_dim=1) for i in range(sample_num)]

    def build(self, input_shape):
        super(AuxiliaryLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        [inputs,sample_seq]=inputs
        [pos_seq, nega_seq] = sample_seq
        inputs=inputs[:,:-1,:]
        mask=tf.cast(tf.expand_dims(mask,2),'float')
        mask=self.stack([mask[:,:-1],mask[:,:-1]])
        click=self.stack([inputs,pos_seq])
        unclick=self.stack([inputs,nega_seq])
        [pos_prob,nega_prob]=[((self.softmax(self.classify[idx](input_))+(1e-10))*mask)[:,:,0] for idx,input_ in enumerate([click,unclick])]
        prob=-tf.reduce_mean(tf.math.log_sigmoid(pos_prob)+tf.math.log_sigmoid(1-nega_prob))

        return prob

class SampleLayer(tf.keras.layers.Layer):
    def __init__(self,sample_num):
        super(SampleLayer, self).__init__()
        self.sample_num=sample_num
        self.stack = StackLayer(use_flat=False, axis=1)

    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = tf.split(inputs, [1] * inputs.shape[1], 1)
        pos_seq = self.stack(inputs[1:])
        inputs = [tf.squeeze(input_, axis=1) for input_ in inputs]
        neg_seq = self.stack([tf.expand_dims(self.stack(list(np.random.choice(
            [j for j_idx, j in enumerate(inputs) if j_idx != idx_], size=self.sample_num,replace=False
        ))), axis=1)for idx_, i in enumerate(inputs)][1:])

        return [pos_seq, neg_seq]


class InterestExtratorLayer(tf.keras.layers.Layer):
    '''
        notice:
            inputs format:[seq,seq_sample]
            if not need sample,
                please seq_sample=[]
    '''
    def __init__(self,classify_units,sample_num):
        super(InterestExtratorLayer, self).__init__()
        self.aux_loss=AuxiliaryLayer(hidden_units=classify_units,sample_num=sample_num)

    def build(self, input_shape):
        super(InterestExtratorLayer, self).build(input_shape)
        self.gru =tf.keras.layers.GRU(input_shape[0][-1],return_sequences=True)

    def call(self, inputs, mask=None , **kwargs):
        [inputs,behavior_sample]=inputs
        gru_hidden=self.gru(inputs)
        if behavior_sample!=[]:
            return gru_hidden,self.aux_loss([gru_hidden,behavior_sample],mask=mask)
        else:
            return gru_hidden

class InterestEolvingLayer(tf.keras.layers.Layer):
    '''
        notice:
            1) achieve:
                only support AIGRU,
                argru[format:<score*update gate>] have a little error,
                may be to update at soon,
                # only support standard gru,because,
                #     cudnn mod have a little complex,may be to update at soon
            2) augru inputs:
                must be keep inputs=[seq,score]
            3) eolving inputs:
                must be keep inputs=[target_list,hidden_list]
    '''
    def __init__(self,attention_units,eolving_type='aigru',supports_masking=True,mask_zero=False):
        super(InterestEolvingLayer, self).__init__()
        self.supports_masking=supports_masking
        self.mask_zero=mask_zero
        if eolving_type!='aigru':
            self.attention_score = AttentionUnitLayer(hidden_units=attention_units)
        else:
            self.attention_seq=ActivationUnitLayer(hidden_units=attention_units,need_stack=False,return_seq=True)
        self.eolving_type=eolving_type

    def build(self, input_shape):
        super(InterestEolvingLayer, self).build(input_shape)
        if self.eolving_type !='aigru':
            self.augru = AUGRU(units=input_shape[1][-1])
        else:
            self.gru=tf.keras.layers.GRU(units=input_shape[1][-1])

    def call(self, inputs,mask=None, **kwargs):
        if self.eolving_type!='aigru':
            score=self.attention_score(inputs,mask=mask)
            output=self.augru([inputs[1],score[:,:-1,:]])
        else:
            attention_seq=self.attention_seq(inputs=inputs,mask=mask)
            output=self.gru(attention_seq)

        return output

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return mask

class PositionalEncodeLayer(tf.keras.layers.Layer):
    '''
        notice:
            input not is list
        source:
            TransFormer
    '''
    def __init__(self,use_add=False):
        super(PositionalEncodeLayer, self).__init__()
        self.use_add=use_add

    def build(self, input_shape):
        super(PositionalEncodeLayer, self).build(input_shape)
        self.max_length,self.embed_dim=input_shape[1:]

    def call(self, inputs, **kwargs):
        def position_cal(i,j,embed_dim):
            return i / pow(10000, (2 * j / embed_dim))

        position_info=tf.convert_to_tensor(
            [[tf.sin(position_cal(i,j,self.embed_dim)) if j%2==0 else tf.cos(position_cal(i,j,self.embed_dim))
                for j in range(self.embed_dim) ]for i in range(self.max_length)])

        if self.use_add:
            return inputs+position_info
        else:
            return position_info

class ProductAttentionLayer(tf.keras.layers.Layer):
    '''
        notice:
            product attention
            support scale<source: Transformer>
    '''
    def __init__(self,use_scale=False,supports_masking=True,mask_mod=1):
        '''
        :param mask_mod:
            {1:have BS mask,2:no BS mask}
        '''
        super(ProductAttentionLayer, self).__init__()
        self.use_scale=use_scale
        self.supports_masking=supports_masking
        self.softmax = tf.keras.layers.Activation('sigmoid')
        self.mask_mod=mask_mod

    def build(self, input_shape):
        super(ProductAttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        [q,k,v]=inputs
        atten_score=tf.matmul(q,k,transpose_b=True)

        if self.use_scale:
            atten_score/=(q.shape[-1]**0.5)

        if mask!=None:
            if self.mask_mod==1:
                mask=tf.cast(mask,'float')
                atten_score=tf.matmul(atten_score,mask)
            if self.mask_mod==2:
                mask = tf.cast(mask,'float')
                mask = mask*(-100000)
                atten_score=tf.add(atten_score,mask)

        atten_score=self.softmax(atten_score)
        atten_v=tf.matmul(atten_score,v)

        return atten_v

class MultHeadAttentionLayer(tf.keras.layers.Layer):
    '''
        AutoInt use nn-struct hidden dense
        notice:
            attention_dim:input=>q,k,v dim
            attention_head_dim:num of head attention
        warning:
            now,not support res&normal&relu,
            because to as hidden of DnnLayer,DnnLayer struct support Add&BN&Activate
    '''
    def __init__(self,attention_dim,attention_head_dim,seed=2020,use_scale=True,use_res=True,use_ln=True,head_concat=False,supports_masking=True,atten_mask_mod=1):
        super(MultHeadAttentionLayer, self).__init__()
        self.attention_dim=attention_dim
        self.attention_head_dim=attention_head_dim
        self.attention_cal=ProductAttentionLayer(use_scale=use_scale,mask_mod=atten_mask_mod)
        self.seed=seed
        self.use_res=use_res
        self.use_ln=use_ln
        self.ln=tf.keras.layers.LayerNormalization()
        self.head_concat=head_concat
        self.supports_masking=supports_masking

    def build(self, input_shape):
        super(MultHeadAttentionLayer, self).build(input_shape)
        #for iter easy==>weight shape[BS,seqLen,head*atten_dim]
        self.query_w=self.add_weight(
            name='query_w',shape=[input_shape[-1],self.attention_head_dim,self.attention_dim],
            initializer=glorot_uniform(self.seed)
        )
        self.key_w=self.add_weight(
            name='key_w',shape=[input_shape[-1],self.attention_head_dim,self.attention_dim],
            initializer=glorot_uniform(self.seed)
        )
        self.value_w=self.add_weight(
            name='value_w',shape=[input_shape[-1],self.attention_head_dim,self.attention_dim],
            initializer=glorot_uniform(self.seed)
        )
        if self.use_res:
            self.res_w=self.add_weight(
                name='res_w',shape=[input_shape[-1],self.attention_head_dim,self.attention_dim]
            )


    def call(self, inputs, mask=None, **kwargs):
        #format-->[headNum,BS,seqLen,embed_dim]
        q=tf.transpose(tf.tensordot(inputs,self.query_w,axes=1),[2,0,1,3])
        k=tf.transpose(tf.tensordot(inputs,self.key_w,axes=1),[2,0,1,3])
        v=tf.transpose(tf.tensordot(inputs,self.key_w,axes=1),[2,0,1,3])

        atten_v=self.attention_cal([q,k,v],mask=mask)
        res=[]

        if self.use_res:
            res=tf.transpose(tf.tensordot(inputs, self.res_w, axes=1), [2, 0, 1, 3])

        if self.use_ln:
            atten_v=self.ln(atten_v)

        if self.head_concat:
            atten_v=tf.transpose(atten_v,[1,0,2,3])

        if self.attention_head_dim==1:
            return tf.squeeze(atten_v,axis=0)

        return [atten_v,res]

    def compute_mask(self, inputs, mask=None):
        return mask

class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    '''
        notice:
            inner_unit==>position wise hidden unit
    '''
    def __init__(self,inner_unit=4,use_res=True,use_ln=True,head_avgPooling=False):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.ln=tf.keras.layers.LayerNormalization()
        self.hidden_units=[inner_unit]
        self.use_res=use_res
        self.use_ln=use_ln
        self.head_avgPooling=head_avgPooling

    def build(self, input_shape):
        super(PositionWiseFeedForwardLayer, self).build(input_shape)
        self.hiddenlayer = DnnLayer(use_bn=False, use_ln=self.use_ln,res_unit=10, hidden_units=self.hidden_units, output_dim=input_shape[-1])

    def call(self, inputs, **kwargs):
        output=self.hiddenlayer(inputs)
        if self.use_res:
            output+=inputs
        if self.head_avgPooling:
            output=tf.reduce_mean(output,1)

        return output

class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,attention_dim,attention_head_dim,ffn_hidden_unit,mean_pooling=False,use_res=True,use_ln=True,use_scale=True,head_concat=True,supports_masking=True,head_avgPooling=True):
        super(SelfAttentionLayer, self).__init__()
        self.mean_pooling=mean_pooling
        self.mult_atten=MultHeadAttentionLayer(attention_dim=attention_dim,attention_head_dim=attention_head_dim,use_res=use_res,use_ln=use_ln,use_scale=use_scale,head_concat=head_concat)
        self.ffn=PositionWiseFeedForwardLayer(inner_unit=ffn_hidden_unit,head_avgPooling=head_avgPooling)
        self.supports_masking=supports_masking

    def build(self, input_shape):
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs,mask=None ,**kwargs):
        atten_=self.mult_atten(inputs,mask=mask)
        if self.mean_pooling:
            atten_=tf.reduce_mean(atten_,axis=1)
        output=self.ffn(atten_[0])

        return output


class BiasPositionEncodeLayer(tf.keras.layers.Layer):
    def __init__(self,sessionMaxLen,sessionMaxNum,seed=2020):
        super(BiasPositionEncodeLayer, self).__init__()
        self.session_item_idx=sessionMaxLen
        self.session_idx=sessionMaxNum
        self.seed=seed

    def build(self, input_shape):
        super(BiasPositionEncodeLayer, self).build(input_shape)
        self.item_dim=input_shape[-1]

        self.session_pos=self.add_weight(
            name='session_position_idx',
            shape=(1,1,1),initializer=glorot_uniform(seed=self.seed)
        )

        self.session_item_pos=self.add_weight(
            name='session_item_idx',
            shape=(1,self.session_item_idx,1),initializer=glorot_uniform(seed=self.seed)
        )

        self.session_item_dim=self.add_weight(
            name='session_item_dim',
            shape=(1,1,self.item_dim),initializer=glorot_uniform(seed=self.seed)
        )

    def call(self, inputs, **kwargs):
        inputs+=self.session_pos+self.session_item_pos+self.session_item_dim
        output=[tf.squeeze(i,1) for i in tf.split(inputs,[1]*inputs.shape[1],axis=1)]

        return output


class SessionDivisonLayer(tf.keras.layers.Layer):
    def __init__(self,sessionMaxLen,sessionMaxNum):
        super(SessionDivisonLayer, self).__init__()
        self.maxlen=sessionMaxLen
        self.maxnum=sessionMaxNum
        self.position=BiasPositionEncodeLayer(sessionMaxLen,sessionMaxNum)

    def build(self, input_shape):
        super(SessionDivisonLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        session_input=tf.concat([tf.expand_dims(i,1) for i in tf.split(inputs,[self.maxlen]*self.maxnum,axis=1)],1)
        pos_session=self.position(session_input)

        return pos_session


class SessionInterestExtractorLayer(tf.keras.layers.Layer):
    def __init__(self,attention_dim,attention_head_dim,ffn_hidden_unit):
        super(SessionInterestExtractorLayer, self).__init__()
        self.attention_dim=attention_dim
        self.attention_head_dim=attention_head_dim
        self.ffn_hidden_unit=ffn_hidden_unit

    def build(self, input_shape):
        super(SessionInterestExtractorLayer, self).build(input_shape)
        session_num=len(input_shape)
        self.self_atten = [
            SelfAttentionLayer(attention_dim=self.attention_dim, attention_head_dim=self.attention_head_dim,
                               ffn_hidden_unit=self.ffn_hidden_unit) for i in range(session_num)]

    def call(self, inputs, **kwargs):
        attention_seq=[self.self_atten[idx_](input) for idx_,input in enumerate(inputs)]

        return tf.concat(attention_seq,1)

class SessionInterestInteractingLayer(tf.keras.layers.Layer):
    def __init__(self,biLstmUnit,lstm_mode):
        super(SessionInterestInteractingLayer, self).__init__()
        self.biLstm=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=biLstmUnit,return_sequences=True),merge_mode=lstm_mode)

    def build(self, input_shape):
        super(SessionInterestInteractingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        hidden_=self.biLstm(inputs)

        return hidden_


class LatentTimeStreamLayer(tf.keras.layers.Layer):
    '''
        DST core:
            cell ode
    '''
    def __init__(self,ode_mode,seed=2020,supports_masking=True,trainable=True):
        '''
        :param cell_steps:cell num
        :param ode_mode:
            >1:sample
            >2:complex
        '''
        super(LatentTimeStreamLayer, self).__init__()
        self.ode_mode=ode_mode
        self.seed=seed
        self.supports_masking=supports_masking
        self.trainable=trainable

    def build(self, input_shape):
        super(LatentTimeStreamLayer, self).build(input_shape)
        self.init_w = self.add_weight(
            name='init_w', shape=(input_shape[1][-1], input_shape[1][-1]),
            initializer=glorot_uniform(seed=self.seed)
        )

        if self.ode_mode==1:
            self.f=self.add_weight(
                name='sample_w',shape=(input_shape[0][-1],input_shape[1][-1]),
                initializer=tf.keras.initializers.zeros()
            )

        elif self.ode_mode==2:
            from model.ctr_model.layer import DnnLayer
            self.f=DnnLayer(hidden_units=[input_shape[1][-1]]*2,hidden_activate=tf.keras.layers.Activation('sigmoid'),res_unit=10)

    def call(self, inputs, mask=None, **kwargs):
        def step(inputs, states):
            z = states[0]
            if self.ode_mode == 1:
                step_out = inputs * self.f + z
            elif self.ode_mode == 2:
                step_out = inputs * self.f(z)  + z

            return step_out, [step_out]

        [t_inputs,init_states]=inputs

        outputs=tf.keras.backend.rnn(step,t_inputs,[init_states],mask=mask)

        return outputs[1]


class TimeDecodedLayer(tf.keras.layers.Layer):
    def __init__(self,sample_num=1,seed=2020,loss_lambda=0.5,supports_masking=True,trainable=True):
        super(TimeDecodedLayer, self).__init__()
        self.guideLoss = GuideLossLayer(sample_num=sample_num,loss_lambda=loss_lambda,trainable=trainable)
        self.seed=seed
        self.supports_masking=supports_masking
        self.trainable=trainable

    def build(self, input_shape):
        super(TimeDecodedLayer, self).build(input_shape)
        self.hidden_w=self.add_weight(
            name='hidden_transform',shape=(input_shape[1][-1],input_shape[0][0][-1]),
            initializer=glorot_uniform(seed=self.seed)
        )

    def call(self, inputs, mask=None, **kwargs):
        [behavior_list,hidden]=inputs
        hiddenT=tf.tensordot(hidden,self.hidden_w,axes=1)
        loss_=self.guideLoss([behavior_list[0],hiddenT],mask=mask)

        behavior_list[0] = behavior_list[0] + hiddenT[:, :behavior_list[0].shape[1], :]
        if self.trainable==False:
            behavior_list[1]=behavior_list[1]+hiddenT[:,behavior_list[0].shape[1]:,:]

        return behavior_list,loss_

class GuideLossLayer(tf.keras.layers.Layer):
    def __init__(self,sample_num=5,hidden_units=[8,8],loss_lambda=0.5,supports_masking=True,trainable=True):

        super(GuideLossLayer, self).__init__()
        self.sample_= SampleLayer(sample_num=sample_num)
        self.full=[DnnLayer(hidden_units=hidden_units) for i in range(3)]
        self.loss_lambda=loss_lambda
        self.dot=tf.keras.layers.Dot(axes=-1)
        self.supports_masking=supports_masking
        self.trainable=trainable


    def build(self, input_shape):
        super(GuideLossLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        [behavior,hiddenT]=inputs
        [pos_seq, neg_seq]=self.sample_(behavior)
        hidden_seq=hiddenT[:,:-1,:]
        [pos_seq,hidden_seq,neg_seq]=[f_(seq_) for f_,seq_ in zip(self.full,[pos_seq,hidden_seq,neg_seq])]

        pos_=tf.reduce_sum(tf.multiply(pos_seq,hidden_seq),axis=-1)
        neg_=tf.reduce_sum(tf.multiply(neg_seq,hidden_seq),axis=-1)

        if mask!=None:
            mask_=tf.cast(mask,dtype=tf.float32)[:,:-1]
            pos_,neg_=pos_*mask_,neg_*mask_

        pos_neg_=tf.math.log_sigmoid(pos_/(neg_+1e-10))

        return  self.loss_lambda*tf.reduce_mean(tf.reduce_sum(pos_+neg_-pos_neg_,axis=1))


class TimeStreamLayer(tf.keras.layers.Layer):
    def __init__(self,ode_mode,sample_num=1,seed=2020,loss_lambda=0.5,trainable=True,supports_masking=True):
        super(TimeStreamLayer, self).__init__()
        self.lts=LatentTimeStreamLayer(ode_mode=ode_mode,seed=seed,trainable=trainable)
        self.decode=TimeDecodedLayer(sample_num=sample_num,seed=seed,loss_lambda=loss_lambda,trainable=trainable)
        self.trainable=trainable
        self.supports_masking=supports_masking


    def build(self, input_shape):
        super(TimeStreamLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        [t_inputs,init_states,behavior]=inputs
        hidden=self.lts([t_inputs,init_states])
        behavior,loss_=self.decode([behavior,hidden],mask=mask)

        return behavior,loss_

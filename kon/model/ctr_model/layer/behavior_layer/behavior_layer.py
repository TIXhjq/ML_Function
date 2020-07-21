#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/3 上午11:40
@File   :behavior_layer.py
================================='''
from numpy.random import random
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import os

from kon.model.ctr_model.layer.behavior_layer.rnn_demo import AUGRU
from kon.model.ctr_model.layer.core_layer.core_layer import *

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
        self.dnn=DnnLayer(hidden_units=hidden_units,output_dim=1,res_unit=10)
        self.softmax=tf.keras.layers.Activation('softmax')
        self.format=AlignLayer()
        self.dense=tf.keras.layers.Dense(1)

    def build(self, input_shape):
        super(AttentionUnitLayer, self).build(input_shape)

    def call(self, inputs, mask=None , **kwargs):
        [stack_candidate, stack_behavior] = inputs
        [stack_candidate, stack_behavior]=self.format([stack_candidate,stack_behavior])
        stack_candidate = tf.tile(stack_candidate, [1,stack_behavior.shape[1],1])
        activation_inputs = tf.concat([stack_candidate, stack_candidate - stack_behavior , stack_candidate * stack_behavior, stack_behavior],axis=-1)
        attention_weight = self.dnn(activation_inputs)
        attention_weight = tf.cast(attention_weight, tf.float32)
        if mask!=None:
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
        self.need_stack = need_stack
        self.return_seq=return_seq

    def build(self, input_shape):
        super(ActivationUnitLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if self.need_stack:
            inputs=[tf.concat(i,axis=-1) for i in inputs]
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
            from kon.model.ctr_model.layer import DnnLayer
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
    '''
        DTS timestream Module
            cover ODE+concat([behavior&dynamic user profile])
    '''
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


class ReadLayer(tf.keras.layers.Layer):
    '''
        NTM read
    '''
    def __init__(self,addressCal,read_head=3):
        super(ReadLayer, self).__init__()
        self.address = addressCal
        self.read_head=read_head

    def build(self, input_shape):
        super(ReadLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        [pre_w,M,read_head]=inputs
        w=self.address([pre_w,M,read_head])

        return tf.keras.backend.batch_dot(tf.transpose(w,[0,2,1]),M,axes=1),w


class WriteLayer(tf.keras.layers.Layer):
    '''
        NTM write
    '''
    def __init__(self,addressCal,write_head=3,seed=2020):
        super(WriteLayer, self).__init__()
        self.address = addressCal
        self.write_head=write_head
        self.seed=seed

    def build(self, input_shape):
        super(WriteLayer, self).build(input_shape)
        [pre_w, pre_M, write_head]=input_shape
        self.erase=self.add_weight(
            name='erase',shape=write_head,
            initializer=glorot_uniform(self.seed)
        )
        self.add=self.add_weight(
            name='add',shape=write_head,
            initializer=glorot_uniform(self.seed)
        )

    def call(self, inputs, **kwargs):
        [pre_w,pre_M,write_head]=inputs
        w=self.address([pre_w,pre_M,write_head])
        erase_w=tf.keras.backend.batch_dot(w,self.erase,axes=1)
        add_w=tf.keras.backend.batch_dot(w, self.add,axes=1)
        M_=pre_M*(1-erase_w)
        M=M_+add_w

        return M,w


class AddressCalLayer(tf.keras.layers.Layer):
    '''
        NTM attention<find address op>
    '''
    def __init__(self,seed=2020,shift_range=1):
        super(AddressCalLayer, self).__init__()
        self.seed=seed
        self.shift_range=shift_range

    def build(self, input_shape):
        super(AddressCalLayer, self).build(input_shape)
        [pre_w, M, k] = input_shape
        self.beta=self.add_weight(
            name='beta',shape=(k[1],M[1]),
            initializer=glorot_uniform(seed=self.seed)
        )
        self.g=self.add_weight(
            name='g',shape=(k[1],M[1]),
            initializer=glorot_uniform(seed=self.seed)
        )
        self.s=self.add_weight(
            name='s',shape=(),
            initializer=glorot_uniform(seed=self.seed)
        )
        # self.gamma=self.add_weight(
        #     name='gamma',shape=(),
        #     initializer=glorot_uniform(seed=self.seed)
        # )

    def call(self, inputs, **kwargs):
        [pre_w,M,k]=inputs
        #M=k:[mult_head,bs,slots,bits]
        M=tf.concat([tf.expand_dims(M,axis=1)for i in range(k.shape[1])],axis=1)
        k=tf.concat([tf.expand_dims(k,axis=2)for i in range(M.shape[2])],axis=2)
        content=tf.math.softmax(self.beta*tf.keras.losses.cosine_similarity(k,M))
        interpolation=self.g*content+(1-self.g)*pre_w
        w=interpolation
        # convolutional=tf.reduce_sum()
        # w=tf.reduce_mean(convolutional)

        return w


class ControlWrapLayer(tf.keras.layers.Layer):
    '''
        NTM coller WrapLayer,
            format input&output
    '''
    def __init__(self,controller_network,controller_input_flat):
        super(ControlWrapLayer, self).__init__()
        self.controller=controller_network
        self.controller_input_concat = StackLayer(use_flat=controller_input_flat)

    def build(self, input_shape):
        super(ControlWrapLayer, self).build(input_shape)
        [inputs,read_input,pre_read]=input_shape
        self.output_dim=inputs[-1]
        self.head_dim=read_input[-2]*read_input[-1]
        final_shape=self.output_dim+2*(self.head_dim)
        self.format_dense = tf.keras.layers.Dense(final_shape)

    def call(self, inputs, **kwargs):
        [inputs,read_input,pre_read]=inputs
        read_input = tf.reshape(read_input, shape=(read_input.shape[0], -1))
        control_inputs = self.controller_input_concat([inputs, read_input])

        control_output = self.controller(control_inputs)

        output,read_head,write_head=tf.split(self.format_dense(control_output),[self.output_dim,self.head_dim,self.head_dim],axis=1)
        output=tf.reshape(output,shape=inputs.shape)
        read_head=tf.reshape(read_head,shape=pre_read.shape)
        write_head=tf.reshape(write_head,shape=pre_read.shape)

        return [output,read_head,write_head]


class MemoryInductionUnitLayer(tf.keras.layers.Layer):
    '''
        MIN of MIMN,
        core:split channel-->mult channel
        p.s [M+behavior]*mask,but channel-memory not use reset
    '''
    def __init__(self,channel_num=5,supports_masking=True):
        super(MemoryInductionUnitLayer, self).__init__()
        self.k=channel_num
        self.supports_masking=supports_masking

    def build(self, input_shape):
        super(MemoryInductionUnitLayer, self).build(input_shape)
        [M, read_w, behavior, S] = input_shape
        self.gru=tf.keras.layers.GRU(units=S[-1],return_sequences=True)


    def call(self, inputs,mask=None, **kwargs):
        [M,read_w,behavior,S]=inputs
        vals,idx_=tf.math.top_k(tf.reduce_sum(read_w,axis=1),sorted=True,k=self.k)
        channel_inputs=tf.concat([M,S,tf.tile(tf.expand_dims(behavior,axis=1),[1,M.shape[1],1])],axis=-1)
        mask_=tf.tile(tf.expand_dims(tf.reduce_sum(tf.one_hot(idx_,M.shape[1]),axis=1),axis=2),[1,1,channel_inputs.shape[-1]])
        channel_inputs=channel_inputs*mask_
        channel_inputs=tf.concat([S,channel_inputs],axis=-1)
        S=self.gru(channel_inputs)

        return S

class RegLossLayer(tf.keras.layers.Layer):
    def __init__(self,reg_lambda):
        super(RegLossLayer, self).__init__()
        self.reg_lambda = reg_lambda

    def build(self, input_shape):
        super(RegLossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean_slot=tf.expand_dims(tf.reduce_mean(inputs,axis=-1),axis=-1)
        res_slot=inputs-tf.tile(mean_slot,[1,1,inputs.shape[-1]])
        res_slot=tf.multiply(res_slot,res_slot)
        reg_loss=self.reg_lambda*tf.losses.mean_squared_error(res_slot,res_slot)

        return reg_loss


class MemoryUtilizationRegLayer(tf.keras.layers.Layer):
    '''
        MIMN reduce hot-item write Method
    '''
    def __init__(self,reg_lambda=0.3,seed=2020):
        super(MemoryUtilizationRegLayer, self).__init__()
        self.reg_loss=RegLossLayer(reg_lambda)
        self.seed=seed

    def build(self, input_shape):
        super(MemoryUtilizationRegLayer, self).build(input_shape)
        [his_w, writeW] = input_shape
        self.Wg=self.add_weight(
            name='Wg',shape=writeW,
            initializer=glorot_uniform(seed=self.seed)
        )

    def call(self, inputs, **kwargs):
        [his_w, writeW]=inputs
        Pt=tf.math.softmax(self.Wg*his_w)
        reBalanceW=writeW*Pt
        loss=self.reg_loss(reBalanceW)

        return [reBalanceW,loss]


class UICLayer(tf.keras.layers.Layer):
    '''
        base struct:NTM
        add struct:MIU(topk interest extract)
        concat([NTM,MIU])==UIC
        p.s if not to use_miu==False:
                UIC=Naive UIC=NTM
    '''
    def __init__(self,controller_network=None,controller_input_flat=True,channel_dim=20,
                 memory_slots=128,memory_bits=20,mult_head=3,seed=2020,use_miu=True,
                 supports_masking=True,addressCal=AddressCalLayer(seed=2020),
                 use_memory_utilization_regularization=True, reg_lambda=0.3,
                 return_final_output=False,return_hidden=False,return_sequence=False):
        super(UICLayer, self).__init__()
        self.controller=ControlWrapLayer(controller_network,controller_input_flat)
        self.read_op=ReadLayer(addressCal=addressCal)
        self.write_op=WriteLayer(addressCal=addressCal)
        self.memory_slots=memory_slots
        self.memory_bits=memory_bits
        self.memory_head=mult_head
        self.seed=seed
        self.supports_masking=supports_masking
        self.channel_op=MemoryInductionUnitLayer()
        self.channel_dim=channel_dim
        self.use_miu=use_miu
        self.return_sequence=return_sequence
        self.return_hidden=return_hidden
        self.return_final_output=return_final_output
        self.use_reg=use_memory_utilization_regularization
        if use_memory_utilization_regularization:
            self.reg_method=MemoryUtilizationRegLayer(reg_lambda=reg_lambda)

    def build(self, input_shape):
        super(UICLayer, self).build(input_shape)
        bs=input_shape[0]
        self.M=self.add_weight(
            name='init_M',shape=(bs,self.memory_slots,self.memory_bits),
            initializer = glorot_uniform(self.seed)
        )
        self.init_read=self.add_weight(
            name='init_read',shape=(bs,self.memory_head,self.memory_bits),
            initializer = glorot_uniform(self.seed)
        )
        self.init_readW=self.add_weight(
            name='init_readW',shape=(bs,self.memory_head,self.memory_slots),
            initializer = glorot_uniform(self.seed)
        )
        self.init_writeW=self.add_weight(
            name='init_writeW',shape=(bs,self.memory_head,self.memory_slots),
            initializer=glorot_uniform(self.seed)
        )
        self.S=self.add_weight(
            name='init_S',shape=(bs,self.memory_slots,self.channel_dim),
            initializer=glorot_uniform(self.seed)
        )
        self.reg_loss = self.add_weight(
            name='init_loss',shape=(bs,self.memory_head),
            initializer=tf.keras.initializers.zeros()
        )


    def call(self, inputs, mask=None, **kwargs):
        def step(inputs, states):
            [M,pre_read,pre_readW,pre_writeW,S,pre_hisW,pre_loss]=states
            read_input,readW=self.read_op([pre_readW,M,pre_read])
            S=self.channel_op([M,readW,inputs,S])
            [output,read_head,write_head]=self.controller([inputs,read_input,pre_read])
            M,writeW=self.write_op([pre_writeW,M,write_head])
            if self.use_reg:
                [writeW,loss]=self.reg_method([pre_hisW,writeW])
                pre_hisW=pre_hisW+writeW
                loss+=pre_loss
            else:
                loss=pre_loss

            return output, [M,read_head,readW,writeW,S,pre_hisW,loss]

        output = tf.keras.backend.rnn(step, inputs, [self.M,self.init_read,self.init_readW,self.init_writeW,self.S,self.init_writeW,self.reg_loss], mask=mask)

        final_output = []
        sequence_output=[]
        final_hidden=[]

        if self.return_final_output:
            final_output=list(output[0])
        if self.return_sequence:
            sequence_output=list(output[1])
        if self.return_hidden:
            final_hidden=list(output[2])
            loss = final_hidden[-1]
            final_hidden[-1] = tf.reduce_mean(loss) / (inputs.shape[1] - 1)

        return final_output+sequence_output+final_hidden



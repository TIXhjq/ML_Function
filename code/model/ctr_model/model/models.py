#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/3 上午9:13
@File   :models.py
================================='''

import pandas as pd
import warnings
import os
from model.feature_transform import feature_tool
from model.base_model import base_model
from utils.data_prepare import data_prepare
from model.ctr_model.layer import *

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
prepare_tool=data_prepare()
#-----------------------------------------------------------------
def TestModel(spareInfo):
    [dense_inputs, spare_inputs, seq_inputs] = prepare_tool.df_prepare(spareInfo=spareInfo)
    cross_embed = SparseEmbed(spareInfo)(spare_inputs)
    input_=StackLayer()(cross_embed)
    dnn_output=DnnLayer(hidden_units=[100,10])(input_)
    output=MergeScoreLayer(use_merge=False)(dnn_output)
    return tf.keras.Model(dense_inputs+spare_inputs+seq_inputs,output)

def FM(spareInfo:list=None):
    [dense_inputs, spare_inputs,seq_inputs]= prepare_tool.df_prepare(spareInfo=spareInfo)
    cross_embed=SparseEmbed(spareInfo)(spare_inputs)
    linear=SparseEmbed(spareInfo,is_linear=True)(spare_inputs)
    embed_list=[cross_embed,linear]
    fm_=FmLayer()(embed_list)
    output=MergeScoreLayer(use_merge=False)(fm_)
    return tf.keras.Model(dense_inputs+spare_inputs,output)

def PNN(spareInfo:list=None,denseInfo:list=None,hidden_units=None,use_inner=True,use_outer=True):
    if hidden_units is None:
        hidden_units=[256,256,256]
    [dense_inputs, spare_inputs,seq_inputs]= prepare_tool.df_prepare(denseInfo=denseInfo, spareInfo=spareInfo)
    cross_embed = SparseEmbed(spareInfo,use_flatten=False)(spare_inputs)
    linear = SparseEmbed(spareInfo, is_linear=True)(spare_inputs)
    cross_fea=linear

    if use_inner:
        cross_fea+=IPnnLayer()(cross_embed)
    if use_outer:
        cross_fea+=OPnnLayer()(cross_embed)
    cross_fea=StackLayer()(cross_fea)
    dnn_ = DnnLayer(hidden_units)(cross_fea)
    output=MergeScoreLayer(use_merge=False)(dnn_)

    return tf.keras.Model(spare_inputs,output)

def DeepCross(spareInfo:list=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 256, 256]
        [dense_inputs, spare_inputs,seq_inputs] = prepare_tool.df_prepare(spareInfo=spareInfo)
        cross_embed = SparseEmbed(spareInfo,use_flatten=True)(spare_inputs)
        dnn_inputs=StackLayer()(dense_inputs+cross_embed)

        dnn_fea=DnnLayer(hidden_units=hidden_units)(dnn_inputs)
        output=MergeScoreLayer(use_merge=False)(dnn_fea)

        return tf.keras.Model(spare_inputs,output)

def Wide_Deep(spareInfo:list=None, denseInfo:list=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, spare_inputs,seq_inputs]= prepare_tool.df_prepare(denseInfo=denseInfo, spareInfo=spareInfo)
    cross_ = SparseEmbed(spareInfo,use_flatten=True)(spare_inputs)
    linear_ = SparseEmbed(spareInfo, is_linear=True)(spare_inputs)
    dnn_inputs=StackLayer()(dense_inputs+cross_)

    dnn_=DnnLayer(hidden_units=hidden_units)(dnn_inputs)
    output=MergeScoreLayer()(linear_+[dnn_])

    return tf.keras.Model(dense_inputs+spare_inputs,output)

def DeepFM(spareInfo:list=None, denseInfo:list=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, spare_inputs,seq_inputs] = prepare_tool.df_prepare(denseInfo=denseInfo, spareInfo=spareInfo)
    cross_embed = SparseEmbed(spareInfo)(spare_inputs)
    linear = SparseEmbed(spareInfo, is_linear=True)(spare_inputs)
    embed_list = [cross_embed, linear]

    fm_=FmLayer()(embed_list)
    dnn_input=StackLayer()(dense_inputs+cross_embed)
    dnn_ = DnnLayer(hidden_units=hidden_units)(dnn_input)
    output = MergeScoreLayer()([fm_, dnn_])

    return tf.keras.Model(dense_inputs+spare_inputs, output)

def DCN(spareInfo:list=None,denseInfo:list=None,hidden_units=None,cross_hidden=3):
    '''
        Notice:
            cross_hidden==> iter_num(x^k=w(x^k-1*x0)+b+x0)
    '''
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, spare_inputs,seq_inputs]= prepare_tool.df_prepare(denseInfo=denseInfo, spareInfo=spareInfo)
    cross_embed = SparseEmbed(spareInfo,use_flatten=False)(spare_inputs)
    combine_inputs=StackLayer()(dense_inputs+cross_embed)

    cross_fea=CrossLayer(cross_hidden=cross_hidden)(combine_inputs)
    deep_fea=DnnLayer(hidden_units=hidden_units)(combine_inputs)
    output=MergeScoreLayer()([cross_fea,deep_fea])

    return tf.keras.Model(dense_inputs+spare_inputs,output)

def NFM(spareInfo:list=None,hidden_units=None,denseInfo:list=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, spare_inputs,seq_inputs] = prepare_tool.df_prepare(
        spareInfo=spareInfo,denseInfo=denseInfo)
    cross_embed = SparseEmbed(spareInfo, use_flatten=True)(spare_inputs)
    linear_embed = SparseEmbed(spareInfo,is_linear=True)(spare_inputs)

    cross_inputs=InnerLayer(use_inner=True,use_add=True)(cross_embed)
    dnn_inputs=StackLayer()(dense_inputs+[cross_inputs])

    dnn_fea=DnnLayer(hidden_units=hidden_units,output_dim=1)(dnn_inputs)
    final_fea=tf.keras.layers.Add()(linear_embed+[dnn_fea])
    output=ScoreLayer()(final_fea)

    return tf.keras.Model(dense_inputs+spare_inputs,output)

def XDeepFM(spareInfo:list=None, denseInfo:list=None, conv_size=None, hidden_units=None):
    '''
    :param conv_size:
        notice:conv_size decision Hk<size>
    '''
    if conv_size is None:
        conv_size = [200, 200, 200]
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, spare_inputs,seq_inputs] = prepare_tool.df_prepare(
        spareInfo=spareInfo,denseInfo=denseInfo)
    cross_embed = SparseEmbed(spareInfo, use_flatten=False)(spare_inputs)
    cin_inputs=tf.keras.layers.Concatenate(axis=1)(cross_embed)
    dnn_inputs=StackLayer()(dense_inputs+cross_embed)

    linear_output = SparseEmbed(
        spareInfo, use_flatten=True, is_linear=True, use_add=True)(spare_inputs)
    cin_output=CIN(conv_size=conv_size,output_dim=1)(cin_inputs)
    dnn_output=DnnLayer(hidden_units=hidden_units,output_dim=1)(dnn_inputs)
    output=ScoreLayer()(tf.keras.layers.Add()([linear_output,cin_output,dnn_output]))

    return tf.keras.Model(dense_inputs+spare_inputs,output)


def AFM(spareInfo:list=None,denseInfo:list=None):
    [dense_inputs, spare_inputs,seq_inputs] = prepare_tool.df_prepare(
        spareInfo=spareInfo, denseInfo=denseInfo)
    cross_embed = SparseEmbed(spareInfo, use_flatten=False)(spare_inputs)
    linear_embed = SparseEmbed(spareInfo, use_flatten=True,is_linear=True)(spare_inputs)

    cross_output=InnerLayer()(cross_embed)
    atten_output=AttentionBaseLayer()(cross_output)
    output=ScoreLayer(use_add=True)(linear_embed+[atten_output])

    return tf.keras.Model(dense_inputs+spare_inputs,output)


def AutoInt(spareInfo:list=None,denseInfo:list=None,attention_dim=8,attention_head_dim=3):
    '''
        notice:
            origin inputs=[dense+sparse],now inputs=[sparse]
            MultHeadAttentionLayer !support Bn&Add&Activate,
            because now want to as hidden of DnnLayer,update at soon...
        core:
            multHead to replace inner of fm
    '''
    [dense_inputs, spare_inputs, seq_inputs] = prepare_tool.df_prepare(
        spareInfo=spareInfo, denseInfo=denseInfo)
    cross_embed = StackLayer(use_flat=False,axis=1)(SparseEmbed(spareInfo, use_flatten=False)(spare_inputs))
    atten_layer=MultHeadAttentionLayer(attention_dim=attention_dim,attention_head_dim=attention_head_dim,use_ln=True,atten_mask_mod=1)
    atten_vec=DnnLayer(res_unit=1,other_dense=[atten_layer])(cross_embed)
    final_input=StackLayer(use_flat=True,axis=-1)([tf.squeeze(i,0) for i in tf.split(atten_vec,[1]*atten_vec.shape[0])])
    output=MergeScoreLayer(use_merge=False)(final_input)

    return tf.keras.Model(dense_inputs+spare_inputs+seq_inputs,output)


def DIN(spareInfo:list=None, denseInfo:list=None, seqInfo:list=None, candidateFea=None, behaviorFea=None,
        attention_units=None, hidden_units=None):
    '''
        Notice:
        about data:
            build candidate need appear behavior,
            about cold start ==> not give method
        about candidate&behavior:
            must item appear behavior==>attention is useful
            because core is find useful seq_item,to metric seq_items
            paper view is all_seq===find===>nextClick relate part_item_seq to improve score
            not to find seq deal info
            find history seq mod===>activate new seq
        about Dice:
            !achieve because DnnLayer have bn
                ==>only use PRelu
    '''
    if hidden_units is None:
        hidden_units = [256, 256, 256]
    if attention_units is None:
        attention_units = [100, 64, 32]
    [dense_inputs, spare_inputs,seq_inputs] = prepare_tool.df_prepare(
        spareInfo=spareInfo, denseInfo=denseInfo ,seqInfo=seqInfo)
    cross_embed = SparseEmbed(spareInfo, use_flatten=False)(spare_inputs)
    seq_embed,behavior_mask= SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=True)(seq_inputs)
    candidate_embed=ExtractLayer(candidateFea,spare_inputs)(cross_embed)
    behavior_embed=ExtractLayer(behaviorFea,seq_inputs,mask_zero=True)(seq_embed)

    base_behavior=SeqBaseLayer()(seq_embed)
    attention_behavior=ActivationUnitLayer(hidden_units=attention_units)([candidate_embed,behavior_embed],mask=behavior_mask[0])
    final_inputs=StackLayer(use_flat=True)(base_behavior+[attention_behavior])

    mlp_output=DnnLayer(hidden_units=hidden_units,hidden_activate=tf.keras.layers.PReLU(),res_unit=2,use_bn=True)(final_inputs)
    output=MergeScoreLayer(use_merge=False)(mlp_output)

    return tf.keras.Model(dense_inputs+spare_inputs+seq_inputs,output)

def DIEN(denseInfo:list=None, spareInfo:list=None, seqInfo:list=None,
         candidateFea=None, behaviorFea=None, classify_units=None, hidden_units=None,
         attention_units=None,sample_num=5):
    '''
    notice:
        at auxloss not use Dice as Dnnlayer Activate...
        because i think sample seq to BN may be...
        In fact,default param:attenion&auxLoss size not support BN,too
    :param classify_units: AuxLoss classify==Dnnlayer ?please input hidden size:ignore
    :param hidden_units: final classify dnnlayer hidden size
    :param attention_units: attention classify dnnlayer hiddensize of ActivateUnits
    :param sample_num: num of dien nega sample item
    '''
    if attention_units is None:
        attention_units = [100, 64, 32]
    if classify_units is None:
        classify_units = [100, 64, 32]
    if hidden_units is None:
        hidden_units = [256, 256, 256]

    [dense_inputs, spare_inputs, seq_inputs] = prepare_tool.df_prepare(spareInfo=spareInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    cross_embed = SparseEmbed(spareInfo, use_flatten=False)(spare_inputs)
    seq_embed,seq_mask= SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=True)(seq_inputs)

    behavior_embed=StackLayer(use_flat=False)(ExtractLayer(behaviorFea,seq_inputs,mask_zero=True)(seq_embed))
    candidate_embed=StackLayer(use_flat=False)(ExtractLayer(candidateFea,spare_inputs)(cross_embed))

    behavior_sample = SampleLayer(sample_num=sample_num)(behavior_embed)
    [hidden_list,aux_loss]=InterestExtratorLayer(classify_units=classify_units,sample_num=sample_num)([behavior_embed,behavior_sample],mask=seq_mask[0])
    final_hidden=InterestEolvingLayer(attention_units=attention_units)([candidate_embed,hidden_list],mask=seq_mask[0])

    final_input=StackLayer()([final_hidden]+cross_embed)
    output=DnnLayer(hidden_units=hidden_units,hidden_activate=tf.keras.layers.PReLU())(final_input)
    output=MergeScoreLayer(use_merge=False,output_dim=2)(output)

    model=tf.keras.Model(dense_inputs + spare_inputs + seq_inputs, output)
    model.add_loss(aux_loss)
    return model

def DSIN(denseInfo:list=None, spareInfo:list=None, seqInfo:list=None,
         candidateFea=None, behaviorFea=None,attention_dim=8,attention_head_dim=5,ffn_hidden_unit=10,
         lstm_units=8,lstm_mode='sum',attention_units=None,classify_units=None,sessionMaxLen=10,sessionMaxNum=20):

    if attention_units is None:
        attention_units = [100, 64, 32]
    if classify_units is None:
        classify_units = [100, 64, 32]

    [dense_inputs, spare_inputs, seq_inputs] = prepare_tool.df_prepare(spareInfo=spareInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    spare_embed = SparseEmbed(spareInfo, use_flatten=False)(spare_inputs)
    seq_embed,seq_mask= SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=True)(seq_inputs)

    behavior_embed=StackLayer(use_flat=False)(ExtractLayer(behaviorFea,seq_inputs,mask_zero=True)(seq_embed))
    candidate_embed=StackLayer(use_flat=False)(ExtractLayer(candidateFea,spare_inputs)(spare_embed))

    pos_behavior=SessionDivisonLayer(sessionMaxLen=sessionMaxLen,sessionMaxNum=sessionMaxNum)(behavior_embed)
    self_behavior=SessionInterestExtractorLayer(attention_dim=attention_dim,attention_head_dim=attention_head_dim,ffn_hidden_unit=ffn_hidden_unit)(pos_behavior)

    self_atten=ActivationUnitLayer(attention_units,need_stack=False)([candidate_embed,self_behavior],mask=seq_mask[0])
    hidden_behavior=SessionInterestInteractingLayer(biLstmUnit=lstm_units,lstm_mode=lstm_mode)(self_behavior)
    lstm_atten=ActivationUnitLayer(attention_units,need_stack=False)([candidate_embed,hidden_behavior],mask=seq_mask[0])
    dnn_inputs=StackLayer(use_flat=True)(spare_embed+dense_inputs+[self_atten,lstm_atten])
    output=DnnLayer(hidden_units=classify_units,use_bn=True,res_unit=2)(dnn_inputs)
    output=MergeScoreLayer(use_merge=False,output_dim=2)(output)

    return tf.keras.Model(dense_inputs+spare_inputs+seq_inputs,output)

def SeqFM(denseInfo:list=None, spareInfo:list=None, seqInfo:list=None,hidden_units=None,res_unit=1,
          atten_dim=8,atten_head=1):
    def DynamicViewMask(seq_shape):
        max_len = seq_shape[1]
        dynamic_mask = tf.convert_to_tensor([[
            1.0 if i < j else 0.0 for j in range(max_len)
        ] for i in range(max_len)])
        seq_mask = tf.equal(dynamic_mask, 0)

        return seq_mask

    def CrossViewMask(cross_inputs:list):
        max_len,spare_len = cross_inputs[0].shape[1],cross_inputs[1].shape[1]
        m = spare_len - 1
        cross_mask = tf.convert_to_tensor([
        [0.0 if (i <= m and j > m) or (j <= m and i > m) else 1.0
        for j in range(max_len + spare_len)] for i in range(max_len + spare_len)])

        cross_mask = tf.equal(cross_mask, 0)
        inputs = tf.concat(cross_inputs, axis=1)

        return inputs, cross_mask

    if hidden_units is None:
        hidden_units = [atten_dim]*2

    [dense_inputs, spare_inputs, seq_inputs] = prepare_tool.df_prepare(spareInfo=spareInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    linear = SparseEmbed(spareInfo, is_linear=True)(spare_inputs)
    spare_embed = StackLayer(axis=1,use_flat=False)(SparseEmbed(spareInfo, use_flatten=False)(spare_inputs))
    seq_embed= StackLayer(use_flat=False)(SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=False)(seq_inputs))

    spare_atten=MultHeadAttentionLayer(attention_dim=atten_dim,attention_head_dim=atten_head)(spare_embed)
    spare_view=IntraViewPoolingLayer()(spare_atten)

    seq_mask=DynamicViewMask(seq_embed.shape)
    seq_atten=MultHeadAttentionLayer(attention_dim=atten_dim,attention_head_dim=atten_head,atten_mask_mod=2)(seq_embed,mask=seq_mask)
    seq_view=IntraViewPoolingLayer()(seq_atten)

    cross_inputs,cross_mask=CrossViewMask([seq_embed,spare_embed])
    cross_atten=MultHeadAttentionLayer(attention_dim=atten_dim,attention_head_dim=atten_head,atten_mask_mod=2)(cross_inputs,mask=cross_mask)
    cross_view=IntraViewPoolingLayer()(cross_atten)

    ffn_inputs=StackLayer(use_flat=False,axis=1)([spare_view,seq_view,cross_view])
    ffn_output=DnnLayer(hidden_units=hidden_units,use_flatten=True,use_ln=True,use_bn=False,res_unit=res_unit)(ffn_inputs)

    output=MergeScoreLayer(use_merge=True,output_dim=2)(linear+[ffn_output])
    # output=ScoreLayer(use_add=True,use_global=True)(linear + [ffn_output])

    return tf.keras.Model(dense_inputs+spare_inputs+seq_inputs,output)


def DTSF(denseInfo: list = None, spareInfo: list = None, seqInfo: list = None):
    [dense_inputs, spare_inputs, seq_inputs] = prepare_tool.df_prepare(spareInfo=spareInfo, denseInfo=denseInfo,
                                                                       seqInfo=seqInfo)
    spare_embed = SparseEmbed(spareInfo, use_flatten=False)(spare_inputs)



def DSTN(denseInfo:list=None, spareInfo:list=None, seqInfo:list=None):
    [dense_inputs, spare_inputs, seq_inputs] = prepare_tool.df_prepare(spareInfo=spareInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    spare_embed = SparseEmbed(spareInfo, use_flatten=False)(spare_inputs)





    

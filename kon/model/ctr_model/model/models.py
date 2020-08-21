#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/3 上午9:13
@File   :models.py
================================='''
from kon.utils.data_prepare import data_prepare
from kon.model.ctr_model.layer.behavior_layer.behavior_layer import *
from kon.model.ctr_model.layer.interactive_layer.interactive_layer import *
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
# model_tool=base_model(submit_data_folder)
# fea_tool=feature_tool(fea_data_folder)
prepare_tool=data_prepare()
#-----------------------------------------------------------------

def TestModel(sparseInfo):
    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo)
    cross_embed = SparseEmbed(sparseInfo)(sparse_inputs)
    input_=StackLayer()(cross_embed)
    dnn_output=DnnLayer(hidden_units=[100,10])(input_)
    output=MergeScoreLayer(use_merge=False)(dnn_output)
    return tf.keras.Model(dense_inputs+sparse_inputs+seq_inputs,output)

def FM(sparseInfo:list=None):
    [dense_inputs, sparse_inputs,seq_inputs]= prepare_tool.df_prepare(sparseInfo=sparseInfo)
    cross_embed=SparseEmbed(sparseInfo)(sparse_inputs)
    linear=SparseEmbed(sparseInfo,is_linear=True)(sparse_inputs)
    embed_list=[cross_embed,linear]
    fm_=FmLayer()(embed_list)
    output=MergeScoreLayer(use_merge=False)(fm_)
    return tf.keras.Model(dense_inputs+sparse_inputs,output)

def PNN(sparseInfo:list=None,denseInfo:list=None,hidden_units=None,use_inner=True,use_outer=True):
    if hidden_units is None:
        hidden_units=[256,256,256]
    [dense_inputs, sparse_inputs,seq_inputs]= prepare_tool.df_prepare(denseInfo=denseInfo, sparseInfo=sparseInfo)
    cross_embed = SparseEmbed(sparseInfo,use_flatten=False)(sparse_inputs)
    linear = SparseEmbed(sparseInfo, is_linear=True)(sparse_inputs)
    cross_fea=linear

    if use_inner:
        cross_fea+=IPnnLayer()(cross_embed)
    if use_outer:
        cross_fea+=OPnnLayer()(cross_embed)
    cross_fea=StackLayer()(cross_fea)
    dnn_ = DnnLayer(hidden_units)(cross_fea)
    output=MergeScoreLayer(use_merge=False)(dnn_)

    return tf.keras.Model(sparse_inputs,output)

def DeepCross(sparseInfo:list=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 256, 256]
        [dense_inputs, sparse_inputs,seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo)
        cross_embed = SparseEmbed(sparseInfo,use_flatten=True)(sparse_inputs)
        dnn_inputs=StackLayer()(dense_inputs+cross_embed)

        dnn_fea=DnnLayer(hidden_units=hidden_units)(dnn_inputs)
        output=MergeScoreLayer(use_merge=False)(dnn_fea)

        return tf.keras.Model(sparse_inputs,output)

def Wide_Deep(sparseInfo:list=None, denseInfo:list=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, sparse_inputs,seq_inputs]= prepare_tool.df_prepare(denseInfo=denseInfo, sparseInfo=sparseInfo)
    cross_ = SparseEmbed(sparseInfo,use_flatten=True)(sparse_inputs)
    linear_ = SparseEmbed(sparseInfo, is_linear=True)(sparse_inputs)
    dnn_inputs=StackLayer()(dense_inputs+cross_)

    dnn_=DnnLayer(hidden_units=hidden_units)(dnn_inputs)
    output=MergeScoreLayer()(linear_+[dnn_])

    return tf.keras.Model(dense_inputs+sparse_inputs,output)

def DeepFM(sparseInfo:list=None, denseInfo:list=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, sparse_inputs,seq_inputs] = prepare_tool.df_prepare(denseInfo=denseInfo, sparseInfo=sparseInfo)
    cross_embed = SparseEmbed(sparseInfo)(sparse_inputs)
    linear = SparseEmbed(sparseInfo, is_linear=True)(sparse_inputs)
    embed_list = [cross_embed, linear]

    fm_=FmLayer()(embed_list)
    dnn_input=StackLayer()(dense_inputs+cross_embed)
    dnn_ = DnnLayer(hidden_units=hidden_units)(dnn_input)
    output = MergeScoreLayer()([fm_, dnn_])

    return tf.keras.Model(dense_inputs+sparse_inputs, output)

def DCN(sparseInfo:list=None,denseInfo:list=None,hidden_units=None,cross_hidden=3):
    '''
        Notice:
            cross_hidden==> iter_num(x^k=w(x^k-1*x0)+b+x0)
    '''
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, sparse_inputs,seq_inputs]= prepare_tool.df_prepare(denseInfo=denseInfo, sparseInfo=sparseInfo)
    cross_embed = SparseEmbed(sparseInfo,use_flatten=False)(sparse_inputs)
    combine_inputs=StackLayer()(dense_inputs+cross_embed)

    cross_fea=CrossLayer(cross_hidden=cross_hidden)(combine_inputs)
    deep_fea=DnnLayer(hidden_units=hidden_units)(combine_inputs)
    output=MergeScoreLayer()([cross_fea,deep_fea])

    return tf.keras.Model(dense_inputs+sparse_inputs,output)

def NFM(sparseInfo:list=None,hidden_units=None,denseInfo:list=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, sparse_inputs,seq_inputs] = prepare_tool.df_prepare(
        sparseInfo=sparseInfo,denseInfo=denseInfo)
    cross_embed = SparseEmbed(sparseInfo, use_flatten=True)(sparse_inputs)
    linear_embed = SparseEmbed(sparseInfo,is_linear=True)(sparse_inputs)

    cross_inputs=InnerLayer(use_inner=True,use_add=True)(cross_embed)
    dnn_inputs=StackLayer()(dense_inputs+[cross_inputs])

    dnn_fea=DnnLayer(hidden_units=hidden_units,output_dim=1)(dnn_inputs)
    final_fea=tf.keras.layers.Add()(linear_embed+[dnn_fea])
    output=ScoreLayer()(final_fea)

    return tf.keras.Model(dense_inputs+sparse_inputs,output)

def XDeepFM(sparseInfo:list=None, denseInfo:list=None, conv_size=None, hidden_units=None):
    '''
    :param conv_size:
        notice:conv_size decision Hk<size>
    '''
    if conv_size is None:
        conv_size = [200, 200, 200]
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    [dense_inputs, sparse_inputs,seq_inputs] = prepare_tool.df_prepare(
        sparseInfo=sparseInfo,denseInfo=denseInfo)
    cross_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)
    cin_inputs=tf.keras.layers.Concatenate(axis=1)(cross_embed)
    dnn_inputs=StackLayer()(dense_inputs+cross_embed)

    linear_output = SparseEmbed(
        sparseInfo, use_flatten=True, is_linear=True, use_add=True)(sparse_inputs)
    cin_output=CIN(conv_size=conv_size,output_dim=1)(cin_inputs)
    dnn_output=DnnLayer(hidden_units=hidden_units,output_dim=1)(dnn_inputs)
    output=ScoreLayer()(tf.keras.layers.Add()([linear_output,cin_output,dnn_output]))

    return tf.keras.Model(dense_inputs+sparse_inputs,output)


def AFM(sparseInfo:list=None,denseInfo:list=None):
    [dense_inputs, sparse_inputs,seq_inputs] = prepare_tool.df_prepare(
        sparseInfo=sparseInfo, denseInfo=denseInfo)
    cross_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)
    linear_embed = SparseEmbed(sparseInfo, use_flatten=True,is_linear=True)(sparse_inputs)

    cross_output=InnerLayer()(cross_embed)
    atten_output=AttentionBaseLayer()(cross_output)
    output=ScoreLayer(use_add=True)(linear_embed+[atten_output])

    return tf.keras.Model(dense_inputs+sparse_inputs,output)


def AutoInt(sparseInfo:list=None,denseInfo:list=None,attention_dim=8,attention_head_dim=3):
    '''
        notice:
            origin inputs=[dense+sparse],now inputs=[sparse]
            MultHeadAttentionLayer !support Bn&Add&Activate,
            because now want to as hidden of DnnLayer,update at soon...
        core:
            multHead to replace inner of fm
    '''
    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(
        sparseInfo=sparseInfo, denseInfo=denseInfo)
    cross_embed = StackLayer(use_flat=False,axis=1)(SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs))
    atten_layer=MultHeadAttentionLayer(attention_dim=attention_dim,attention_head_dim=attention_head_dim,use_ln=True,atten_mask_mod=1)
    atten_vec=DnnLayer(res_unit=1,other_dense=[atten_layer])(cross_embed)
    final_input=StackLayer(use_flat=True,axis=-1)([tf.squeeze(i,0) for i in tf.split(atten_vec,[1]*atten_vec.shape[0])])
    output=MergeScoreLayer(use_merge=False)(final_input)

    return tf.keras.Model(dense_inputs+sparse_inputs+seq_inputs,output)


def DIN(sparseInfo:list=None, denseInfo:list=None, seqInfo:list=None, candidateFea=None, behaviorFea=None,
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
    [dense_inputs, sparse_inputs,seq_inputs] = prepare_tool.df_prepare(
        sparseInfo=sparseInfo, denseInfo=denseInfo ,seqInfo=seqInfo)
    cross_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)
    seq_embed,behavior_mask= SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=True)(seq_inputs)
    candidate_embed=ExtractLayer(candidateFea,sparse_inputs)(cross_embed)
    behavior_embed=ExtractLayer(behaviorFea,seq_inputs,mask_zero=True)(seq_embed)

    base_behavior=SeqBaseLayer()(seq_embed)
    attention_behavior=ActivationUnitLayer(hidden_units=attention_units)([candidate_embed,behavior_embed],mask=behavior_mask[0])
    final_inputs=StackLayer(use_flat=True)(base_behavior+[attention_behavior])

    mlp_output=DnnLayer(hidden_units=hidden_units,hidden_activate=tf.keras.layers.PReLU(),res_unit=2,use_bn=True)(final_inputs)
    output=MergeScoreLayer(use_merge=False)(mlp_output)

    return tf.keras.Model(dense_inputs+sparse_inputs+seq_inputs,output)

def DIEN(denseInfo:list=None, sparseInfo:list=None, seqInfo:list=None,
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

    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    cross_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)
    seq_embed,seq_mask= SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=True)(seq_inputs)

    behavior_embed=StackLayer(use_flat=False)(ExtractLayer(behaviorFea,seq_inputs,mask_zero=True)(seq_embed))
    candidate_embed=StackLayer(use_flat=False)(ExtractLayer(candidateFea,sparse_inputs)(cross_embed))

    behavior_sample = SampleLayer(sample_num=sample_num)(behavior_embed)
    [hidden_list,aux_loss]=InterestExtratorLayer(classify_units=classify_units,sample_num=sample_num)([behavior_embed,behavior_sample],mask=seq_mask[0])
    final_hidden=InterestEolvingLayer(attention_units=attention_units)([candidate_embed,hidden_list],mask=seq_mask[0])

    final_input=StackLayer()([final_hidden]+cross_embed)
    output=DnnLayer(hidden_units=hidden_units,hidden_activate=tf.keras.layers.PReLU())(final_input)
    output=MergeScoreLayer(use_merge=False,output_dim=2)(output)

    model=tf.keras.Model(dense_inputs + sparse_inputs + seq_inputs, output)
    model.add_loss(aux_loss)
    return model

def DSIN(denseInfo:list=None, sparseInfo:list=None, seqInfo:list=None,
         candidateFea=None, behaviorFea=None,attention_dim=8,attention_head_dim=5,ffn_hidden_unit=10,
         lstm_units=8,lstm_mode='sum',attention_units=None,classify_units=None,sessionMaxLen=10,sessionMaxNum=20):

    if attention_units is None:
        attention_units = [100, 64, 32]
    if classify_units is None:
        classify_units = [100, 64, 32]

    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    sparse_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)
    seq_embed,seq_mask= SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=True)(seq_inputs)

    behavior_embed=StackLayer(use_flat=False)(ExtractLayer(behaviorFea,seq_inputs,mask_zero=True)(seq_embed))
    candidate_embed=StackLayer(use_flat=False)(ExtractLayer(candidateFea,sparse_inputs)(sparse_embed))

    pos_behavior=SessionDivisonLayer(sessionMaxLen=sessionMaxLen,sessionMaxNum=sessionMaxNum)(behavior_embed)
    self_behavior=SessionInterestExtractorLayer(attention_dim=attention_dim,attention_head_dim=attention_head_dim,ffn_hidden_unit=ffn_hidden_unit)(pos_behavior)

    self_atten=ActivationUnitLayer(attention_units,need_stack=False)([candidate_embed,self_behavior],mask=seq_mask[0])
    hidden_behavior=SessionInterestInteractingLayer(biLstmUnit=lstm_units,lstm_mode=lstm_mode)(self_behavior)
    lstm_atten=ActivationUnitLayer(attention_units,need_stack=False)([candidate_embed,hidden_behavior],mask=seq_mask[0])
    dnn_inputs=StackLayer(use_flat=True)(sparse_embed+dense_inputs+[self_atten,lstm_atten])
    output=DnnLayer(hidden_units=classify_units,use_bn=True,res_unit=2)(dnn_inputs)
    output=MergeScoreLayer(use_merge=False,output_dim=2)(output)

    return tf.keras.Model(dense_inputs+sparse_inputs+seq_inputs,output)


def SeqFM(denseInfo:list=None, sparseInfo:list=None, seqInfo:list=None,hidden_units=None,res_unit=1,
          atten_dim=8,atten_head=1):
    def DynamicViewMask(seq_shape):
        max_len = seq_shape[1]
        dynamic_mask = tf.convert_to_tensor([[
            1.0 if i < j else 0.0 for j in range(max_len)
        ] for i in range(max_len)])
        seq_mask = tf.equal(dynamic_mask, 0)

        return seq_mask

    def CrossViewMask(cross_inputs:list):
        max_len,sparse_len = cross_inputs[0].shape[1],cross_inputs[1].shape[1]
        m = sparse_len - 1
        cross_mask = tf.convert_to_tensor([
        [0.0 if (i <= m and j > m) or (j <= m and i > m) else 1.0
        for j in range(max_len + sparse_len)] for i in range(max_len + sparse_len)])

        cross_mask = tf.equal(cross_mask, 0)
        inputs = tf.concat(cross_inputs, axis=1)

        return inputs, cross_mask

    if hidden_units is None:
        hidden_units = [atten_dim]*2

    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    linear = SparseEmbed(sparseInfo, is_linear=True)(sparse_inputs)
    sparse_embed = StackLayer(axis=1,use_flat=False)(SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs))
    seq_embed= StackLayer(use_flat=False)(SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=False)(seq_inputs))

    sparse_atten=MultHeadAttentionLayer(attention_dim=atten_dim,attention_head_dim=atten_head)(sparse_embed)
    sparse_view=IntraViewPoolingLayer()(sparse_atten)

    seq_mask=DynamicViewMask(seq_embed.shape)
    seq_atten=MultHeadAttentionLayer(attention_dim=atten_dim,attention_head_dim=atten_head,atten_mask_mod=2)(seq_embed,mask=seq_mask)
    seq_view=IntraViewPoolingLayer()(seq_atten)

    cross_inputs,cross_mask=CrossViewMask([seq_embed,sparse_embed])
    cross_atten=MultHeadAttentionLayer(attention_dim=atten_dim,attention_head_dim=atten_head,atten_mask_mod=2)(cross_inputs,mask=cross_mask)
    cross_view=IntraViewPoolingLayer()(cross_atten)

    ffn_inputs=StackLayer(use_flat=False,axis=1)([sparse_view,seq_view,cross_view])
    ffn_output=DnnLayer(hidden_units=hidden_units,use_flatten=True,use_ln=True,use_bn=False,res_unit=res_unit)(ffn_inputs)

    output=MergeScoreLayer(use_merge=True,output_dim=2)(linear+[ffn_output])
    # output=ScoreLayer(use_add=True,use_global=True)(linear + [ffn_output])

    return tf.keras.Model(dense_inputs+sparse_inputs+seq_inputs,output)


def DTS(denseInfo: list = None, sparseInfo: list = None, seqInfo: list = None,
         userFea:list=None,timestampFea:list=None,behaviorFea:list=None,targetFea:list=None,
         ode_mode=1,sample_num=1,is_train=True,loss_lambda:int=0.5):

    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    sparse_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)
    seq_embed,seq_mask= SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=True)(seq_inputs)

    timestampEmbed = tf.expand_dims(StackLayer(use_flat=False)(ExtractLayer(timestampFea, seq_inputs)(seq_embed)),axis=-1)
    behaviorEmbed = StackLayer(use_flat=False)(ExtractLayer(behaviorFea, seq_inputs)(seq_embed))
    userEmbed = tf.squeeze(StackLayer(use_flat=False)(ExtractLayer(userFea, sparse_inputs)(sparse_embed)),axis=1)
    targetEmbed,sparse_embed=ExtractLayer(targetFea, sparse_inputs,need_remove=True)(sparse_embed)
    behaviorEmbed=[behaviorEmbed,targetEmbed]

    behavior,loss_=TimeStreamLayer(sample_num=sample_num,ode_mode=ode_mode,trainable=is_train,loss_lambda=loss_lambda)([timestampEmbed,userEmbed,behaviorEmbed],mask=seq_mask[0])
    behavior,targetItem=behavior[0],behavior[1]

    behavior=tf.reduce_mean(behavior,axis=1)
    dnn_input=StackLayer(use_flat=True)([behavior]+sparse_embed)
    dnn_output=DnnLayer(hidden_units=[64,32])(dnn_input)
    output=MergeScoreLayer(use_merge=False)(dnn_output)

    model=tf.keras.Model(dense_inputs+sparse_inputs+seq_inputs,output)
    model.add_loss(loss_)

    return model


def BST(denseInfo: list = None, sparseInfo: list = None, seqInfo: list = None,behaviorFea=None,
        attention_units=8,hidden_units=None,ffn_hidden_unit=8,attention_head=3):
    if hidden_units is None:
        hidden_units = [100, 64, 32]

    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    sparse_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)
    seq_embed,seq_mask= SparseEmbed(seqInfo, use_flatten=False, is_linear=False,mask_zero=True)(seq_inputs)

    behaviorEmb=StackLayer(use_flat=False,axis=1)(ExtractLayer(behaviorFea,seq_inputs)(seq_embed))
    transformerFea=[SelfAttentionLayer(attention_dim=attention_units,attention_head_dim=attention_head,ffn_hidden_unit=ffn_hidden_unit)(emb_)
        for emb_ in tf.split(behaviorEmb,[2]*(behaviorEmb.shape[1]//2),axis=1)]

    dnn_inputs=StackLayer(axis=-1)(sparse_embed+transformerFea)
    dnn_output=DnnLayer(hidden_units=hidden_units)(dnn_inputs)
    output=MergeScoreLayer(use_merge=False)(dnn_output)

    return tf.keras.Model(dense_inputs+sparse_inputs+seq_inputs,output)

def MIMN(denseInfo:list=None,sparseInfo:list=None,seqInfo:list=None,behaviorFea=None,candidateFea=None,
         controller_hidden_units=None,attention_hidden=None,classify_hidden=None,channel_dim=20,memory_slots=128,
         memory_bits=20, mult_head=3,use_miu=True):
    '''
        Warning!!! MIMN need set static batchSize==>please set date_prepare(batch_size=?[e.g 32]),
        not support dynamic batchSize!!!
    '''
    if not controller_hidden_units:
        controller_hidden_units=[128,64]
    if not attention_hidden:
        attention_hidden = [128, 64]
    if not classify_hidden:
        classify_hidden = [128, 64]

    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    sparse_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)
    seq_embed, seq_mask = SparseEmbed(seqInfo, use_flatten=False, is_linear=False, mask_zero=True)(seq_inputs)
    seq_embed=StackLayer(use_flat=False)(ExtractLayer(need_fea=behaviorFea,need_inputs=seq_inputs)(seq_embed))
    target_embed=StackLayer(use_flat=False)(ExtractLayer(need_fea=candidateFea,need_inputs=sparse_inputs,need_remove=False)(sparse_embed))

    [M,pre_read,pre_readW,_,S,__,loss]=UICLayer(controller_network=DnnLayer(hidden_units=controller_hidden_units),
                                         controller_input_flat=True, channel_dim=channel_dim,memory_slots=memory_slots,
                                         memory_bits=memory_bits, mult_head=mult_head, use_miu=use_miu,return_hidden=True)(seq_embed)
    print(loss)
    sFea=ActivationUnitLayer(hidden_units=attention_hidden,need_stack=False)([target_embed, S])
    read_input, readW = ReadLayer(addressCal=AddressCalLayer())([pre_readW, M, pre_read])
    mFea=ControlWrapLayer(controller_network=DnnLayer(controller_hidden_units)
                          ,controller_input_flat=True)([tf.squeeze(target_embed,axis=1), read_input, pre_read])[1]

    dnn_inputs=StackLayer(use_flat=True)([sFea,mFea]+sparse_embed)
    dnn_output=DnnLayer(hidden_units=classify_hidden)(dnn_inputs)
    output=MergeScoreLayer(use_merge=False)(dnn_output)

    model=tf.keras.Model(dense_inputs+sparse_inputs+seq_inputs,output)
    model.add_loss(loss)

    return model

def DSTN(denseInfo:list=None, sparseInfo:list=None, seqInfo:list=None):
    [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    sparse_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)



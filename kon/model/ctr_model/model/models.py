#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/3 上午9:13
@File   :models.py
================================='''
from kon.utils.data_prepare import data_prepare, InputFeature
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
# class FeatureInput(object):

def TestModel(inputFea:InputFeature=None):
    input_=StackLayer()(inputFea.sparse_embed)
    dnn_output=DnnLayer(hidden_units=[100,10])(input_)
    output=MergeScoreLayer(use_merge=False)(dnn_output)
    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)

def FM(inputFea:InputFeature=None):
    embed_list=[inputFea.sparse_embed,inputFea.linear_embed]
    fm_=FmLayer()(embed_list)
    output=MergeScoreLayer(use_merge=False)(tf.squeeze(fm_,axis=1))

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs,output)

def PNN(inputFea:InputFeature=None,hidden_units=None,use_inner=True,use_outer=True):
    if hidden_units is None:
        hidden_units=[256,256,256]
    cross_fea=inputFea.linear_embed

    if use_inner:
        cross_fea+=IPnnLayer()(inputFea.sparse_embed)
    if use_outer:
        cross_fea+=OPnnLayer()(inputFea.sparse_embed)
    cross_fea=StackLayer()(cross_fea)
    dnn_ = DnnLayer(hidden_units)(cross_fea)
    output=MergeScoreLayer(use_merge=False)(dnn_)

    return tf.keras.Model(inputFea.sparse_inputs,output)

def DeepCross(inputFea:InputFeature=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 256, 256]

        dnn_inputs=StackLayer()(inputFea.dense_inputs+inputFea.sparse_embed)

        dnn_fea=DnnLayer(hidden_units=hidden_units)(dnn_inputs)
        output=MergeScoreLayer(use_merge=False)(dnn_fea)

        return tf.keras.Model(inputFea.sparse_inputs,output)

def Wide_Deep(inputFea:InputFeature=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]

    dnn_inputs=StackLayer()(inputFea.dense_inputs+inputFea.sparse_embed)

    dnn_=DnnLayer(hidden_units=hidden_units)(dnn_inputs)
    output=MergeScoreLayer()(inputFea.linear_embed+[dnn_])

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs,output)

def DeepFM(inputFea:InputFeature=None, hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]
    embed_list = [inputFea.sparse_embed, inputFea.linear_embed]

    fm_=FmLayer()(embed_list)
    dnn_input=StackLayer()(inputFea.dense_inputs+inputFea.sparse_embed)
    dnn_ = DnnLayer(hidden_units=hidden_units)(dnn_input)
    output = MergeScoreLayer()([fm_, dnn_])

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs, output)

def DCN(inputFea:InputFeature=None,hidden_units=None,cross_hidden=3):
    '''
        Notice:
            cross_hidden==> iter_num(x^k=w(x^k-1*x0)+b+x0)
    '''
    if hidden_units is None:
        hidden_units = [256, 128, 64]

    combine_inputs=StackLayer()(inputFea.dense_inputs+inputFea.sparse_embed)

    cross_fea=CrossLayer(cross_hidden=cross_hidden)(combine_inputs)
    deep_fea=DnnLayer(hidden_units=hidden_units)(combine_inputs)
    output=MergeScoreLayer()([cross_fea,deep_fea])

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs,output)

def NFM(inputFea:InputFeature=None,hidden_units=None):
    if hidden_units is None:
        hidden_units = [256, 128, 64]

    cross_inputs=InnerLayer(use_inner=True,use_add=True)(inputFea.sparse_embed)
    dnn_inputs=StackLayer()(inputFea.dense_inputs+[cross_inputs])

    dnn_fea=DnnLayer(hidden_units=hidden_units,output_dim=1)(dnn_inputs)
    final_fea=tf.keras.layers.Add()(inputFea.linear_embed+[dnn_fea])
    output=ScoreLayer()(final_fea)

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs,output)

def XDeepFM(inputFea:InputFeature=None, conv_size=None, hidden_units=None):
    '''
    :param conv_size:
        notice:conv_size decision Hk<size>
    '''
    if conv_size is None:
        conv_size = [200, 200, 200]
    if hidden_units is None:
        hidden_units = [256, 128, 64]

    cin_inputs=tf.keras.layers.Concatenate(axis=1)(inputFea.sparse_embed)
    dnn_inputs=StackLayer()(inputFea.dense_inputs+inputFea.sparse_embed)

    cin_output=CIN(conv_size=conv_size,output_dim=1)(cin_inputs)
    dnn_output=DnnLayer(hidden_units=hidden_units,output_dim=1)(dnn_inputs)
    output=ScoreLayer(use_add=True)([inputFea.linear_embed,cin_output,dnn_output])

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs,output)


def AFM(inputFea:InputFeature=None):

    cross_output=InnerLayer()(inputFea.sparse_embed)
    atten_output=AttentionBaseLayer()(cross_output)
    output=ScoreLayer(use_add=True)(inputFea.linear_embed+[atten_output])

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs,output)


def AutoInt(inputFea:InputFeature=None,attention_dim=8,attention_head_dim=3):
    '''
        notice:
            origin inputs=[dense+sparse],now inputs=[sparse]
            MultHeadAttentionLayer !support Bn&Add&Activate,
            because now want to as hidden of DnnLayer,update at soon...
        core:
            multHead to replace inner of fm
    '''
    cross_embed = StackLayer(use_flat=False,axis=1)(inputFea.sparse_embed)
    atten_layer=MultHeadAttentionLayer(attention_dim=attention_dim,attention_head_dim=attention_head_dim,use_ln=True,atten_mask_mod=1)
    atten_vec=DnnLayer(res_unit=1,other_dense=[atten_layer])(cross_embed)
    final_input=StackLayer(use_flat=True,axis=-1)([tf.squeeze(i,0) for i in tf.split(atten_vec,[1]*atten_vec.shape[0])])
    output=MergeScoreLayer(use_merge=False)(final_input)

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)


def DIN(inputFea:InputFeature=None, candidateFea=None, behaviorFea=None,
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

    candidate_embed=ExtractLayer(candidateFea,inputFea.sparse_inputs)(inputFea.sparse_embed)
    behavior_embed=ExtractLayer(behaviorFea,inputFea.seq_inputs,mask_zero=True)(inputFea.seq_embed_list[0])

    base_behavior=SeqBaseLayer()(inputFea.seq_embed_list[0])
    attention_behavior=ActivationUnitLayer(hidden_units=attention_units)([candidate_embed,behavior_embed],mask=inputFea.seq_embed_list[1][0])
    final_inputs=StackLayer(use_flat=True)(base_behavior+[attention_behavior])

    mlp_output=DnnLayer(hidden_units=hidden_units,hidden_activate=tf.keras.layers.PReLU(),res_unit=2,use_bn=True)(final_inputs)
    output=MergeScoreLayer(use_merge=False)(mlp_output)

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)

def DIEN(inputFea:InputFeature=None,candidateFea=None, behaviorFea=None, classify_units=None, hidden_units=None,
         attention_units=None,sample_num=5,useCore=False):
    '''
    notice:
        at auxLoss not use Dice as DnnLayer Activate...
        because i think sample seq to BN may be...
        In fact,default param:attention&auxLoss size not support BN,too
    :param classify_units: AuxLoss classify==DnnLayer ?please input hidden size:ignore
    :param hidden_units: final classify DnnLayer hidden size
    :param attention_units: attention classify DnnLayer hiddenSize of ActivateUnits
    :param sample_num: num of DIEN Nega sample item
    '''
    if attention_units is None:
        attention_units = [100, 64, 32]
    if classify_units is None:
        classify_units = [100, 64, 32]
    if hidden_units is None:
        hidden_units = [256, 256, 256]

    behavior_embed=StackLayer(use_flat=False)(ExtractLayer(behaviorFea,inputFea.seq_inputs,mask_zero=True)(inputFea.seq_embed_list[0]))
    candidate_embed=StackLayer(use_flat=False)(ExtractLayer(candidateFea,inputFea.sparse_inputs)(inputFea.sparse_embed))

    behavior_sample = SampleLayer(sample_num=sample_num)(behavior_embed)
    [hidden_list,aux_loss]=InterestExtratorLayer(classify_units=classify_units,sample_num=sample_num)([behavior_embed,behavior_sample],mask=inputFea.seq_embed_list[1][0])
    final_hidden=InterestEolvingLayer(attention_units=attention_units)([candidate_embed,hidden_list],mask=inputFea.seq_embed_list[1][0])

    final_input=StackLayer()([final_hidden]+inputFea.sparse_embed)
    output=DnnLayer(hidden_units=hidden_units,hidden_activate=tf.keras.layers.PReLU())(final_input)
    output=MergeScoreLayer(use_merge=False,output_dim=2)(output)

    if useCore:
      return output,aux_loss

    model=tf.keras.Model(inputFea.dense_inputs + inputFea.sparse_inputs + inputFea.seq_inputs, output)
    model.add_loss(aux_loss)

    return model

def DSIN(inputFea:InputFeature=None,candidateFea=None, behaviorFea=None,attention_dim=8,attention_head_dim=5,ffn_hidden_unit=10,
         lstm_units=8,lstm_mode='sum',attention_units=None,classify_units=None,sessionMaxLen=10,sessionMaxNum=20):

    if attention_units is None:
        attention_units = [100, 64, 32]
    if classify_units is None:
        classify_units = [100, 64, 32]

    behavior_embed=StackLayer(use_flat=False)(ExtractLayer(behaviorFea,inputFea.seq_inputs,mask_zero=True)(inputFea.seq_embed_list[0]))
    candidate_embed=StackLayer(use_flat=False)(ExtractLayer(candidateFea,inputFea.sparse_inputs)(inputFea.sparse_embed))

    pos_behavior=SessionDivisonLayer(sessionMaxLen=sessionMaxLen,sessionMaxNum=sessionMaxNum)(behavior_embed)
    self_behavior=SessionInterestExtractorLayer(attention_dim=attention_dim,attention_head_dim=attention_head_dim,ffn_hidden_unit=ffn_hidden_unit)(pos_behavior)

    self_atten=ActivationUnitLayer(attention_units,need_stack=False)([candidate_embed,self_behavior],mask=inputFea.seq_embed_list[1][0])
    hidden_behavior=SessionInterestInteractingLayer(biLstmUnit=lstm_units,lstm_mode=lstm_mode)(self_behavior)
    lstm_atten=ActivationUnitLayer(attention_units,need_stack=False)([candidate_embed,hidden_behavior],mask=inputFea.seq_embed_list[1][0])
    dnn_inputs=StackLayer(use_flat=True)(inputFea.sparse_embed+inputFea.dense_inputs+[self_atten,lstm_atten])
    output=DnnLayer(hidden_units=classify_units,use_bn=True,res_unit=2)(dnn_inputs)
    output=MergeScoreLayer(use_merge=False,output_dim=2)(output)

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)


def SeqFM(inputFea:InputFeature=None,hidden_units=None,res_unit=1,atten_dim=8,atten_head=1):
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

    linear = SparseEmbed(inputFea.sparse_info, is_linear=True)(inputFea.sparse_inputs)
    sparse_embed = StackLayer(axis=1,use_flat=False)(SparseEmbed(inputFea.sparse_info, use_flatten=False)(inputFea.sparse_inputs))
    seq_embed= StackLayer(use_flat=False)(SparseEmbed(inputFea.seq_info, use_flatten=False, is_linear=False,mask_zero=False)(inputFea.seq_inputs))

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

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)


def DTS(inputFea:InputFeature=None,userFea:list=None,timestampFea:list=None,behaviorFea:list=None,targetFea:list=None,
         ode_mode=1,sample_num=1,is_train=True,loss_lambda:int=0.5):

    timestampEmbed = tf.expand_dims(StackLayer(use_flat=False)(ExtractLayer(timestampFea, inputFea.seq_inputs)(inputFea.seq_embed_list[0])),axis=-1)
    behaviorEmbed = StackLayer(use_flat=False)(ExtractLayer(behaviorFea, inputFea.seq_inputs)(inputFea.seq_embed_list[0]))
    userEmbed = tf.squeeze(StackLayer(use_flat=False)(ExtractLayer(userFea, inputFea.sparse_inputs)(inputFea.sparse_embed)),axis=1)
    targetEmbed,sparse_embed=ExtractLayer(targetFea, inputFea.sparse_inputs,need_remove=True)(inputFea.sparse_embed)
    behaviorEmbed=[behaviorEmbed,targetEmbed]

    behavior,loss_=TimeStreamLayer(sample_num=sample_num,ode_mode=ode_mode,trainable=is_train,loss_lambda=loss_lambda)([timestampEmbed,userEmbed,behaviorEmbed],mask=inputFea.seq_embed_list[1][0])
    behavior,targetItem=behavior[0],behavior[1]

    behavior=tf.reduce_mean(behavior,axis=1)
    dnn_input=StackLayer(use_flat=True)([behavior]+sparse_embed)
    dnn_output=DnnLayer(hidden_units=[64,32])(dnn_input)
    output=MergeScoreLayer(use_merge=False)(dnn_output)

    model=tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)
    model.add_loss(loss_)

    return model


def BST(inputFea:InputFeature=None,behaviorFea=None,
        attention_units=8,hidden_units=None,ffn_hidden_unit=8,attention_head=3):
    if hidden_units is None:
        hidden_units = [100, 64, 32]

    behaviorEmb=StackLayer(use_flat=False,axis=1)(ExtractLayer(behaviorFea,inputFea.seq_inputs)(inputFea.seq_embed_list[0]))
    transformerFea=[SelfAttentionLayer(attention_dim=attention_units,attention_head_dim=attention_head,ffn_hidden_unit=ffn_hidden_unit)(emb_)
        for emb_ in tf.split(behaviorEmb,[2]*(behaviorEmb.shape[1]//2),axis=1)]

    dnn_inputs=StackLayer(axis=-1)(inputFea.sparse_embed+transformerFea)
    dnn_output=DnnLayer(hidden_units=hidden_units)(dnn_inputs)
    output=MergeScoreLayer(use_merge=False)(dnn_output)

    return tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)

def MIMN(inputFea:InputFeature=None,behaviorFea=None,candidateFea=None,
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

    seq_embed=StackLayer(use_flat=False)(ExtractLayer(need_fea=behaviorFea,need_inputs=inputFea.seq_inputs)(inputFea.seq_embed_list[0]))
    target_embed=StackLayer(use_flat=False)(ExtractLayer(need_fea=candidateFea,need_inputs=inputFea.sparse_inputs,need_remove=False)(inputFea.sparse_embed))

    [M,pre_read,pre_readW,_,S,__,loss]=UICLayer(controller_network=DnnLayer(hidden_units=controller_hidden_units),
                                         controller_input_flat=True, channel_dim=channel_dim,memory_slots=memory_slots,
                                         memory_bits=memory_bits, mult_head=mult_head, use_miu=use_miu,return_hidden=True)(seq_embed)
    print(loss)
    sFea=ActivationUnitLayer(hidden_units=attention_hidden,need_stack=False)([target_embed, S])
    read_input, readW = ReadLayer(addressCal=AddressCalLayer())([pre_readW, M, pre_read])
    mFea=ControlWrapLayer(controller_network=DnnLayer(controller_hidden_units)
                          ,controller_input_flat=True)([tf.squeeze(target_embed,axis=1), read_input, pre_read])[1]

    dnn_inputs=StackLayer(use_flat=True)([sFea,mFea]+inputFea.sparse_embed)
    dnn_output=DnnLayer(hidden_units=classify_hidden)(dnn_inputs)
    output=MergeScoreLayer(use_merge=False)(dnn_output)

    model=tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)
    model.add_loss(loss)

    return model

def DSTN(inputFea:InputFeature=None):
    pass
    # [dense_inputs, sparse_inputs, seq_inputs] = prepare_tool.df_prepare(sparseInfo=sparseInfo, denseInfo=denseInfo,seqInfo=seqInfo)
    # sparse_embed = SparseEmbed(sparseInfo, use_flatten=False)(sparse_inputs)


def SIM(inputFea:InputFeature=None,reduceFea:list=None,candidateFea:list=None,behaviorFea:list=None,
        attention_dim:int=None,attention_head_dim:int=None,hidden_units:list=None,
        recent_classify_units=None,recent_hidden_units:int=None,recent_attention_units:int=None,sample_num=5
        ):
    '''
    :param reduceFea: hardSearch Feature
    warning:
        1.only building hardSearch,not building softSearch,now i'don't know how to write alsh
        2.example is not true,inFact seq should use recent logs is not all logs to feed to behavior model(DIEN),i'm lazy

    '''
    if hidden_units is None:
        hidden_units = [128,128,128]
    if attention_dim is None:
        attention_dim = 3
    if attention_head_dim is None:
        attention_head_dim = 3
    if recent_hidden_units is None:
        recent_hidden_units = [128,128,128]
    if recent_attention_units is None:
        recent_attention_units = [128,128,128]
    if recent_classify_units is None:
        recent_classify_units = [128,128,128]

    reduceSeq=ExtractLayer(reduceFea,inputFea.seq_inputs)(inputFea.seq_embed_list[0])

    reduceSeq=ESULayer(attention_dim=attention_dim,attention_head_dim=attention_head_dim)(reduceSeq)
    dnnOutput=DnnLayer(hidden_units=hidden_units,hidden_activate=tf.keras.layers.PReLU(),use_bn=True,use_flatten=True)(reduceSeq)
    shortOutput,auxLoss=DIEN(inputFea,candidateFea=candidateFea,behaviorFea=behaviorFea,classify_units=recent_classify_units,
                             hidden_units=recent_hidden_units,attention_units=recent_attention_units,sample_num=sample_num,useCore=True)
    output=MergeScoreLayer(use_merge=False)(dnnOutput)
    model=tf.keras.Model(inputFea.dense_inputs+inputFea.sparse_inputs+inputFea.seq_inputs,output)
    model.add_loss(auxLoss)

    return model


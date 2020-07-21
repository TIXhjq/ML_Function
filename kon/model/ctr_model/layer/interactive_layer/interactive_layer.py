#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :2020/5/3 上午11:41
@File   :interactive_layer.py
================================='''
import itertools
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

class InnerLayer(tf.keras.layers.Layer):
    '''
        Interactive layer
    '''
    def __init__(self,use_inner:bool=True,mod=1,seed=2020,perm=None,use_add=False):
        '''
        :param mod:
            0.output matrix
            1.output matrix result
        '''
        super(InnerLayer,self).__init__()
        self.use_inner=use_inner
        self.mod=mod
        self.seed=seed
        self.perm=perm
        self.use_add=use_add
        if use_add:
            self.add_=tf.keras.layers.Add()

    def build(self, input_shape):
        super().build(input_shape)
        len_data=((len(input_shape)-1)*len(input_shape))//2
        # self.dot =[tf.keras.layers.Dot(axes=self.mod) for i in range(len_data)]
        # self.mult = [tf.keras.layers.Multiply() for i in range(len_data)]

    def call(self,inputs,**kwargs):
        if self.use_inner:
            cross_list=[tf.multiply(emb1,emb2) for emb1,emb2 in itertools.combinations(inputs, 2)]
        else:
            cross_list = [tf.keras.backend.batch_dot(tf.transpose(emb1,perm=self.perm), emb2) for dot_, (emb1, emb2) in zip(self.dot, itertools.combinations(inputs, 2))]
        if self.use_add:
            cross_list=self.add_(cross_list)
        return cross_list

class IPnnLayer(tf.keras.layers.Layer):
    def __init__(self,seed=2020):
        super(IPnnLayer, self).__init__()
        self.seed=seed
        self.inner=InnerLayer()
        self.concat=tf.keras.layers.Concatenate()

    def build(self,input_shape):
        super(IPnnLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inner_list=self.inner(inputs)
        return inner_list

class ExtractLayer(tf.keras.layers.Layer):
    def __init__(self,need_fea,need_inputs,supports_masking=True,mask_zero=False,need_remove=False):
        super(ExtractLayer, self).__init__()
        self.need_fea=need_fea
        self.need_inputs=need_inputs
        self.supports_masking=supports_masking
        self.mask_zero=mask_zero
        self.need_remove=need_remove

    def build(self, input_shape):
        super(ExtractLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        self.need_idx=[idx_ for idx_,input_ in enumerate(self.need_inputs) if input_.name[:-2] in self.need_fea]
        need_inputs=[inputs[idx_] for idx_ in self.need_idx]

        format_inputs=inputs
        if self.need_remove:
            format_inputs=[i for idx_,i in enumerate(inputs) if idx_ not in self.need_idx]
            return [need_inputs,format_inputs]
        else:
            return need_inputs


    def compute_mask(self,inputs,mask=None):
        if not self.mask_zero:
            return None
        return [mask[i] for i in self.need_idx]

class OPnnLayer(tf.keras.layers.Layer):
    def __init__(self,use_reduce=True,seed=2020,use_flatten=True):
        '''
        notice! reduce_mem==>embedding info drop
            op:sum(embedding_list),but it represent ?
        '''
        super(OPnnLayer, self).__init__()
        self.seed=seed
        self.use_reduce=use_reduce
        self.outer=InnerLayer(use_inner=False,perm=[0,2,1],mod=(1,2))
        self.add=tf.keras.layers.Add()
        self.use_flatten=use_flatten

    def build(self, input_shape):
        super(OPnnLayer, self).build(input_shape)
        fea_size=len(input_shape)
        if self.use_reduce:
            fea_size=1
        self.flat=[tf.keras.layers.Flatten() for i in range(fea_size)]

    def call(self, inputs, **kwargs):
        if self.use_reduce:
            sum_inputs=self.add(inputs)
            # sum_inputs=tf.expand_dims(sum_inputs,axis=-1)
            outer_list=self.outer([sum_inputs,sum_inputs])
        else:
            # inputs=[tf.expand_dims(input_,axis=-1) for input_ in inputs]
            outer_list = self.outer(inputs)

        if self.use_flatten:
            outer_list=[flat_(outer_) for outer_,flat_ in zip(outer_list,self.flat)]

        return outer_list

class FmLayer(tf.keras.layers.Layer):
    def __init__(self,use_inner:bool=True,mod=1,use_add=True,**kwargs):
        '''
        :param mod:
            0.output matrix
            1.output matrix result
        '''
        super(FmLayer, self).__init__(**kwargs)
        self.cross=InnerLayer(use_inner=use_inner,mod=mod,use_add=use_add)
        self.add = tf.keras.layers.Add()
        self.use_add=use_add

    def build(self,input_shape):
        super(FmLayer, self).build(input_shape)
        self.cross.build(input_shape)

    def call(self, inputs, **kwargs):
        '''
        :param inputs:[cross_embed,linear_embed]
        '''
        cross = self.cross(inputs[0])
        output = self.add([cross]+inputs[1])
        if self.use_add:
            return output
        else:
            return cross+inputs[1]

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self,initializer:str='random_normal'):
        super(LinearLayer,self).__init__()
        self.initalizer=initializer

    def build(self, input_shape):
        super(LinearLayer, self).build(input_shape)
        self.w = self.add_weight(shape=(input_shape[-1],1),
                                 initializer=self.initalizer,
                                 trainable=True)
        self.b = self.add_weight(shape=(1,),
                                 initializer=self.initalizer,
                                 trainable=True)

    def call(self,inputs,**kwargs):
        return [tf.tensordot(a=input,b=self.w,axes=1)+self.b for input in inputs]

class SparseEmbed(tf.keras.layers.Layer):
    '''
        embedding core:
            supports sparse embed & linear
        supports:
            flatten,add
    '''
    def __init__(self,sparse_info:list,is_linear=False,use_flatten=True,use_add=False,seed=2020,support_masking=True,mask_zero=False):
        super(SparseEmbed,self).__init__()
        self.sparse_info=sparse_info
        self.flatten=None
        self.supports_masking=support_masking
        self.is_linear = is_linear
        self.mask_zero=mask_zero
        self.use_add = use_add
        self.seed=seed

        if use_flatten:
            self.flatten=[tf.keras.layers.Flatten()for i in sparse_info]
        if use_add:
            self.add=tf.keras.layers.Add()

    def build(self, input_shape):
        if not self.is_linear:
            self.embed=[tf.keras.layers.Embedding(
                name=info_.fea_name,input_dim=info_.word_size,output_dim=info_.cross_unit,
                mask_zero=info_.mask_zero,embeddings_initializer=glorot_uniform(seed=self.seed),
                input_length=info_.input_length,trainable=info_.is_trainable,weights=info_.pre_weight
            ) if info_.cross_unit!=0 else [] for info_ in self.sparse_info]
        else:
            self.embed=[tf.keras.layers.Embedding(
                name=info_.fea_name,input_dim=info_.word_size,output_dim=info_.linear_unit
            )for info_ in self.sparse_info]
        super(SparseEmbed, self).build(input_shape)

    def call(self,inputs,**kwargs):

        embed_list = [emb_(input_) if info_.cross_unit != 0 else input_ for emb_, input_, info_ in
                      zip(self.embed ,inputs, self.sparse_info)]


        if self.flatten:
            embed_list=[flat_(embed_) for flat_,embed_ in zip(self.flatten,embed_list)]

        if self.use_add:
            embed_list=self.add(embed_list)

        self.embed_list=embed_list

        if self.mask_zero:
            return embed_list,[emb._keras_mask if info_.cross_unit!=0 else [] for emb,info_ in zip(embed_list,self.sparse_info)]
        else:
            return embed_list

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return [embed._keras_mask for embed in self.embed_list]


class CrossLayer(tf.keras.layers.Layer):
    '''
        DCN core:
            x^k=(x^k-1*x0)+b+x0(Recursive Format)
    '''
    def __init__(self,cross_hidden=3,seed=2020,**kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        self.outer=InnerLayer(use_inner=False,mod=(-2,-1),perm=([0,2,1]))
        self.cross_hidden=cross_hidden
        self.seed=seed
        self.dot_=[tf.keras.layers.Dot(axes=1,name='Dot_{}'.format(str(i))) for i in range(cross_hidden)]
        self.add_=[tf.keras.layers.Add(name='Add_{}'.format(str(i)))for i in range(cross_hidden)]


    def build(self, input_shape):
        self.kernel=[
            self.add_weight(name='outer_weight_{}'.format(str(i)),
            shape=[input_shape[-1],1],initializer=glorot_uniform(seed=self.seed)
        )for i in range(self.cross_hidden)]
        self.bias=[
            self.add_weight(name='outer_bias_{}'.format(str(i)),
            shape=[input_shape[-1],1],initializer=tf.keras.initializers.zeros()
        )for i in range(self.cross_hidden)]
        super(CrossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs=tf.expand_dims(inputs,axis=-1)
        pre_inputs=inputs
        for i in range(self.cross_hidden):
            pre_inputs=tf.keras.backend.batch_dot(inputs,tf.keras.backend.dot(
                tf.transpose(pre_inputs,perm=[0,2,1]),self.kernel[i]))+pre_inputs+self.bias[i]

        return pre_inputs


class CIN(tf.keras.layers.Layer):
    '''
        XDeep core:
            x^k=sum[w(<x0,x^k-1>)]
            x1->....x^k---> RNN(Recursive Format)
            <x0,x^k-1>==hk*m*d--->con1D--->hk*d==>x^k
            final_output<unit=1>=concat(all feature map sum<axis=d>)

        feature map:
            row=D,col=m*h
    '''
    def __init__(self, conv_size=None, output_dim=1):
        super(CIN, self).__init__()
        if conv_size is None:
            conv_size = [200, 200, 200]
        self.conv_size=conv_size
        self.concat=tf.keras.layers.Concatenate()
        self.output_dim=output_dim
        if output_dim==1:
            self.logit_layer=tf.keras.layers.Dense(1)

    def build(self, input_shape):
        super(CIN, self).build(input_shape)
        self.hidden_conv=[tf.keras.layers.Conv1D(size,1) for size in self.conv_size]

    def call(self, inputs, **kwargs):
        x0 = tf.split(inputs, [1] * inputs.shape[-1], -1)
        pre_=x0
        sum_pooling_list=[]

        for conv_ in self.hidden_conv:
            z = tf.matmul(x0, pre_, transpose_b=True)
            z = tf.transpose(z, perm=[1, 0, 3, 2])
            z=tf.reshape(z,[-1,z.shape[1],z.shape[2]*z.shape[3]])
            z=conv_(z)
            pre_ = tf.transpose(z,[0,2,1])
            pre_=tf.split(pre_, [1] * pre_.shape[-1], -1)
            sum_pooling_list.append(tf.reduce_sum(z, axis=-1))
        output=self.concat(sum_pooling_list)
        if self.output_dim==1:
            output=self.logit_layer(output)

        return output

class AttentionBaseLayer(tf.keras.layers.Layer):
    '''
        AFM core:
            base attention
        advise to go directly DIN
    '''
    def __init__(self,attention_dim=4,seed=2020,output_dim=1):
        super(AttentionBaseLayer, self).__init__()
        self.add=tf.keras.layers.Add()
        self.atten_dim=attention_dim
        self.seed=seed
        self.single_mlp = tf.keras.layers.Dense(1, 'relu', use_bias=False, kernel_initializer=glorot_uniform(self.seed))
        self.single_softmax=tf.keras.layers.Activation('softmax')
        self.output_layer=tf.keras.layers.Dense(output_dim)

    def build(self, input_shape):
        super(AttentionBaseLayer, self).build(input_shape)

        self.kernel_w=self.add_weight(
            name='single_score_w',
            shape=(input_shape[0][-1],self.atten_dim),
            initializer=glorot_uniform(seed=self.seed)
        )
        self.kernel_b=self.add_weight(
            name='single_score_b',
            shape=(self.atten_dim,),
            initializer=glorot_uniform(seed=self.seed)
        )


    def call(self, inputs, **kwargs):
        inputs=tf.concat(inputs,axis=1)
        score_=self.single_mlp(tf.add(tf.keras.backend.dot(inputs,self.kernel_w),self.kernel_b))
        score_w=self.single_softmax(score_)
        atten_inputs=tf.reduce_sum(score_w*inputs,axis=1)
        output=self.output_layer(atten_inputs)

        return output


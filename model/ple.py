# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:29:13 2021

@author: shujie.wang
"""

import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import os
from tensorflow.keras import optimizers,initializers

from tensorflow.python.keras.initializers import glorot_normal #xariver。截断的正态分布，标准差。
from tensorflow.python.keras.layers import Layer
import pickle as pkl
import gc
from time import time

class MyMeanPool(Layer):
    def __init__(self, axis, **kwargs):
        super(MyMeanPool, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x, mask):
        mask = tf.expand_dims(tf.cast(mask,tf.float32),axis = -1)
        x = x * mask
        return K.sum(x, axis=self.axis) / (K.sum(mask, axis=self.axis) + 1e-9)

class PleLayer(tf.keras.layers.Layer):
    '''
    n_experts:list,每个任务使用几个expert。[2,3]第一个任务使用2个expert，第二个任务使用3个expert。
    n_expert_share:int,共享的部分设置的expert个数。
    expert_dim:int,每个专家网络输出的向量维度。
    n_task:int,任务个数。
    '''
    def __init__(self,n_task,n_experts,expert_dim,n_expert_share,dnn_reg_l2 = 1e-5):
        super(PleLayer, self).__init__()
        self.n_task = n_task
        
        # 生成多个任务特定网络和1个共享网络。
        self.E_layer = []
        for i in range(n_task):
            sub_exp = [Dense(expert_dim,activation = 'relu') for j in range(n_experts[i])]
            self.E_layer.append(sub_exp)
            
        self.share_layer = [Dense(expert_dim,activation = 'relu') for j in range(n_expert_share)]
        #定义门控网络
        self.gate_layers = [Dense(n_expert_share+n_experts[i],kernel_regularizer=regularizers.l2(dnn_reg_l2),
                                  activation = 'softmax') for i in range(n_task)]

    def call(self,x):
        #特定网络和共享网络
        E_net = [[expert(x) for expert in sub_expert] for sub_expert in self.E_layer]
        share_net = [expert(x) for expert in self.share_layer]
        
        #门的权重乘上，指定任务和共享任务的输出。
        towers = []
        for i in range(self.n_task):
            g = self.gate_layers[i](x)
            g = tf.expand_dims(g,axis = -1) #(bs,n_expert_share+n_experts[i],1)
            _e = share_net+E_net[i]  
            _e = Concatenate(axis = 1)([expert[:,tf.newaxis,:] for expert in _e]) #(bs,n_expert_share+n_experts[i],expert_dim)
            _tower = tf.matmul(_e, g,transpose_a=True)
            towers.append(Flatten()(_tower)) #(bs,expert_dim)
        return towers

def build_ple(sparse_cols,dense_cols,sparse_max_len,embed_dim,expert_dim = 4,
              varlens_cols = [],varlens_max_len = [],dnn_hidden_units = (64,64),
              n_task = 2,n_experts = [2,2],n_expert_share = 4,dnn_reg_l2 = 1e-6,
              drop_rate = 0.0,embedding_reg_l2 = 1e-6,targets = []):

   #输入部分，分为sparse,varlens,dense部分。
    sparse_inputs = {f:Input([1],name = f) for f in sparse_cols}
    dense_inputs = {f:Input([1],name = f) for f in dense_cols}
    varlens_inputs = {f:Input([None,1],name = f) for f in varlens_cols}
        
    input_embed = {}
    #离散特征，embedding到k维
    for f in sparse_cols:
        _input = sparse_inputs[f]
        embedding = Embedding(sparse_max_len[f], embed_dim, 
            embeddings_regularizer=tf.keras.regularizers.l2(embedding_reg_l2)) 
        input_embed[f] =Flatten()(embedding(_input)) #(bs,k)
        
    #多标签离散变量
    for f in varlens_inputs:
        _input = varlens_inputs[f]
        mask = Masking(mask_value = 0).compute_mask(_input)
        embedding = Embedding(varlens_max_len[f], embed_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        _embed =Reshape([-1,embed_dim])(embedding(_input))
        out_embed = MyMeanPool(axis=1)(_embed,mask)
        input_embed[f] = out_embed
        
    input_embed.update(dense_inputs) #加入连续变量
    input_embed = Concatenate(axis = -1)([input_embed[f] for f in input_embed])    
                                  
    for num in dnn_hidden_units:
        input_embed = Dropout(drop_rate)(Dense(num,activation = 'relu',
                    kernel_regularizer=regularizers.l2(dnn_reg_l2))(input_embed))
    #Ple网络层
    towers = PleLayer(n_task,n_experts,expert_dim,n_expert_share)(input_embed)
    outputs = [Dense(1,activation = 'sigmoid',kernel_regularizer=regularizers.l2(dnn_reg_l2),
                       name = f,use_bias = True)(_t) for f,_t in zip(targets,towers)]
    inputs = [sparse_inputs[f] for f in sparse_inputs]+[varlens_inputs[f] for f in varlens_inputs]\
                +[dense_inputs[f] for f in dense_inputs]
    model = Model(inputs,outputs) 
    return model

if __name__ == '__main__':    
    sparse_cols = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    varlen_cols = ['manual_tag_list','manual_keyword_list']
    dense_cols = ['videoplayseconds']
    
    sparse_max_len = {f:1000 for f in sparse_cols}
    varlens_max_len = {f:1000 for f in varlen_cols}
    targets = ['task1','task2','task3','task4']
    from util import binary_focal_loss
    
    model = build_ple(sparse_cols,dense_cols,sparse_max_len,embed_dim = 16,expert_dim = 32,
              varlens_cols = [],varlens_max_len = [],dnn_hidden_units = (128,128),
              n_task = 4,n_experts = [2,2,2,2],n_expert_share = 4,dnn_reg_l2 = 1e-6,
              drop_rate = 0.0,embedding_reg_l2 = 1e-6,targets = targets)
    focal_loss = binary_focal_loss(gamma=2, alpha=0.25)
    losses = {x:focal_loss for x in targets}
    model.compile(loss = losses,optimizer = 'Adam')
    model.summary()
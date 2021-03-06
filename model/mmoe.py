# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:28:27 2021

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
    
class Mmoe_layer(tf.keras.layers.Layer):
    def __init__(self,expert_dim,n_expert,n_task):
        super(Mmoe_layer, self).__init__()
        self.n_task = n_task
        self.expert_layer = [Dense(expert_dim,activation = 'relu') for i in range(n_expert)]
        self.gate_layers = [Dense(n_expert,activation = 'softmax') for i in range(n_task)]
    
    def call(self,x):
        #多个专家网络
        E_net = [expert(x) for expert in self.expert_layer]
        E_net = Concatenate(axis = 1)([e[:,tf.newaxis,:] for e in E_net]) #(bs,n_expert,n_dims)
        #多个门网络
        gate_net = [gate(x) for gate in self.gate_layers]     #n_task个(bs,n_expert)
        
        #每个towers等于，对应的门网络乘上所有的专家网络。
        towers = []
        for i in range(self.n_task):
            g = tf.expand_dims(gate_net[i],axis = -1)  #(bs,n_expert,1)
            _tower = tf.matmul(E_net, g,transpose_a=True)
            towers.append(Flatten()(_tower))           #(bs,expert_dim)
            
        return towers

def build_mmoe(sparse_cols,dense_cols,sparse_max_len,embed_dim,expert_dim,
              varlens_cols,varlens_max_len,n_expert,n_task,target = [],
              dnn_hidden_units = (64,),dnn_reg_l2 = 1e-5,drop_rate = 0.1,
                embedding_reg_l2 = 1e-6):
    
    
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
    
    #mmoe网络层
    towers = Mmoe_layer(expert_dim,n_expert,n_task)(input_embed)
    outputs = [Dense(1,activation = 'sigmoid', kernel_regularizer=regularizers.l2(dnn_reg_l2),
                     name = f,use_bias = True)(_t) for _t,f in zip(towers,target)]
    inputs = [sparse_inputs[f] for f in sparse_inputs]+[varlens_inputs[f] for f in varlens_inputs]\
                +[dense_inputs[f] for f in dense_inputs]
    model = Model(inputs,outputs) 
    return model

if __name__ == '__main__':    
    target = ["read_comment", "like", "click_avatar", "forward"]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    varlen_features = ['manual_tag_list','manual_keyword_list']
    dense_features = ['videoplayseconds']
    #1.加载数据
    with open('../user_data/data.pkl','rb') as f:
        train,val,test,encoder = pkl.load(f)
    train_num = len(train)
    
    #2.生成输入特征设置
    sparse_max_len = {f:len(encoder[f]) + 1 for f in sparse_features}
    varlens_max_len = {f:len(encoder[f]) + 1 for f in varlen_features}
    feature_names = sparse_features+varlen_features+dense_features
    
    # 3.generate input data for model
    train_model_input = {name: train[name] if name not in varlen_features else np.stack(train[name]) for name in feature_names } #训练模型的输入，字典类型。名称和具体值
    val_model_input = {name: val[name] if name not in varlen_features else np.stack(val[name]) for name in feature_names }
    test_model_input = {name: test[name] if name not in varlen_features else np.stack(test[name]) for name in feature_names}
    
    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]
    
    del train,val #多余的特征删除，释放内存。
    gc.collect()
    
    # 4.Define Model,train,predict and evaluate
    model = build_mmoe(sparse_features,dense_features,sparse_max_len,embed_dim = 16,expert_dim = 32,
              n_task = 4,n_expert = 4,varlens_cols = varlen_features,varlens_max_len = varlens_max_len,
              dnn_hidden_units = (64,64),target = target,dnn_reg_l2 = 1e-5,drop_rate = 0.1)

    adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(adam, loss = 'binary_crossentropy' ,metrics = [tf.keras.metrics.AUC()],)
    
    history = model.fit(train_model_input, train_labels,validation_data = (val_model_input,val_labels),
                        batch_size=10240, epochs=4, verbose=1)
        
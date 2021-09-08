# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:27:12 2021

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

def secondary_fm(W):
    #先相加再平方。
    frs_part = Add()(W)
    frs_part = Multiply()([frs_part,frs_part]) 
    #先平方再相加
    scd_part = Add()([Multiply()([_x,_x]) for _x in W])
    #相减，乘0.5.
    fm_part = Subtract()([frs_part,scd_part])
    fm_part = Lambda(lambda x:K.sum(x,axis = 1,keepdims = True)*0.5)(fm_part)
    return fm_part


def build_FM(sparse_cols,dense_cols,sparse_max_len,embed_dim = 16, 
               dnn_hidden_units=(128, 128),varlens_cols = [],varlens_max_len = {},
               dropout = 0,embedding_reg_l2 = 1e-6,dnn_reg_l2 = 0.0):
    ''' 
    sparse_cols,dense_cols:离散变量名，连续变量名。
    sparse_max_len：字典：离散变量对应的最大的取值范围。
    varlens_cols:可变离散变量名。
    varlens_max_len:可变离散变量的最大取值范围。
    '''
    
    #输入部分，分为sparse,varlens,dense部分。
    sparse_inputs = {f:Input([1],name = f) for f in sparse_cols}
    dense_inputs = {f:Input([1],name = f) for f in dense_cols}
    varlens_inputs = {f:Input([None,1],name = f) for f in varlens_cols}
        
    input_embed = {}
    #离散特征，embedding到k维，得到其隐向量。wi
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
    
    #连续变量
    for f in dense_inputs:
        _input = dense_inputs[f]
        _embed = Dense(embed_dim,use_bias = False,activation = 'linear')(_input)
        input_embed[f] = _embed
        
    feature_name =  sparse_cols+varlens_cols+dense_cols
    fm_embed = [input_embed[f] for f in feature_name]
    fm_part = secondary_fm(fm_embed)
    
    #离散变量和连续变量拼接成dnn feature
    dnn_feature = Concatenate(axis = -1)(fm_embed)
    for num in dnn_hidden_units:
        dnn_feature = Dropout(dropout)(Dense(num,activation='relu',
                    kernel_regularizer=regularizers.l2(dnn_reg_l2))(dnn_feature))
        
    dnn_output = Dense(1,activation = 'linear', kernel_regularizer=regularizers.l2(dnn_reg_l2),
          use_bias = True)(dnn_feature)
    logits = Activation('sigmoid')(Add()([fm_part,dnn_output]))
    inputs = [sparse_inputs[f] for f in sparse_inputs]+[varlens_inputs[f] for f in varlens_inputs]\
                +[dense_inputs[f] for f in dense_inputs]
    model = Model(inputs,logits) 
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
    
    train_labels = train['read_comment'].values
    val_labels = val['read_comment'].values
    
    del train,val #多余的特征删除，释放内存。
    gc.collect()
    
    model = build_FM(sparse_features,dense_features,sparse_max_len,embed_dim = 16, 
               dnn_hidden_units=(64,64),varlens_cols = varlen_features,varlens_max_len = varlens_max_len,
               dropout = 0.1,embedding_reg_l2 = 1e-6,dnn_reg_l2 = 0.0)
    
    adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(adam, loss = 'binary_crossentropy' ,metrics = [tf.keras.metrics.AUC()],)
    
    history = model.fit(train_model_input, train_labels,validation_data = (val_model_input,val_labels),
                        batch_size=10240, epochs=4, verbose=1)
        
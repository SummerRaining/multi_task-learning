# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 23:56:12 2021

@author: tunan
"""
import numpy as np
import pandas as pd
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import os
from tensorflow.keras import optimizers,initializers

from tensorflow.python.keras.initializers import glorot_normal #xariver。截断的正态分布，标准差。
from tensorflow.python.keras.layers import Layer
import pickle as pkl
import gc
from time import time
from model.evaluation import uAUC,compute_weighted_score,evaluate_deepctr
from model.lr_cosine import CosineAnnealing
from model.FocalLoss import binary_focal_loss
from model.mmoe import build_mmoe
from model.ple import build_ple

if __name__ == '__main__':
    lr_list = [1,2,2,2]
    batch_size = 20480
    
    target = ["read_comment", "like", "click_avatar", "forward"]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    varlen_features = ['manual_tag_list','manual_keyword_list']
    dense_features = ['videoplayseconds']
    max_lens = {'manual_tag_list':11,'manual_keyword_list':18}
    #1.加载数据
    with open('./user_data/data.pkl','rb') as f:
        train,val,test,encoder = pkl.load(f)
    train_num = len(train)
    
    #2.生成输入特征设置
    sparse_max_len = {f:len(encoder[f]) + 1 for f in sparse_features}
    varlens_max_len = {f:len(encoder[f]) + 1 for f in varlen_features}
    feature_names = sparse_features+varlen_features+dense_features
    
    # 3.generate input data for model
    train_model_input = {name: train[name] if name not in varlen_features else np.stack(train[name]) for name in feature_names } #训练模型的输入，字典类型。名称和具体值
    val_model_input = {name: val[name] if name not in varlen_features else np.stack(val[name]) for name in feature_names }
    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]
    
    # 4.Define Model,train,predict and evaluate
    train_model = build_mmoe(sparse_features,dense_features,sparse_max_len,
                             varlens_cols = varlen_features,varlens_max_len = varlens_max_len,
                             n_task = 4,embed_dim = 16,n_expert=4,target = target,
                             expert_dim= 32, dnn_hidden_units=(128,128),drop_rate = 0.0,
                             embedding_reg_l2 = 1e-6,dnn_reg_l2 = 0)
# =============================================================================
#     train_model = build_ple(sparse_features,dense_features,sparse_max_len,
#                             embed_dim = 16,expert_dim = 32,varlens_cols = varlen_features,
#                             varlens_max_len = varlens_max_len,dnn_hidden_units = (128,128),
#                             n_task = 4,n_experts = [2,2,2,2],n_expert_share = 4,dnn_reg_l2 = 1e-6,
#                             drop_rate = 0.0,embedding_reg_l2 = 1e-6,targets = target)
# =============================================================================
    
    # focal_loss = binary_focal_loss(gamma=1, alpha=0.1)
    # losses = {x:focal_loss for x in target}
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    train_model.compile(adam, loss = 'binary_crossentropy',metrics = [tf.keras.metrics.AUC()],)
    reduce_lr = CosineAnnealing(eta_max=1, eta_min=0, num_step_per_epoch=(train_num // batch_size), lr_list = lr_list)
    history = train_model.fit(train_model_input, train_labels,validation_data = (val_model_input,val_labels)
                              ,batch_size=batch_size,epochs=7, verbose=1,callbacks=[reduce_lr])
    result = pd.DataFrame(history.history)
    result['average_auc'] = result[[f'val_{i}_auc' for i in target]].mean(axis = 1)
    best_epoch = result['average_auc'].argmax()
    average_auc = result['average_auc'].iloc[best_epoch]
    result = result[[f'val_{i}_auc' for i in target]].iloc[best_epoch]
    print(f'第个{best_epoch+1}epoch达到了最大值{average_auc},'+';'.join([f'{target[i]}:{result[i]}' for i in range(4)]))
    
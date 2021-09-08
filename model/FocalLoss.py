# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:30:45 2021

@author: shujie.wang
"""

import tensorflow.keras.backend as K
import tensorflow as tf

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*(1-alpha) + (K.ones_like(y_true)-y_true)*alpha
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

if __name__ == '__main__':
    import pickle as pkl
    import numpy as np
    import gc
    from deepfm import build_FM
    from tensorflow.keras import optimizers,initializers
    from lr_cosine import CosineAnnealing
    
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
    
    loss = binary_focal_loss(gamma=2, alpha=0.1)
    reduce_lr = CosineAnnealing(eta_max = 1,eta_min = 0,
                num_step_per_epoch=(train_num//10240)+1,lr_list = [1,2,2])
    adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(adam, loss = loss ,metrics = [tf.keras.metrics.AUC()],)
    
    history = model.fit(train_model_input, train_labels,validation_data = (val_model_input,val_labels),
                        batch_size=10240, epochs=5, verbose=1,callbacks=[reduce_lr],)
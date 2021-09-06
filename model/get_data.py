# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 23:35:55 2021

@author: tunan
"""

import numpy as np
import pandas as pd
import os
from time import time
import pickle as pkl
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import copy

def split(x):
    if not isinstance(x,str):
        return []
    key_ans = x.strip().split(';')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))

def preprocess(sample,dense_features):
    '''
    特征工程：对数值型特征取对数;对id型特征+1;缺失值补充0。
    '''
    sample[dense_features] = sample[dense_features].fillna(0.0)
    sample[dense_features] = np.log(sample[dense_features] + 1.0)
    
    sample[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    sample["videoplayseconds"] = np.log(sample["videoplayseconds"] + 1.0)
    sample[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        sample[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
    return sample


if __name__ == "__main__":    
    target = ["read_comment", "like", "click_avatar", "forward"]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    varlen_features = ['manual_tag_list','manual_keyword_list']
    dense_features = ['videoplayseconds']
    data = pd.read_csv('../data/wechat_algo_data1/user_action.csv')
    test = pd.read_csv('../data/wechat_algo_data1/test_a.csv') #预测数据
    test['date_'] = 15
    data = pd.concat([data,test])
    
    #1. merge features to data
    feed = pd.read_csv('../data/wechat_algo_data1/feed_info.csv') #视频特征。
    feed = feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id','manual_tag_list','manual_keyword_list']]
    data = data.merge(feed, how='left',on='feedid') #行为数据拼接，作者id，bgm_song_id 
    data = preprocess(data,dense_features) #特征处理
    data = data[dense_features+sparse_features+varlen_features+['date_']+target]
    
    #2. varlen features encode
    encoder = {}
    global key2index
    for f in ['manual_keyword_list','manual_tag_list']:
        key2index = {}
        f_list = list(map(split, data[f].values))
        f_length = np.array(list(map(len, f_list)))
        max_len = max(f_length)
        print(f'{f} max length is {max_len}')
        # Notice : padding=`post`
        data[f] = list(pad_sequences(f_list, maxlen=max_len, padding='post', ))
        encoder[f] = copy.copy(key2index)
    
    # 3.sparse feature encode
    for featid in sparse_features:
        print(f"encode {featid} feature id")
        encoder[featid] = {uid:ucode+1 for ucode,uid in enumerate(data[featid].unique())} 
        data[featid] = data[featid].apply(lambda x: encoder[featid].get(x,0))
        
    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())
    data = data.sample(frac = 1.0)

    train = data[data['date_'] < 14].drop(['date_'],axis = 1)
    val = data[data['date_'] == 14].drop(['date_'],axis = 1)  # 第14天样本作为验证集
    test = data[data['date_'] == 15].drop(['date_'],axis = 1)
    with open('../user_data/data.pkl','wb') as f:
        pkl.dump([train,val,test,encoder],f)

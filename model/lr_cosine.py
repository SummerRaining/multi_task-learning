from keras import *
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras.backend as K

class CosineAnnealing(callbacks.Callback):
    """Cosine annealing according to DECOUPLED WEIGHT DECAY REGULARIZATION.

    # Arguments
        eta_max: float, eta_max in eq(5).
        eta_min: float, eta_min in eq(5).
        total_iteration: int, Ti in eq(5).
        iteration: int, T_cur in eq(5).
        verbose: 0 or 1.
    """

    def __init__(self, eta_max=1, eta_min=0, num_step_per_epoch = 100,lr_list = [],verbose=0, **kwargs):
        
        super(CosineAnnealing, self).__init__()

        global lr_log
        
        self.lr_list = lr_list
        lr_log = []
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        
        self.iteration = 0
        self.cur_epoch = 0
        self.num_start = 0
        self.total_epoch = lr_list[self.num_start]
        self.num_step_per_epoch = num_step_per_epoch
        self.total_iteration = self.total_epoch*num_step_per_epoch
    
    def on_train_begin(self, logs=None):
        self.lr = K.get_value(self.model.optimizer.lr)
        #防止多个epoch分开训练。
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * 0.5 * (1 + np.cos(np.pi * self.iteration / self.total_iteration))
        new_lr = self.lr * eta_t
        K.set_value(self.model.optimizer.lr, new_lr)
        
    def on_train_end(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.lr)
    
    def on_epoch_end(self, epoch, logs=None):
        self.cur_epoch += 1
        if self.cur_epoch == self.total_epoch:
            self.cur_epoch = 0
            self.num_start += 1
            self.total_epoch = self.lr_list[min(self.num_start,len(self.lr_list)-1)]
            
            self.iteration = 0
            self.total_iteration = self.total_epoch*self.num_step_per_epoch

    def on_batch_end(self, epoch, logs=None):
        self.iteration += 1
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * 0.5 * (1 + np.cos(np.pi * self.iteration / self.total_iteration))
        new_lr = self.lr * eta_t
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealing '
                  'learning rate to %s.' % (epoch + 1, new_lr))
        lr_log.append(logs['lr'])

if __name__ == '__main__':
    # 准备数据集
    num_train, num_test = 2000, 100
    num_features = 200
    
    true_w, true_b = np.ones((num_features, 1)) * 0.01, 0.05
    
    features = np.random.normal(0, 1, (num_train + num_test, num_features))
    noises = np.random.normal(0, 1, (num_train + num_test, 1)) * 0.01
    labels = np.dot(features, true_w) + true_b + noises
    
    train_data, test_data = features[:num_train, :], features[num_train:, :]
    train_labels, test_labels = labels[:num_train], labels[num_train:]
    
    # 选择模型
    model = keras.models.Sequential([
        layers.Dense(units=128, activation='relu', input_dim=200), 
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00)),
        layers.Dense(1)
    ])
    
    model.summary()
    model.compile(optimizer='adam',loss='mse',metrics=['mse'])
    #需要传入参数，max，min。lr会在max和min之间衰减，乘上原来的lr。
    #num_step_per_epoch每个epoch会训练多少次。
    #lr_list每次的重启周期，例如这里2个epoch是一个周期，4个epoch一个周期，8，15，32.等。
    
    lr_list = [2,4,8,16,32]
    reduce_lr = CosineAnnealing(eta_max=1, eta_min=0, num_step_per_epoch=(2000 // 16), lr_list = lr_list)
    # for e in range(62):
        # model.fit(train_data, train_labels, batch_size=16, epochs=1, validation_data=(test_data, test_labels), callbacks=[reduce_lr])
    model.fit(train_data, train_labels, batch_size=16, epochs=62, validation_data=(test_data, test_labels), callbacks=[reduce_lr])
    plt.plot(lr_log)
    

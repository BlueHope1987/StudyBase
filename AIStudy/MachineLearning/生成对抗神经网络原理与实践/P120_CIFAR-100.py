#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
CIFAR-100与CIFAR-10的数据格式非常类似，只不过CIFAR-100数据集包含100个类别归纳到20个超级类中，每个类别包含600张训练图片（500张训练集和100张测试集）
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # TF2

#读取CIFAR-100样本数据
(x_train,y_labels),(x_test,y_test)=tf.keras.datasets.cifar100.load_data()

for i in range(10):

    #按类别筛选
    label_idx=np.where(y_labels==i)
    samples=x_train[label_idx[0]]

    #随机选取10张
    idx=np.random.randint(0,500,10)
    samples=samples[idx]

    for j in range(10):
        #10行、10列
        plt.subplot(10,10,i*10+j+1)

        #差值算法（最近邻差值）
        plt.imshow(samples[j],interpolation='nearest')
        plt.axis('off')

#plt.savefig('cifar10.jpg',format='jpg')
#plt.close('all')
plt.show()


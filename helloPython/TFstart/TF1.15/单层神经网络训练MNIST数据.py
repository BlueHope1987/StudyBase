#TensorFlow官方文档样例——单层神经网络训练MNIST数据
#https://blog.csdn.net/hwl19951007/article/details/81115341

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
 
 
# 从TensorFlow的样例中获取下载mnist的文件，并解压输入
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
# 展示数据集信息
print("训练集数量：{}个，验证集数量：{}个， 测试集数量：{}个".format(mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples))
print("数据集格式：{}，标签格式：{}".format(mnist.train.images[0].shape, mnist.train.labels[0].shape))
 
# 定义迭代次数以及随机抽取批次大小
NUM_STEPS = 10000
MINIBATCH_SIZE = 64
 
x = tf.placeholder(tf.float32, [None, 784])     # 初始化输入图像矩阵x
W = tf.Variable(tf.zeros([784, 10]))            # 初始化权重矩阵w
b = tf.Variable(tf.zeros([10]))                 # 初始化偏置矩阵b
y_labels = tf.placeholder("float", [None, 10])  # 初始化标签矩阵y_labels
 
y = tf.nn.softmax(tf.matmul(x, W) + b)          # 调用激活函数softmax计算预测值
cross_entropy = -tf.reduce_sum(y_labels*tf.log(y))    # 以交叉熵作为损失函数
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)  # 调用Adma算法，最小化损失函数
 
# 启动模型，初始化变量
with tf.Session() as sess:
 
    # 迭代训练模型
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = mnist.train.next_batch(MINIBATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_xs, y_labels: batch_ys})
 
        if _ % 1000 == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))    # 计算预测准确的数量
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))     # 计算准确率
            validation_accuary = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_labels: mnist.validation.labels})  # 计算验证集准确率
            test_accuary = sess.run(accuracy, feed_dict={x: mnist.test.images, y_labels: mnist.test.labels})  # 计算测试集准确率
            print("训练次数为{}次时，验证集准确率为：{:.4}%，测试集准确率为：{:.4}%，".format(_, validation_accuary*100, test_accuary * 100))  # 输出测试集准确率
 
print("训练完成，本次共训练{}次，最终测试集准确率为：{:.4}%".format(NUM_STEPS, test_accuary * 100))   # 输出测试集准确率
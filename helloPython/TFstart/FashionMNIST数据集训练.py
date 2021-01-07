#【tensorflow2.0】fashion mnist 数据集训练
#https://www.guyuehome.com/19898

import tensorflow as tf
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
print(tf.__version__)
print(tf.test.is_gpu_available())
# 加载mnist数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_all, Y_train_all),(X_test, Y_test) = fashion_mnist.load_data()
X_train_all = X_train_all/255
X_test = X_test/255
# 将训练集拆分出验证集,让模型每跑完一次数据就验证一次准确度
x_valid, x_train  = X_train_all[:5000], X_train_all[5000:]
y_valid, y_train  = Y_train_all[:5000], Y_train_all[5000:]
# 模型构建 使用的是tf.keras.Sequential
# relu：y=max(0,x) 即取0和x中的最大值
# softmax: 将输出向量变成概率分布，例如 x = [x1, x2, x3], 则
#                                     y = [e^x1/sum, e^x2/sum, e^x3/sum],
#                                     sum = e^x1+e^x2+e^x3
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)), # Flatten函数的作用是将输入的二维数组进行展开，使其变成一维的数组
        tf.keras.layers.Dense(256,activation='relu'), # 创建权连接层，激活函数使用relu
        tf.keras.layers.Dropout(0.2),                 # 使用dropout缓解过拟合的发生
        tf.keras.layers.Dense(10, activation='softmax') # 输出层
    ]
)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # 损失函数使用交叉熵
              metrics=['accuracy'])
model.summary() # 打印模型信息
# history记录模型训练过程中的一些值
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_valid,y_valid))
print('history:',history.history)
# 将history中的数据以图片表示出来
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.ylim(0,1)
plt.show()
model.evaluate(X_test,  Y_test, verbose=2)
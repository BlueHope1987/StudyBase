#使用tensorflow实现mnist手写识别(单层神经网络实现)
#https://www.cnblogs.com/imae/p/10629890.html

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
mnist =  input_data.read_data_sets("data/",one_hot = True)
#导入Tensorflwo和mnist数据集

#构建输入层
x = tf.placeholder(tf.float32,[None,784],name='X')
y = tf.placeholder(tf.float32,[None,10],name='Y')

#隐藏层神经元数量
H1_NN = 256 #第一层神经元数量
W1 = tf.Variable(tf.random_normal([784,H1_NN])) #权重
b1 = tf.Variable(tf.zeros([H1_NN])) #偏置项
Y1 = tf.nn.relu(tf.matmul(x,W1)+b1) #第一层输出
W2 = tf.Variable(tf.random_normal([H1_NN,10]))#权重
b2 = tf.Variable(tf.zeros(10))#偏置项

forward = tf.matmul(Y1,W2)+b2 #定义前向传播
pred = tf.nn.softmax(forward) #激活函数输出

#损失函数
#loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),
#                                            reduction_indices=1))
#(log(0))超出范围报错

loss_function  = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=y))

#训练参数
train_epochs = 50 #训练次数
batch_size = 50 #每次训练多少个样本
total_batch = int(mnist.train.num_examples/batch_size) #随机抽取样本
display_step = 1 #训练情况输出
learning_rate = 0.01 #学习率

#优化器
opimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

#准确率函数
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#记录开始训练时间
from time import time
startTime = time()
#初始化变量
sess =tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs,ys = mnist.train.next_batch(batch_size)#读取批次数据
        sess.run(opimizer,feed_dict={x:xs,y:ys})#执行批次数据训练
    
    #total_batch个批次训练完成后，使用验证数据计算误差与准确率
    loss,acc=sess.run([loss_function,accuracy],
                     feed_dict={x:mnist.validation.images,
                                y:mnist.validation.labels})
    #输出训练情况
    if(epoch+1) % display_step == 0:
        print("Train Epoch:",'%02d' % (epoch + 1),
               "Loss=","{:.9f}".format(loss),"Accuracy=","{:.4f}".format(acc))
duration = time()-startTime
print("Trian Finshed takes:","{:.2f}".format(duration))#显示预测耗时

#由于pred预测结果是one_hot编码格式，所以需要转换0~9数字
prediction_resul =  sess.run(tf.argmax(pred,1),feed_dict={x:mnist.test.images})

prediction_resul[0:10]

#模型评估
accu_test = sess.run(accuracy,
                    feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("Accuray:",accu_test)



compare_lists = prediction_resul == np.argmax(mnist.test.labels,1)
print(compare_lists)
err_lists = [i for i in range(len(mnist.test.labels)) if compare_lists[i] == False]
print(err_lists,len(err_lists))

index_list = []
def print_predct_errs(labels,#标签列表
                        perdiction):#预测值列表
    count = 0
    compare_lists = (perdiction == np.argmax(labels,1))
    err_lists = [i for i in range(len(labels)) if compare_lists[i] == False]
    for x in err_lists:
        index_list.append(x)
        print("index="+str(x)+
        "标签值=",np.argmax(labels[x]),
        "预测值=",perdiction[x])
        count = count+1
    print("总计:",count)
    return index_list

print_predct_errs(mnist.test.labels,prediction_resul) 

def plot_images_labels_prediction(images,labels,prediction,index,num=25):
    fig = plt.gcf() # 获取当前图片
    fig.set_size_inches(10,12)
    if num>=25:
        num=25 #最多显示25张图片
    for i in range(0,num):
        ax = plt.subplot(5,5, i+1) #获取当前要处理的子图
        
        ax.imshow(np.reshape(images[index],(28,28)),cmap='binary')#显示第index个图像
        title = 'label=' + str(np.argmax(labels[index]))#构建该图上要显示的title
        if len(prediction)>0:
            title += 'predict= '+str(prediction[index])
        
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1
    plt.show()

plot_images_labels_prediction(mnist.test.images,mnist.test.labels,prediction_resul,index=index_list[100])